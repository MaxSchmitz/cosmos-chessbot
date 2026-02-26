"""π₀.₅ Vision-Language-Action policy — remote inference via openpi server."""

import functools
import logging
import time
from typing import Optional

import msgpack
import numpy as np
from PIL import Image
import websockets.sync.client

from .base_policy import BasePolicy, PolicyAction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimal msgpack-numpy (vendored from openpi-client)
# ---------------------------------------------------------------------------

def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_Packer = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)

# ---------------------------------------------------------------------------
# SO-101 ↔ DROID observation mapping
# ---------------------------------------------------------------------------

# DROID expects 7 joints + 1 gripper; SO-101 has 5 joints + 1 gripper.
# We zero-pad joints 6-7 when sending and truncate actions on return.
SO101_ARM_JOINTS = 5
DROID_ARM_JOINTS = 7
DROID_ACTION_DIM = 8  # 7 joints + 1 gripper


class PI05Policy(BasePolicy):
    """π₀.₅ policy using remote openpi WebSocket server.

    The openpi server (serve_policy.py) runs on a GPU machine.
    This client sends observations over WebSocket and gets action chunks back.

    Usage:
        # Start server on brev:
        #   cd ~/openpi && uv run scripts/serve_policy.py --env=DROID --port=8001
        # SSH tunnel from local:
        #   ssh -f -N -L 8001:localhost:8001 ubuntu@isaacsim

        policy = PI05Policy(host="localhost", port=8001)
        action = policy.select_action(
            images={"overhead": overhead_img, "wrist": wrist_img},
            robot_state=np.array([...]),  # 6 values: 5 joints + 1 gripper
            instruction="pick up the pawn on e2",
        )
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8001,
        timeout: float = 30.0,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._ws = None
        self._packer = _Packer()
        self._connect()

    def _connect(self):
        """Connect to the openpi WebSocket server."""
        uri = f"ws://{self.host}:{self.port}"
        logger.info(f"Connecting to pi0.5 server at {uri}...")
        retries = 0
        while True:
            try:
                self._ws = websockets.sync.client.connect(
                    uri, compression=None, max_size=None
                )
                # Server sends metadata on connect
                metadata = _unpackb(self._ws.recv())
                logger.info(f"Connected to pi0.5 server. Metadata: {metadata}")
                return
            except (ConnectionRefusedError, OSError) as e:
                retries += 1
                if retries > 5:
                    raise ConnectionError(
                        f"Cannot connect to pi0.5 server at {uri}. "
                        "Make sure the server is running and SSH tunnel is active."
                    ) from e
                logger.info(f"Waiting for server... (attempt {retries})")
                time.sleep(3)

    def _ensure_connected(self):
        """Reconnect if the WebSocket dropped."""
        if self._ws is None:
            self._connect()
        try:
            self._ws.ping()
        except Exception:
            logger.warning("WebSocket connection lost, reconnecting...")
            self._ws = None
            self._connect()

    def reset(self):
        """Reset policy state between episodes."""
        pass

    def _prepare_obs(
        self,
        images: dict[str, Image.Image],
        robot_state: np.ndarray,
        instruction: Optional[str] = None,
    ) -> dict:
        """Convert our observation format to DROID observation format.

        Maps:
            images["overhead"] → observation/exterior_image_1_left (224x224)
            images["wrist"]    → observation/wrist_image_left (224x224)
            robot_state[:5]    → observation/joint_position (zero-padded to 7)
            robot_state[5]     → observation/gripper_position
            instruction        → prompt
        """
        # Get images, falling back to available keys
        overhead_key = next(
            (k for k in ("overhead", "egocentric", "exterior") if k in images),
            next(iter(images)),
        )
        wrist_key = next(
            (k for k in ("wrist",) if k in images),
            None,
        )

        overhead_img = images[overhead_key].resize((224, 224))
        overhead_arr = np.array(overhead_img, dtype=np.uint8)

        if wrist_key and wrist_key in images:
            wrist_img = images[wrist_key].resize((224, 224))
            wrist_arr = np.array(wrist_img, dtype=np.uint8)
        else:
            wrist_arr = np.zeros((224, 224, 3), dtype=np.uint8)

        # Pad SO-101 joints (5) to DROID joints (7)
        state = np.asarray(robot_state, dtype=np.float64)
        if len(state) == SO101_ARM_JOINTS + 1:
            joint_pos = np.zeros(DROID_ARM_JOINTS, dtype=np.float64)
            joint_pos[:SO101_ARM_JOINTS] = state[:SO101_ARM_JOINTS]
            gripper_pos = np.array([state[SO101_ARM_JOINTS]], dtype=np.float64)
        else:
            # Assume already in DROID format
            joint_pos = state[:DROID_ARM_JOINTS]
            gripper_pos = np.array([state[DROID_ARM_JOINTS]], dtype=np.float64)

        obs = {
            "observation/exterior_image_1_left": overhead_arr,
            "observation/wrist_image_left": wrist_arr,
            "observation/joint_position": joint_pos,
            "observation/gripper_position": gripper_pos,
            "prompt": instruction or "pick up the chess piece",
        }
        return obs

    def _infer(self, obs: dict) -> dict:
        """Send observation to server and get action prediction."""
        self._ensure_connected()
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Server error: {response}")
        return _unpackb(response)

    def select_action(
        self,
        images: dict[str, Image.Image],
        robot_state: np.ndarray,
        instruction: Optional[str] = None,
    ) -> PolicyAction:
        """Select action given current observations.

        Args:
            images: Camera views {"overhead": img, "wrist": img}
            robot_state: [5 joints + 1 gripper] for SO-101
            instruction: Language instruction for pi0.5

        Returns:
            PolicyAction with (horizon, 6) actions for SO-101
        """
        obs = self._prepare_obs(images, robot_state, instruction)
        result = self._infer(obs)

        # result["actions"] is (horizon, 8) for DROID
        actions = np.array(result["actions"])

        # Map back to SO-101: take first 5 joints + last dim (gripper)
        so101_actions = np.zeros((actions.shape[0], SO101_ARM_JOINTS + 1))
        so101_actions[:, :SO101_ARM_JOINTS] = actions[:, :SO101_ARM_JOINTS]
        so101_actions[:, SO101_ARM_JOINTS] = actions[:, DROID_ARM_JOINTS]  # gripper

        return PolicyAction(
            actions=so101_actions,
            success_probability=1.0,
            metadata={
                "policy": "pi05_remote",
                "raw_actions_shape": actions.shape,
                "instruction": instruction,
                "timing": result.get("policy_timing", {}),
            },
        )

    def plan_action(
        self,
        images: dict[str, Image.Image],
        robot_state: np.ndarray,
        instruction: Optional[str] = None,
    ) -> list[PolicyAction]:
        """Pi0.5 doesn't support planning; returns single action."""
        return [self.select_action(images, robot_state, instruction)]
