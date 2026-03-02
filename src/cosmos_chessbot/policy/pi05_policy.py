"""pi0.5 Vision-Language-Action policy -- remote inference via WebSocket server."""

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

# SO-101: 5 arm joints + 1 gripper = 6 total
SO101_ACTION_DIM = 6


class PI05Policy(BasePolicy):
    """pi0.5 policy using remote WebSocket server with lerobot-native format.

    The server (scripts/serve_pi05.py) runs on a GPU machine and loads
    a fine-tuned pi0.5 checkpoint trained with lerobot.

    Usage:
        # Start server on brev:
        #   cd ~/cosmos-chessbot && uv run python scripts/serve_pi05.py \
        #       --checkpoint outputs/pi05_chess/checkpoints/last/pretrained_model --port 8001
        # SSH tunnel from local:
        #   ssh -f -N -L 8001:localhost:8001 ubuntu@isaacsim

        policy = PI05Policy(host="localhost", port=8001)
        action = policy.select_action(
            images={"egocentric": overhead_img, "wrist": wrist_img},
            robot_state=np.array([...]),  # 6 values: 5 joints + 1 gripper
            instruction="Pick the piece at e2 and place it at e4",
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
        """Connect to the pi0.5 WebSocket server."""
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
        """Reset policy state between episodes (clears server-side action queue)."""
        self._ensure_connected()
        data = self._packer.pack({"command": "reset"})
        self._ws.send(data)
        response = _unpackb(self._ws.recv())
        logger.info(f"Policy reset: {response}")

    def _prepare_obs(
        self,
        images: dict[str, Image.Image],
        robot_state: np.ndarray,
        instruction: Optional[str] = None,
    ) -> dict:
        """Convert observations to lerobot-native format.

        Maps:
            images["egocentric"/"overhead"] -> observation.images.egocentric (480x640)
            images["wrist"]                 -> observation.images.wrist (480x640)
            robot_state (6-dim)             -> observation.state
            instruction                     -> task
        """
        # Get images, falling back to available keys
        overhead_key = next(
            (k for k in ("egocentric", "overhead", "exterior") if k in images),
            next(iter(images)),
        )
        wrist_key = next(
            (k for k in ("wrist",) if k in images),
            None,
        )

        # Send at training resolution (480x640) -- preprocessor handles resize to 224x224
        overhead_img = images[overhead_key]
        overhead_arr = np.array(overhead_img, dtype=np.uint8)

        if wrist_key and wrist_key in images:
            wrist_img = images[wrist_key]
            wrist_arr = np.array(wrist_img, dtype=np.uint8)
        else:
            # Use zeros if no wrist camera available
            h, w = overhead_arr.shape[:2]
            wrist_arr = np.zeros((h, w, 3), dtype=np.uint8)

        state = np.asarray(robot_state, dtype=np.float32)

        obs = {
            "observation.images.egocentric": overhead_arr,
            "observation.images.wrist": wrist_arr,
            "observation.state": state,
            "task": instruction or "pick up the chess piece",
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
            images: Camera views {"egocentric": img, "wrist": img}
            robot_state: [5 joints + 1 gripper] for SO-101
            instruction: Language instruction for pi0.5

        Returns:
            PolicyAction with single (6,) action for SO-101
        """
        obs = self._prepare_obs(images, robot_state, instruction)
        result = self._infer(obs)

        # Server returns single action (6,) from its internal chunk queue
        action = np.array(result["action"], dtype=np.float32)

        return PolicyAction(
            actions=action.reshape(1, -1),  # (1, 6) for consistency
            success_probability=1.0,
            metadata={
                "policy": "pi05_finetuned",
                "instruction": instruction,
                "inference_time": result.get("inference_time", 0),
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
