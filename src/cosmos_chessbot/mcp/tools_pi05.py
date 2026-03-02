"""Pi0.5 VLA policy tools -- closed-loop control via remote server."""

from __future__ import annotations

import functools
import json
import logging
import time

import msgpack
import numpy as np
from mcp.server.fastmcp import Context

from .server import mcp
from .state import ServerState

logger = logging.getLogger("cosmos-mcp.pi05")

# ---------------------------------------------------------------------------
# msgpack-numpy helpers (same as pi05_policy.py / serve_pi05.py)
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


def _connect_pi05(url: str):
    """Connect to pi0.5 WebSocket server. Returns (ws, metadata)."""
    import websockets.sync.client

    ws = websockets.sync.client.connect(url, compression=None, max_size=None)
    metadata = _unpackb(ws.recv())
    return ws, metadata


@mcp.tool()
def run_pi05_episode(
    ctx: Context,
    task: str,
    num_steps: int = 50,
    step_delay: float = 0.05,
    source: str | None = None,
    target: str | None = None,
) -> str:
    """Run a closed-loop pi0.5 episode on the robot.

    Captures fresh camera images and robot state at each step, sends to
    the pi0.5 server for inference, and executes the returned action.

    When source and target are provided, a HUD overlay is drawn on the
    overhead image each step (green=pick, magenta=place). Board corners
    are auto-detected from the first frame via YOLO pose.

    Args:
        task: Language instruction (e.g. "Pick the piece at e2 and place it at e4").
        num_steps: Number of control steps to execute (default 50).
        step_delay: Delay between steps in seconds (default 0.05).
        source: Optional source location for HUD overlay (square name or "x,y").
        target: Optional target location for HUD overlay (square name or "x,y").
    """
    state: ServerState = ctx.request_context.lifespan_context
    if state.robot is None:
        return json.dumps({"error": "Robot not connected"})
    if not state.pi05_url:
        return json.dumps({"error": "Pi0.5 server URL not configured (set CHESS_PI05_URL)"})

    try:
        use_hud = source and target
        hud_corners = None
        hud_H = None

        if use_hud:
            task = "Pick up the highlighted piece and place it at the target"
            logger.info("HUD enabled: source=%s target=%s", source, target)

        # Connect to pi0.5 server
        logger.info("Connecting to pi0.5 server at %s", state.pi05_url)
        ws, metadata = _connect_pi05(state.pi05_url)
        packer = _Packer()
        logger.info("Pi0.5 connected: %s", metadata)

        # Reset policy state for new episode
        ws.send(packer.pack({"command": "reset"}))
        reset_resp = _unpackb(ws.recv())
        logger.info("Pi0.5 reset: %s", reset_resp)

        trajectory = []
        from .tools_robot import _read_joints, _send_joints

        for step_i in range(num_steps):
            t0 = time.time()

            # 1. Read current robot state
            joints = _read_joints(state)

            # 2. Capture camera images
            overhead_img = state.overhead_camera.capture()
            wrist_img = state.wrist_camera.capture()
            overhead_arr = np.array(overhead_img, dtype=np.uint8)
            wrist_arr = np.array(wrist_img, dtype=np.uint8)

            # 2b. Apply HUD overlay -- detect corners from first frame
            if use_hud:
                from ..vision.hud_overlay import apply_hud, compute_homography, detect_corners
                if hud_corners is None:
                    hud_corners = detect_corners(overhead_arr)
                    if hud_corners is not None:
                        hud_H = compute_homography(hud_corners)
                        logger.info("HUD: detected board corners from first frame")
                    else:
                        logger.warning("HUD: could not detect board corners")
                apply_hud(overhead_arr, source, target, hud_corners, hud_H)

            # 3. Build observation in lerobot-native format
            obs = {
                "observation.images.egocentric": overhead_arr,
                "observation.images.wrist": wrist_arr,
                "observation.state": joints,
                "task": task,
            }

            # 4. Send to server and get action
            ws.send(packer.pack(obs))
            response = ws.recv()
            result = _unpackb(response)
            action = np.array(result["action"], dtype=np.float32)

            # 5. Execute action on robot
            _send_joints(state, action)

            dt = time.time() - t0
            trajectory.append({
                "step": step_i,
                "state": joints.tolist(),
                "action": action.tolist(),
                "dt": round(dt, 3),
            })

            if step_i < 3 or step_i % 10 == 0:
                logger.info(
                    "Step %d: pan=%.1f lift=%.1f elbow=%.1f wrist=%.1f roll=%.1f grip=%.1f (%.3fs)",
                    step_i, *action[:6], dt,
                )

            time.sleep(step_delay)

        ws.close()

        # Summary
        actions = np.array([t["action"] for t in trajectory])
        summary = {
            "success": True,
            "steps_executed": len(trajectory),
            "task": task,
            "joint_ranges": {
                "pan": [float(actions[:, 0].min()), float(actions[:, 0].max())],
                "lift": [float(actions[:, 1].min()), float(actions[:, 1].max())],
                "elbow": [float(actions[:, 2].min()), float(actions[:, 2].max())],
                "wrist": [float(actions[:, 3].min()), float(actions[:, 3].max())],
                "roll": [float(actions[:, 4].min()), float(actions[:, 4].max())],
                "gripper": [float(actions[:, 5].min()), float(actions[:, 5].max())],
            },
            "first_action": trajectory[0]["action"],
            "last_action": trajectory[-1]["action"],
            "final_state": trajectory[-1]["state"],
        }
        return json.dumps(summary)

    except Exception as e:
        import traceback
        logger.error("Pi0.5 episode failed: %s\n%s", e, traceback.format_exc())
        return json.dumps({"error": str(e)})


@mcp.tool()
def pi05_step(
    ctx: Context,
    task: str,
    execute: bool = True,
    source: str | None = None,
    target: str | None = None,
) -> str:
    """Run a single pi0.5 inference step.

    Captures current camera images and robot state, sends to pi0.5 server,
    and optionally executes the returned action.

    Args:
        task: Language instruction for pi0.5.
        execute: Whether to execute the returned action (default true).
        source: Optional source location for HUD overlay (square name or "x,y").
        target: Optional target location for HUD overlay (square name or "x,y").
    """
    state: ServerState = ctx.request_context.lifespan_context
    if state.robot is None:
        return json.dumps({"error": "Robot not connected"})
    if not state.pi05_url:
        return json.dumps({"error": "Pi0.5 server URL not configured (set CHESS_PI05_URL)"})

    try:
        if source and target:
            task = "Pick up the highlighted piece and place it at the target"

        ws, metadata = _connect_pi05(state.pi05_url)
        packer = _Packer()

        # Reset for fresh prediction
        ws.send(packer.pack({"command": "reset"}))
        _unpackb(ws.recv())

        from .tools_robot import _read_joints, _send_joints

        joints = _read_joints(state)
        overhead_arr = np.array(state.overhead_camera.capture(), dtype=np.uint8)
        wrist_arr = np.array(state.wrist_camera.capture(), dtype=np.uint8)

        # Apply HUD overlay -- auto-detects corners from image
        if source and target:
            from ..vision.hud_overlay import apply_hud
            apply_hud(overhead_arr, source, target)

        obs = {
            "observation.images.egocentric": overhead_arr,
            "observation.images.wrist": wrist_arr,
            "observation.state": joints,
            "task": task,
        }

        ws.send(packer.pack(obs))
        result = _unpackb(ws.recv())
        action = np.array(result["action"], dtype=np.float32)

        if execute:
            _send_joints(state, action)

        ws.close()

        return json.dumps({
            "success": True,
            "current_state": joints.tolist(),
            "predicted_action": action.tolist(),
            "executed": execute,
            "inference_time": result.get("inference_time", 0),
        })

    except Exception as e:
        return json.dumps({"error": str(e)})
