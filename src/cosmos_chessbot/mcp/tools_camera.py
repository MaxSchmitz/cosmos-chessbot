"""Camera capture tools."""

from __future__ import annotations

import base64
import io
import json
import time

from mcp.server.fastmcp import Context, Image
from mcp.types import ImageContent, TextContent

from .server import mcp
from .state import ServerState


def _pil_to_base64_jpeg(img, quality: int = 85) -> str:
    """Encode a PIL Image as base64 JPEG."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


@mcp.tool()
def reconnect_cameras(ctx: Context) -> str:
    """Release and re-open both cameras to recover from USB failures.

    Call this when capture_overhead or capture_wrist return errors.
    Waits 2 seconds after reconnecting for cameras to stabilize.
    """
    state: ServerState = ctx.request_context.lifespan_context
    results = {}
    for name, cam in [("overhead", state.overhead_camera), ("wrist", state.wrist_camera)]:
        try:
            cam.reconnect()
            time.sleep(1.0)
            img = cam.capture()
            results[name] = {"status": "ok", "width": img.width, "height": img.height}
        except Exception as e:
            results[name] = {"status": "error", "error": str(e)}
    return json.dumps(results)


@mcp.tool()
def capture_overhead(ctx: Context) -> list:
    """Capture a frame from the overhead camera looking down at the chess board.

    Returns the image and its dimensions. Use this to see the full board state.
    The robot arm should be parked first so it doesn't occlude the board.
    """
    state: ServerState = ctx.request_context.lifespan_context
    try:
        img = state.overhead_camera.capture()
        return [
            ImageContent(
                type="image",
                data=_pil_to_base64_jpeg(img),
                mimeType="image/jpeg",
            ),
            TextContent(
                type="text",
                text=json.dumps({"width": img.width, "height": img.height, "camera": "overhead"}),
            ),
        ]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


@mcp.tool()
def capture_wrist(ctx: Context) -> list:
    """Capture a frame from the wrist camera mounted on the robot gripper.

    Returns a close-up view useful for verifying grasp and placement.
    """
    state: ServerState = ctx.request_context.lifespan_context
    try:
        img = state.wrist_camera.capture()
        return [
            ImageContent(
                type="image",
                data=_pil_to_base64_jpeg(img),
                mimeType="image/jpeg",
            ),
            TextContent(
                type="text",
                text=json.dumps({"width": img.width, "height": img.height, "camera": "wrist"}),
            ),
        ]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
