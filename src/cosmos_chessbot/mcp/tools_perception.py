"""Board perception tools."""

from __future__ import annotations

import json

import numpy as np
from mcp.server.fastmcp import Context

from .server import mcp
from .state import ServerState


@mcp.tool()
def detect_board_state(ctx: Context) -> str:
    """Detect the current chess board position using YOLO-DINO-MLP.

    Captures from the overhead camera, runs piece detection, and returns
    the FEN string with per-piece metadata. The robot arm should be parked
    out of the camera view first.

    Returns JSON with detected FEN, piece list, confidence, and the
    internal board FEN for comparison.
    """
    state: ServerState = ctx.request_context.lifespan_context
    try:
        img = state.overhead_camera.capture()
        img_np = np.array(img)

        result = state.yolo_detector.detect_fen_with_metadata(img_np)

        result["internal_fen"] = state.board.fen()
        result["turn"] = "white" if state.board.turn else "black"
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def calibrate_board(ctx: Context) -> str:
    """Detect the 4 board corners and initialize pixel-to-world calibration.

    Must be called before execute_trajectory or pixel_to_world.
    The arm should be parked so it doesn't occlude the board corners.

    Returns the detected corner pixel positions and calibration status.
    """
    state: ServerState = ctx.request_context.lifespan_context
    try:
        img = state.overhead_camera.capture()
        img_np = np.array(img)

        corners = state.yolo_detector._detect_corners(img_np)
        if corners is None:
            return json.dumps({"success": False, "error": "Board corners not detected"})

        from ..utils.pixel_to_board import BoardCalibration

        # YOLO returns corners as [TL, TR, BR, BL].
        # BoardCalibration expects [a1, h1, h8, a8] which maps to [BL, BR, TR, TL].
        pixel_corners = [
            tuple(corners[3].tolist()),  # a1 = BL
            tuple(corners[2].tolist()),  # h1 = BR
            tuple(corners[1].tolist()),  # h8 = TR
            tuple(corners[0].tolist()),  # a8 = TL
        ]

        h, w = img_np.shape[:2]
        state.board_calibration = BoardCalibration(
            pixel_corners=pixel_corners,
            image_size=(w, h),
            square_size=state.board_square_size,
            table_z=state.board_table_z,
            center_y=state.board_center_offset_y,
        )

        return json.dumps({
            "success": True,
            "corners_pixel": {
                "a1": list(pixel_corners[0]),
                "h1": list(pixel_corners[1]),
                "h8": list(pixel_corners[2]),
                "a8": list(pixel_corners[3]),
            },
            "image_size": [w, h],
            "square_size_m": state.board_square_size,
        }, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
