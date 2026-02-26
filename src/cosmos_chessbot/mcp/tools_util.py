"""Utility tools."""

from __future__ import annotations

import json

from mcp.server.fastmcp import Context

from .server import mcp
from .state import ServerState


@mcp.tool()
def pixel_to_world(ctx: Context, px: float, py: float) -> str:
    """Convert image pixel coordinates to 3D world coordinates on the board plane.

    Requires calibrate_board to have been called first.

    Args:
        px: Pixel X coordinate in image space.
        py: Pixel Y coordinate in image space.
    """
    state: ServerState = ctx.request_context.lifespan_context
    if state.board_calibration is None:
        return json.dumps({"error": "Board not calibrated. Call calibrate_board first."})

    try:
        x, y, z = state.board_calibration.pixel_to_board(px, py)
        return json.dumps({"world_x": x, "world_y": y, "world_z": z})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def compare_fen(ctx: Context, expected_fen: str, actual_fen: str) -> str:
    """Compare two FEN strings and report square-by-square differences.

    Useful for verifying move execution by comparing expected vs detected state.

    Args:
        expected_fen: The FEN you expect after a move.
        actual_fen: The FEN detected by perception.
    """
    try:
        from ..reasoning.fen_comparison import compare_fen_states

        result = compare_fen_states(expected_fen, actual_fen)
        return json.dumps({
            "match": result.match,
            "num_differences": len(result.differences),
            "differences": [
                {
                    "square": d.square,
                    "expected": d.expected_piece,
                    "actual": d.actual_piece,
                }
                for d in result.differences
            ],
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
