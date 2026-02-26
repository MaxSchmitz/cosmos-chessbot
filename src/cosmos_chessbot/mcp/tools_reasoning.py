"""Cosmos reasoning tools (forwarded to remote GPU server)."""

from __future__ import annotations

import json

from mcp.server.fastmcp import Context

from .server import mcp
from .state import ServerState


@mcp.tool()
def reason_action(
    ctx: Context,
    move_uci: str,
    from_square: str,
    to_square: str,
    use_wrist_camera: bool = True,
) -> str:
    """Pre-action physical reasoning about a chess move using Cosmos-Reason2.

    Analyzes obstacles, adjacent pieces, grasp strategy, and risks before
    executing a move. Captures fresh images from both cameras.

    Requires the remote Cosmos server to be running.

    Args:
        move_uci: UCI move (e.g. 'e2e4').
        from_square: Source square (e.g. 'e2').
        to_square: Target square (e.g. 'e4').
        use_wrist_camera: Include wrist camera view (default true).
    """
    state: ServerState = ctx.request_context.lifespan_context
    if state.game_reasoning is None:
        return json.dumps({"error": "Cosmos server not configured. Set CHESS_COSMOS_URL."})

    try:
        overhead = state.overhead_camera.capture()
        wrist = state.wrist_camera.capture() if use_wrist_camera else None

        result = state.game_reasoning.reason_about_action(
            image=overhead,
            move_uci=move_uci,
            from_square=from_square,
            to_square=to_square,
            wrist_image=wrist,
        )
        return json.dumps({
            "obstacles": result.obstacles,
            "adjacent_pieces": result.adjacent_pieces,
            "grasp_strategy": result.grasp_strategy,
            "trajectory_advice": result.trajectory_advice,
            "risks": result.risks,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def plan_trajectory(
    ctx: Context,
    move_uci: str,
    from_square: str,
    to_square: str,
    piece_type: str = "piece",
    use_wrist_camera: bool = True,
) -> str:
    """Plan a 2D pixel trajectory for a chess move using Cosmos Action CoT.

    Returns waypoints in normalized 0-1000 pixel coordinates with labels.
    If board calibration is active, also returns 3D world coordinates
    that can be passed directly to execute_trajectory.

    Requires the remote Cosmos server to be running.

    Args:
        move_uci: UCI move (e.g. 'e2e4').
        from_square: Source square (e.g. 'e2').
        to_square: Target square (e.g. 'e4').
        piece_type: Type of piece being moved (pawn, knight, bishop, rook, queen, king).
        use_wrist_camera: Include wrist camera view (default true).
    """
    state: ServerState = ctx.request_context.lifespan_context
    if state.game_reasoning is None:
        return json.dumps({"error": "Cosmos server not configured. Set CHESS_COSMOS_URL."})

    try:
        overhead = state.overhead_camera.capture()
        wrist = state.wrist_camera.capture() if use_wrist_camera else None

        result = state.game_reasoning.plan_trajectory(
            image=overhead,
            move_uci=move_uci,
            from_square=from_square,
            to_square=to_square,
            piece_type=piece_type,
            wrist_image=wrist,
        )

        waypoints_2d = [
            {"point_2d": list(wp.point_2d), "label": wp.label}
            for wp in result.waypoints
        ]

        # Convert to 3D if calibration is available
        waypoints_3d = None
        labels = [wp.label for wp in result.waypoints]
        if state.board_calibration and result.waypoints:
            coords = state.board_calibration.waypoints_to_3d(result.waypoints)
            waypoints_3d = [
                {"x": float(c[0]), "y": float(c[1]), "z": float(c[2])}
                for c in coords
            ]

        return json.dumps({
            "waypoints_2d": waypoints_2d,
            "waypoints_3d": waypoints_3d,
            "labels": labels,
            "move_uci": result.move_uci,
            "reasoning": result.reasoning,
            "confidence": result.confidence,
            "num_waypoints": len(result.waypoints),
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def verify_goal(
    ctx: Context,
    move_uci: str,
    from_square: str,
    to_square: str,
    piece_type: str = "piece",
    use_wrist_camera: bool = True,
) -> str:
    """Post-action visual verification using Cosmos-Reason2.

    Captures fresh images and checks whether the piece landed correctly,
    is stable, and no adjacent pieces were disturbed.

    Requires the remote Cosmos server to be running.

    Args:
        move_uci: The move that was just executed.
        from_square: Source square.
        to_square: Target square.
        piece_type: Type of piece moved.
        use_wrist_camera: Include wrist camera view (default true).
    """
    state: ServerState = ctx.request_context.lifespan_context
    if state.game_reasoning is None:
        return json.dumps({"error": "Cosmos server not configured. Set CHESS_COSMOS_URL."})

    try:
        overhead = state.overhead_camera.capture()
        wrist = state.wrist_camera.capture() if use_wrist_camera else None

        result = state.game_reasoning.verify_goal(
            image=overhead,
            move_uci=move_uci,
            from_square=from_square,
            to_square=to_square,
            piece_type=piece_type,
            wrist_image=wrist,
        )
        return json.dumps({
            "success": result.success,
            "reason": result.reason,
            "physical_issues": result.physical_issues,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
