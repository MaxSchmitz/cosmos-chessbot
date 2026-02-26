"""Shared server state container for the MCP server.

All long-lived objects (cameras, models, robot, etc.) live here.
Initialized once during server lifespan, accessed by all tool modules.
"""

from __future__ import annotations

import chess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..policy.waypoint_policy import WaypointPolicy
    from ..stockfish.engine import StockfishEngine
    from ..utils.pixel_to_board import BoardCalibration
    from ..vision.camera import Camera
    from ..vision.yolo_dino_detector import YOLODINOFenDetector


@dataclass
class ServerState:
    """Container for all MCP server state."""

    # Cameras
    overhead_camera: Camera
    wrist_camera: Camera

    # Perception
    yolo_detector: YOLODINOFenDetector

    # Chess engine
    engine: StockfishEngine
    board: chess.Board = field(default_factory=chess.Board)

    # Robot (may not be connected)
    robot: Optional[Any] = None
    joint_names: list[str] = field(default_factory=lambda: [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ])
    home_position: dict[str, float] = field(default_factory=lambda: {
        "shoulder_pan.pos": 7.55,
        "shoulder_lift.pos": -98.12,
        "elbow_flex.pos": 100.00,
        "wrist_flex.pos": 62.88,
        "wrist_roll.pos": 0.08,
        "gripper.pos": 1.63,
    })
    park_position: dict[str, float] = field(default_factory=lambda: {
        "shoulder_pan.pos": 90.0,
        "shoulder_lift.pos": -98.12,
        "elbow_flex.pos": 100.00,
        "wrist_flex.pos": 62.88,
        "wrist_roll.pos": 0.08,
        "gripper.pos": 1.63,
    })

    # Board calibration (set by calibrate_board tool)
    board_calibration: Optional[BoardCalibration] = None

    # Cosmos reasoning (remote HTTP client)
    game_reasoning: Optional[Any] = None

    # Policy
    waypoint_policy: Optional[WaypointPolicy] = None

    # Board geometry config
    board_square_size: float = 0.05
    board_center_offset_y: float = 0.20
    board_table_z: float = 0.0
