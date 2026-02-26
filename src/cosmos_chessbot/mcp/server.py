"""MCP server for direct robot control.

Exposes cameras, perception, chess engine, Cosmos reasoning, and robot
manipulation as individual tools that Claude can call to close the
perception-reasoning-action loop.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

import chess
from mcp.server.fastmcp import FastMCP

from .state import ServerState

# All logging to stderr -- stdout is reserved for MCP stdio transport
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(name)s %(message)s")
logger = logging.getLogger("cosmos-mcp")


def _connect_robot(port: str):
    """Connect to SO-101 via lerobot. Returns robot or None."""
    try:
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

        robot = SO101Follower(SO101FollowerConfig(
            port=port,
            id="my_follower_arm",
        ))
        robot.connect()
        logger.info("Robot connected on %s", port)
        return robot
    except ImportError:
        logger.warning("lerobot not installed -- robot control unavailable")
        return None
    except Exception as e:
        logger.warning("Failed to connect robot: %s", e)
        return None


@asynccontextmanager
async def lifespan(app: FastMCP):
    """Initialize all hardware and models at startup."""
    # Redirect stdout to stderr so library print() calls don't corrupt
    # the MCP stdio transport. We save the real stdout for later restore.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    logger.info("Initializing cosmos-chessbot MCP server...")

    # Read config from environment
    overhead_cam_id = int(os.environ.get("CHESS_OVERHEAD_CAM", "1"))
    wrist_cam_id = int(os.environ.get("CHESS_WRIST_CAM", "0"))
    stockfish_path = os.environ.get("CHESS_STOCKFISH", "stockfish")
    cosmos_url = os.environ.get("CHESS_COSMOS_URL")
    robot_port = os.environ.get("CHESS_ROBOT_PORT", "/dev/tty.usbmodem58FA0962531")
    dry_run = os.environ.get("CHESS_DRY_RUN", "false").lower() == "true"
    yolo_pieces = os.environ.get(
        "CHESS_YOLO_PIECES", "runs/detect/yolo26_chess_combined/weights/best.pt",
    )
    yolo_corners = os.environ.get(
        "CHESS_YOLO_CORNERS", "runs/pose/board_corners/weights/best.pt",
    )
    mlp_weights = os.environ.get("CHESS_MLP_WEIGHTS")
    static_corners = os.environ.get("CHESS_STATIC_CORNERS")
    square_size = float(os.environ.get("CHESS_SQUARE_SIZE", "0.05"))
    center_y = float(os.environ.get("CHESS_CENTER_Y", "0.20"))
    table_z = float(os.environ.get("CHESS_TABLE_Z", "0.0"))

    # Cameras
    from ..vision.camera import Camera, CameraConfig

    overhead_camera = Camera(CameraConfig(device_id=overhead_cam_id, name="overhead"))
    wrist_camera = Camera(CameraConfig(device_id=wrist_cam_id, name="wrist"))
    overhead_camera.__enter__()
    wrist_camera.__enter__()
    logger.info("Cameras ready (overhead=%d, wrist=%d)", overhead_cam_id, wrist_cam_id)

    # YOLO detector
    from ..vision.yolo_dino_detector import YOLODINOFenDetector

    yolo_detector = YOLODINOFenDetector(
        yolo_weights=yolo_pieces,
        corner_weights=yolo_corners,
        mlp_weights=mlp_weights,
        static_corners=static_corners,
    )
    logger.info("YOLO detector ready")

    # Stockfish
    from ..stockfish.engine import StockfishEngine

    engine = StockfishEngine(engine_path=stockfish_path)
    engine.start()
    logger.info("Stockfish ready")

    # Cosmos reasoning (remote)
    game_reasoning = None
    if cosmos_url:
        from ..reasoning.remote_reasoning import RemoteChessGameReasoning

        game_reasoning = RemoteChessGameReasoning(server_url=cosmos_url)
        logger.info("Cosmos reasoning client ready (%s)", cosmos_url)

    # Robot
    robot = None
    if not dry_run:
        robot = _connect_robot(robot_port)

    # Waypoint policy
    from ..policy.waypoint_policy import WaypointPolicy

    waypoint_policy = WaypointPolicy()

    state = ServerState(
        overhead_camera=overhead_camera,
        wrist_camera=wrist_camera,
        yolo_detector=yolo_detector,
        engine=engine,
        board=chess.Board(),
        robot=robot,
        game_reasoning=game_reasoning,
        waypoint_policy=waypoint_policy,
        board_square_size=square_size,
        board_center_offset_y=center_y,
        board_table_z=table_z,
    )

    logger.info("MCP server ready (dry_run=%s, robot=%s)", dry_run, robot is not None)

    try:
        yield state
    finally:
        if state.robot is not None:
            try:
                state.robot.disconnect()
            except Exception:
                pass
        overhead_camera.__exit__(None, None, None)
        wrist_camera.__exit__(None, None, None)
        engine.stop()
        sys.stdout = real_stdout
        logger.info("MCP server shut down")


mcp = FastMCP("cosmos-chessbot", lifespan=lifespan)

# Import tool modules to register them on the mcp instance.
# These must come after `mcp` is defined.
from . import tools_camera  # noqa: E402, F401
from . import tools_chess  # noqa: E402, F401
from . import tools_perception  # noqa: E402, F401
from . import tools_reasoning  # noqa: E402, F401
from . import tools_robot  # noqa: E402, F401
from . import tools_util  # noqa: E402, F401
