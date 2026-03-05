"""Main control loop orchestrator."""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import chess

from ..vision import (
    Camera, CameraConfig, YOLODINOFenDetector,
)


@dataclass
class BoardState:
    """Extracted chess board state."""
    fen: str
    confidence: float
    anomalies: list[str]
    raw_response: str

    @property
    def board_detected(self) -> bool:
        return self.fen != "NO_BOARD_DETECTED" and self.confidence > 0.0
from ..stockfish import StockfishEngine
from ..reasoning import (
    ChessGameReasoning,
    GameState,
    MoveDetection,
    compare_fen_states,
    calculate_expected_fen,
    generate_correction_move,
)

logger = logging.getLogger(__name__)


class GamePhase(Enum):
    """Phases of the full game loop."""
    WAIT_FOR_OPPONENT = "wait_for_opponent"
    DETECT_OPPONENT_MOVE = "detect_opponent_move"
    ROBOT_SENSE = "robot_sense"
    ROBOT_PERCEIVE = "robot_perceive"
    ROBOT_PLAN = "robot_plan"
    ROBOT_COMPILE = "robot_compile"
    ROBOT_ACT = "robot_act"
    ROBOT_VERIFY = "robot_verify"
    ROBOT_RECOVER = "robot_recover"
    GAME_OVER = "game_over"


@dataclass
class OrchestratorConfig:
    """Configuration for the chess orchestrator."""

    egocentric_camera_id: int = 1
    wrist_camera_id: int = 0
    stockfish_path: str = "stockfish"
    cosmos_model: str = "nvidia/Cosmos-Reason2-2B"
    cosmos_server_url: Optional[str] = None
    """If set, use remote Cosmos server instead of local inference."""
    data_dir: Path = Path("data/raw")

    # Policy configuration
    policy_type: str = "pi05"
    """Policy to use: 'pi05' or 'waypoint'"""
    policy_checkpoint: Optional[Path] = None
    """Path to policy checkpoint (None uses base model)"""
    enable_planning: bool = True
    """Enable planning for Cosmos Policy (ignored for pi0.5)"""

    # Pi0.5 configuration
    pi05_server_url: str = "ws://localhost:8001"
    """WebSocket URL for remote pi0.5 inference server."""
    pi05_num_steps: int = 699
    """Max action steps per pi0.5 episode."""
    pi05_fps: float = 30.0
    """Action execution rate for pi0.5."""
    pi05_slowdown: int = 1
    """Interpolation factor to slow pi0.5 motion (2=half speed)."""

    color: str = "white"
    """Robot plays as 'white' or 'black'."""

    # Robot configuration
    robot_port: str = "/dev/tty.usbmodem58FA0962531"
    """USB port for SO-101 robot arm."""
    robot_calibration_dir: Optional[str] = None
    """Directory with robot calibration files."""
    overhead_camera_index: int = 1
    """Camera index for egocentric/overhead view."""
    wrist_camera_index: int = 0
    """Camera index for wrist view."""
    dry_run: bool = False
    """If True, skip robot connection (policy inference only)."""

    # Board geometry
    board_square_size: float = 0.05
    """Physical board square size in metres (default 5cm = 40cm total board)."""
    board_center_offset_y: float = 0.20
    """Distance from robot base to board centre, in metres."""
    board_table_z: float = 0.0
    """Board surface height relative to robot base (0 = same level)."""

    # Perception configuration
    perception_backend: str = "yolo"
    """Perception backend: 'yolo' (YOLO26-DINO-MLP) or 'cosmos' (Cosmos-Reason2)"""
    yolo_piece_weights: Optional[str] = None
    """Path to YOLO26 piece detection weights."""
    yolo_corner_weights: Optional[str] = None
    """Path to YOLO26 corner pose detection weights."""
    yolo_mlp_weights: Optional[str] = None
    """Path to DINO-MLP classifier weights."""
    static_corners: Optional[str] = None
    """Path to static calibrated corners JSON (skips corner detection model)."""


class ChessOrchestrator:
    """Main control loop for the chess manipulation system.

    Implements the sense -> perceive -> plan -> compile -> act -> verify loop.
    """

    def __init__(self, config: OrchestratorConfig):
        self.config = config

        # Initialize cameras
        self.egocentric_camera = Camera(
            CameraConfig(
                device_id=config.egocentric_camera_id,
                name="egocentric"
            )
        )
        self.wrist_camera = Camera(
            CameraConfig(
                device_id=config.wrist_camera_id,
                name="wrist"
            )
        )

        # Initialize perception backend (YOLO-DINO)
        logger.info("Using YOLO26-DINO-MLP perception backend")
        self._yolo_detector = YOLODINOFenDetector(
            yolo_weights=config.yolo_piece_weights or "models/yolo_pieces.pt",
            corner_weights=config.yolo_corner_weights or "models/yolo_corners.pt",
            mlp_weights=config.yolo_mlp_weights,
            static_corners=config.static_corners,
        )

        # Initialize Stockfish
        self.engine = StockfishEngine(engine_path=config.stockfish_path)

        # Initialize game reasoning (for pre-action, verification, recovery)
        if config.cosmos_server_url:
            from ..reasoning.remote_reasoning import RemoteChessGameReasoning
            logger.info("Using remote Cosmos game reasoning at %s", config.cosmos_server_url)
            self.game_reasoning = RemoteChessGameReasoning(server_url=config.cosmos_server_url)
        else:
            logger.info("Initializing local Cosmos game reasoning...")
            self.game_reasoning = ChessGameReasoning(model_name=config.cosmos_model)

        # Internal board for tracking game state
        self.board = chess.Board()

        # Initialize robot (lerobot SO-101)
        self.robot = None
        self._joint_names = [
            'shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
            'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos',
        ]
        self._home_position = {
            'shoulder_pan.pos': 7.55,
            'shoulder_lift.pos': -98.12,
            'elbow_flex.pos': 100.00,
            'wrist_flex.pos': 62.88,
            'wrist_roll.pos': 0.08,
            'gripper.pos': 1.63,
        }
        # Park position: arm tucked left so it doesn't block overhead camera
        self._park_position = {
            'shoulder_pan.pos': -80.0,
            'shoulder_lift.pos': -99.1,
            'elbow_flex.pos': 95.2,
            'wrist_flex.pos': 71.7,
            'wrist_roll.pos': -78.0,
            'gripper.pos': 3.1,
        }

        # Board calibration (pixel-to-world mapping, bootstrapped on first sense)
        self.board_calibration = None

        # Initialize selected policy
        self._init_policy()

    def _init_policy(self):
        """Initialize the selected manipulation policy."""
        if self.config.policy_type == "pi05":
            # Pi0.5 uses remote WebSocket server via _execute_pi05_hud();
            # no local policy object needed.
            self.policy = None
            print(f"Pi0.5 policy: remote server at {self.config.pi05_server_url}")
        elif self.config.policy_type == "waypoint":
            from ..policy.waypoint_policy import WaypointPolicy
            print("Initializing Waypoint policy (Cosmos IK)...")
            self.policy = WaypointPolicy()
        else:
            raise ValueError(f"Unknown policy type: {self.config.policy_type}")

    def __enter__(self):
        """Initialize all components."""
        self.egocentric_camera.__enter__()
        self.wrist_camera.__enter__()
        self.engine.start()

        # Connect robot via lerobot
        if not self.config.dry_run:
            self._connect_robot()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup all components."""
        if self.robot is not None:
            try:
                self.robot.disconnect()
                logger.info("Robot disconnected")
            except Exception as e:
                logger.warning("Error disconnecting robot: %s", e)
            self.robot = None
        self.egocentric_camera.__exit__(exc_type, exc_val, exc_tb)
        self.wrist_camera.__exit__(exc_type, exc_val, exc_tb)
        self.engine.stop()

    def _connect_robot(self):
        """Connect to SO-101 via lerobot."""
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
            from lerobot.cameras.opencv import OpenCVCameraConfig

            logger.info("Connecting to SO-101 on %s...", self.config.robot_port)

            camera_config = {
                "egocentric": OpenCVCameraConfig(
                    index_or_path=self.config.overhead_camera_index,
                    width=640, height=480, fps=30,
                ),
                "wrist": OpenCVCameraConfig(
                    index_or_path=self.config.wrist_camera_index,
                    width=640, height=480, fps=30,
                ),
            }

            config_kwargs = {
                "port": self.config.robot_port,
                "id": "my_follower_arm",
                "cameras": camera_config,
            }
            if self.config.robot_calibration_dir:
                config_kwargs["calibration_dir"] = self.config.robot_calibration_dir

            self.robot = SO101Follower(SO101FollowerConfig(**config_kwargs))
            self.robot.connect()
            logger.info("Robot connected successfully")

        except ImportError:
            logger.error("lerobot not installed -- robot control unavailable")
            self.robot = None
        except Exception as e:
            logger.error("Failed to connect to robot: %s", e)
            self.robot = None

    def _init_board_calibration(self, image) -> None:
        """Bootstrap board calibration from YOLO corner detection.

        Detects the 4 board corners in ``image`` and creates a
        ``BoardCalibration`` instance that maps pixel coordinates to 3D
        world coordinates on the board plane.

        YOLO returns corners as [TL, TR, BR, BL] from camera perspective.
        BoardCalibration expects [a1, h1, h8, a8].
        Mapping: a1=BL(idx 3), h1=BR(idx 2), h8=TR(idx 1), a8=TL(idx 0).
        """
        if self._yolo_detector is None:
            logger.warning("No YOLO detector -- cannot bootstrap board calibration")
            return

        import numpy as np
        if not isinstance(image, np.ndarray):
            image_np = np.array(image)
        else:
            image_np = image

        corners = self._yolo_detector._detect_corners(image_np)
        if corners is None:
            logger.warning("Board corner detection failed -- calibration unavailable")
            return

        # corners shape: (4, 2) as [TL, TR, BR, BL]
        # BoardCalibration wants [a1, h1, h8, a8]
        pixel_corners = [
            tuple(corners[3]),  # a1 = BL
            tuple(corners[2]),  # h1 = BR
            tuple(corners[1]),  # h8 = TR
            tuple(corners[0]),  # a8 = TL
        ]

        h, w = image_np.shape[:2]

        from ..utils.pixel_to_board import BoardCalibration
        self.board_calibration = BoardCalibration(
            pixel_corners=pixel_corners,
            image_size=(w, h),
            square_size=self.config.board_square_size,
            table_z=self.config.board_table_z,
            center_y=self.config.board_center_offset_y,
        )
        logger.info(
            "Board calibration initialised (square_size=%.3fm, image=%dx%d)",
            self.config.board_square_size, w, h,
        )

    def sense(self) -> tuple:
        """Capture images from cameras.

        Returns:
            Tuple of (overhead_image, wrist_image)
        """
        overhead = self.egocentric_camera.capture()
        wrist = self.wrist_camera.capture()
        return overhead, wrist

    def perceive(self, overhead_image) -> BoardState:
        """Extract board state from egocentric camera.

        Args:
            overhead_image: PIL Image or numpy array from egocentric camera

        Returns:
            BoardState with FEN, confidence, and anomalies
        """
        import numpy as np
        if not isinstance(overhead_image, np.ndarray):
            image_np = np.array(overhead_image)
        else:
            image_np = overhead_image
        fen = self._yolo_detector.detect_fen(image_np, verbose=logger.isEnabledFor(logging.DEBUG))
        return BoardState(fen=fen, confidence=1.0, anomalies=[], raw_response="")

    def plan(self, board_state: BoardState) -> str:
        """Get best chess move from Stockfish.

        Args:
            board_state: Current board state

        Returns:
            Best move in UCI format (e.g., 'e2e4')
        """
        fen = board_state.fen
        # YOLO returns board-only FEN (no turn/castling/ep fields).
        # Stockfish requires a full 6-field FEN.  Append reasonable defaults
        # based on whose turn it is according to our internal board.
        if len(fen.split()) == 1:
            turn = "w" if self.board.turn == chess.WHITE else "b"
            fen = f"{fen} {turn} KQkq - 0 1"
        return self.engine.get_best_move(fen=fen)

    def compile_intent(self, move: str, board_state: BoardState, image=None, wrist_image=None) -> dict:
        """Compile symbolic move into physical manipulation intent.

        Uses Cosmos-Reason2 to reason about physical constraints and
        plan a 2D pixel-space trajectory (Action CoT).

        Args:
            move: UCI move (e.g., 'e2e4')
            board_state: Current board state
            image: Current egocentric image (for reasoning)

        Returns:
            Intent dictionary with pick/place squares, trajectory, and constraints
        """
        # Parse move
        from_square = move[:2]
        to_square = move[2:4]

        # Determine piece type for more specific prompts
        piece_type = "piece"
        try:
            piece = chess.Board(board_state.fen).piece_at(chess.parse_square(from_square))
            if piece:
                piece_type = chess.piece_name(piece.piece_type)
        except Exception:
            pass

        # Use Cosmos-Reason2 for pre-action reasoning if available
        action_reasoning = None
        if self.game_reasoning and image:
            print(f"Reasoning about action: {move}")
            action_reasoning = self.game_reasoning.reason_about_action(
                image=image,
                move_uci=move,
                from_square=from_square,
                to_square=to_square,
                wrist_image=wrist_image,
            )
            print(f"  Obstacles: {action_reasoning.obstacles}")
            print(f"  Grasp strategy: {action_reasoning.grasp_strategy}")
            print(f"  Risks: {action_reasoning.risks}")

        # Plan trajectory using Action CoT (retry up to 3 times on 0 waypoints)
        trajectory_plan = None
        waypoints_3d = None
        if self.game_reasoning and image:
            for attempt in range(3):
                print(f"Planning trajectory (Action CoT): {move}"
                      + (f" (retry {attempt})" if attempt > 0 else ""))
                trajectory_plan = self.game_reasoning.plan_trajectory(
                    image=image,
                    move_uci=move,
                    from_square=from_square,
                    to_square=to_square,
                    piece_type=piece_type,
                    wrist_image=wrist_image,
                )
                if trajectory_plan.waypoints:
                    break
                print("  WARNING: 0 waypoints returned, retrying...")
            print(f"  Waypoints: {len(trajectory_plan.waypoints)}")
            for wp in trajectory_plan.waypoints:
                print(f"    {wp.label}: ({wp.point_2d[0]}, {wp.point_2d[1]})")

            # Convert to 3D if calibration available
            if hasattr(self, 'board_calibration') and self.board_calibration and trajectory_plan.waypoints:
                waypoints_3d = self.board_calibration.waypoints_to_3d(
                    trajectory_plan.waypoints
                )
                print(f"  3D waypoints computed ({len(waypoints_3d)} points)")

        return {
            "pick_square": from_square,
            "place_square": to_square,
            "move_uci": move,
            "piece_type": piece_type,
            "action_reasoning": action_reasoning,
            "trajectory_plan": trajectory_plan,
            "waypoints_3d": waypoints_3d,
            "constraints": {
                "approach": "from_above",
                "clearance": 0.05,  # meters
                "avoidance": [],  # squares to avoid
            },
        }

    def execute(self, intent: dict) -> bool:
        """Execute physical manipulation with selected policy.

        Args:
            intent: Manipulation intent from compile_intent

        Returns:
            True if execution succeeded
        """
        instruction = (
            f"Pick the piece at {intent['pick_square']} "
            f"and place it at {intent['place_square']}"
        )

        # Waypoint: execute Cosmos trajectory via geometric IK
        if self.config.policy_type == "waypoint":
            from ..policy.waypoint_policy import WaypointPolicy
            assert isinstance(self.policy, WaypointPolicy)
            waypoints_3d = intent.get("waypoints_3d")
            trajectory_plan = intent.get("trajectory_plan")
            if not waypoints_3d or not trajectory_plan:
                logger.error("Waypoint policy requires waypoints_3d from compile_intent")
                return False
            labels = [wp.label for wp in trajectory_plan.waypoints]
            return self.policy.run_waypoint_trajectory(
                waypoints_3d=waypoints_3d,
                labels=labels,
                get_state_fn=self._get_robot_state,
                send_action_fn=self._send_joint_targets,
            )

        # Pi0.5 with HUD overlay: chunked closed-loop execution
        if self.config.policy_type == "pi05":
            return self._execute_pi05_hud(intent)

        # Other policies: single-shot inference
        overhead, wrist = self.sense()
        robot_state = self._get_robot_state()
        images = {"egocentric": overhead, "wrist": wrist}

        if self.config.enable_planning and hasattr(self.policy, 'plan_action'):
            candidates = self.policy.plan_action(images, robot_state, instruction)
            action = candidates[0]
            print(f"Planning: Selected action with {action.success_probability:.2%} confidence")
            print(f"  (from {len(candidates)} candidates)")
        else:
            action = self.policy.select_action(images, robot_state, instruction)
            print(f"Direct action with {action.success_probability:.2%} confidence")

        success = self._execute_robot_action(action.actions)
        return success

    def _execute_pi05_hud(self, intent: dict) -> bool:
        """Execute a move using pi0.5 with HUD overlay via remote server.

        Connects to the pi0.5 WebSocket server, runs sequential chunked
        inference with HUD overlay applied to the egocentric camera feed,
        and returns the robot to home position when done.

        Args:
            intent: Manipulation intent with pick_square and place_square.

        Returns:
            True if execution completed without errors.
        """
        import msgpack
        import numpy as np
        import torch

        source = intent["pick_square"]
        target = intent["place_square"]
        task = "Move the small object from the green circle to the magenta circle"

        logger.info("Pi0.5 + HUD execution: %s -> %s", source, target)

        # msgpack helpers for numpy arrays
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

        try:
            import websockets.sync.client

            ws = websockets.sync.client.connect(
                self.config.pi05_server_url,
                compression=None, max_size=None,
            )
            packer = msgpack.Packer(default=_pack_array)

            # Read server metadata
            metadata = msgpack.unpackb(ws.recv(), object_hook=_unpack_array)
            logger.info("Pi0.5 server: %s", metadata)

            # Reset policy state
            ws.send(packer.pack({"command": "reset"}))
            msgpack.unpackb(ws.recv(), object_hook=_unpack_array)

        except Exception as e:
            logger.error("Failed to connect to pi0.5 server at %s: %s",
                         self.config.pi05_server_url, e)
            return False

        from ..vision.hud_overlay import (
            apply_hud, compute_homography, detect_corners,
        )

        hud_corners = None
        hud_H = None
        total_steps = 0
        chunk_count = 0
        step_delay = 1.0 / self.config.pi05_fps
        max_steps = self.config.pi05_num_steps
        slowdown = self.config.pi05_slowdown

        try:
            while total_steps < max_steps:
                # Capture observation
                if self.robot is None:
                    logger.error("No robot connected")
                    return False

                obs_raw = self.robot.get_observation()
                joints = np.array([
                    float(obs_raw[n].item() if hasattr(obs_raw[n], "item") else obs_raw[n])
                    for n in self._joint_names
                ], dtype=np.float32)

                # Extract images
                ego_key = next((k for k in obs_raw if "egocentric" in k), None)
                wrist_key = next((k for k in obs_raw if "wrist" in k and "pos" not in k), None)
                if ego_key is None:
                    logger.error("No egocentric image in observation")
                    return False

                overhead = np.array(obs_raw[ego_key], dtype=np.uint8)
                if overhead.ndim == 3 and overhead.shape[0] == 3:
                    overhead = np.transpose(overhead, (1, 2, 0))

                if wrist_key and obs_raw[wrist_key] is not None:
                    wrist = np.array(obs_raw[wrist_key], dtype=np.uint8)
                    if wrist.ndim == 3 and wrist.shape[0] == 3:
                        wrist = np.transpose(wrist, (1, 2, 0))
                else:
                    wrist = np.zeros_like(overhead)

                # Sticky corner detection: only update on high-confidence detections
                corners, detection_conf = detect_corners(overhead, return_conf=True)
                if corners is not None and detection_conf >= 0.98:
                    hud_corners = corners
                    hud_H = compute_homography(hud_corners)
                    if chunk_count == 0:
                        logger.info("HUD: detected board corners (conf=%.3f)", detection_conf)

                # Apply HUD overlay
                apply_hud(overhead, source, target, hud_corners, hud_H)

                # Build observation for pi0.5 server
                obs = {
                    "observation.images.egocentric": overhead,
                    "observation.images.wrist": wrist,
                    "observation.state": joints,
                    "task": task,
                }

                # Request action chunk from server
                ws.send(packer.pack(obs))
                response = ws.recv()
                result = msgpack.unpackb(response, object_hook=_unpack_array)

                chunk = np.array(
                    result.get("action_chunk") or result.get("action"),
                    dtype=np.float32,
                )
                if chunk.ndim == 1:
                    chunk = chunk.reshape(1, -1)

                # Interpolate for slowdown
                if slowdown > 1:
                    n, dim = chunk.shape
                    indices = np.arange(n)
                    new_indices = np.linspace(0, n - 1, (n - 1) * slowdown + 1)
                    interp = np.zeros((len(new_indices), dim), dtype=chunk.dtype)
                    for d in range(dim):
                        interp[:, d] = np.interp(new_indices, indices, chunk[:, d])
                    chunk = interp

                chunk_count += 1
                if chunk_count <= 2 or chunk_count % 10 == 0:
                    print(f"  Pi0.5 chunk {chunk_count}: {chunk.shape[0]} actions, "
                          f"total_steps={total_steps}/{max_steps}")

                # Execute chunk
                for i in range(len(chunk)):
                    if total_steps >= max_steps:
                        break
                    action = chunk[i]
                    action_dict = {}
                    for j, name in enumerate(self._joint_names):
                        if j < len(action):
                            action_dict[name] = torch.tensor(
                                [float(action[j])], dtype=torch.float32,
                            )
                    self.robot.send_action(action_dict)
                    total_steps += 1
                    time.sleep(step_delay)

        except Exception as e:
            logger.error("Pi0.5 execution failed: %s", e)
            return False
        finally:
            try:
                ws.close()
            except Exception:
                pass

        print(f"  Pi0.5 episode complete: {total_steps} steps, {chunk_count} chunks")

        # Return to home position
        self.home_robot()
        return True

    def _get_robot_state(self):
        """Get current robot state (joint positions + gripper) from SO-101.

        Returns:
            numpy array of shape (6,) with joint angles in degrees:
            [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        """
        import numpy as np

        if self.robot is None:
            return np.zeros(6)

        obs = self.robot.get_observation()
        positions = []
        for name in self._joint_names:
            val = obs[name]
            if hasattr(val, 'item'):
                val = val.item()
            positions.append(float(val))

        return np.array(positions, dtype=np.float32)

    def _execute_robot_action(self, actions):
        """Execute predicted actions on the robot via lerobot.

        Supports both single-step (action_dim,) and multi-step (horizon, action_dim)
        action tensors. Each step is sent as joint angle targets in degrees.

        Args:
            actions: Action tensor from policy. Shape (6,) or (horizon, 6)
                     Values are joint angle targets in degrees.

        Returns:
            True if execution succeeded
        """
        import torch

        if self.robot is None:
            logger.warning("No robot connected -- cannot execute action")
            return False

        try:
            import numpy as np

            # Normalize shape to (horizon, action_dim)
            if hasattr(actions, 'numpy'):
                actions_np = actions.cpu().numpy() if hasattr(actions, 'cpu') else actions.numpy()
            else:
                actions_np = np.asarray(actions)

            if actions_np.ndim == 1:
                actions_np = actions_np[np.newaxis, :]  # (1, action_dim)

            for step_idx in range(actions_np.shape[0]):
                step_actions = actions_np[step_idx]

                # Build lerobot action dict: {joint_name: torch.tensor([deg_value])}
                action_dict = {}
                for i, name in enumerate(self._joint_names):
                    if i < len(step_actions):
                        action_dict[name] = torch.tensor(
                            [float(step_actions[i])], dtype=torch.float32,
                        )

                self.robot.send_action(action_dict)

                # Brief pause between multi-step actions
                if actions_np.shape[0] > 1:
                    time.sleep(0.05)

            return True

        except Exception as e:
            logger.error("Robot action execution failed: %s", e)
            return False

    def _send_joint_targets(self, targets_deg: "np.ndarray") -> None:
        """Send joint angle targets (degrees) to the robot.

        Thin wrapper used as a callback by PPOPolicy.run_control_loop().
        """
        import torch

        if self.robot is None:
            logger.warning("No robot connected -- cannot send targets")
            return

        action_dict = {}
        for i, name in enumerate(self._joint_names):
            if i < len(targets_deg):
                action_dict[name] = torch.tensor(
                    [float(targets_deg[i])], dtype=torch.float32,
                )
        self.robot.send_action(action_dict)

    def home_robot(self):
        """Send robot to home position."""
        import torch

        if self.robot is None:
            logger.warning("No robot connected -- cannot home")
            return

        home_dict = {
            name: torch.tensor([val], dtype=torch.float32)
            for name, val in self._home_position.items()
        }
        self.robot.send_action(home_dict)
        logger.info("Home position sent")

    def park_robot(self):
        """Park the arm out of the overhead camera's field of view.

        Rotates shoulder_pan to -90 degrees (pointing left) so the arm
        does not occlude the board during sensing/perception.
        """
        import time
        import torch

        if self.robot is None:
            logger.warning("No robot connected -- cannot park")
            return

        park_dict = {
            name: torch.tensor([val], dtype=torch.float32)
            for name, val in self._park_position.items()
        }
        self.robot.send_action(park_dict)
        time.sleep(2.0)  # wait for arm to reach park position
        logger.info("Robot parked (arm out of camera view)")

    def verify(self, expected_fen: str, intent: dict = None, vision_backend="yolo") -> tuple:
        """Verify board state after action using visual check + FEN comparison.

        Two-stage verification:
        1. Visual goal check (Cosmos Reason2) — catches physical issues
           (piece tipped, adjacent bumped, gripper didn't release)
        2. FEN comparison — catches logical placement errors

        Args:
            expected_fen: Expected FEN after move
            intent: Intent dict from compile_intent (for move details)
            vision_backend: Which vision system to use ("cosmos" or "yolo")

        Returns:
            Tuple of (success, actual_board_state, fen_comparison, goal_verification)
        """
        self.park_robot()
        overhead, wrist = self.sense()

        # Stage 1: Visual goal verification (Cosmos Reason2)
        goal_verification = None
        if self.game_reasoning and intent:
            print("  Visual goal check (Cosmos Reason2)...")
            goal_verification = self.game_reasoning.verify_goal(
                image=overhead,
                move_uci=intent["move_uci"],
                from_square=intent["pick_square"],
                to_square=intent["place_square"],
                piece_type=intent.get("piece_type", "piece"),
                wrist_image=wrist,
            )
            status = "PASS" if goal_verification.success else "FAIL"
            print(f"  Visual check: {status} "
                  f"(confidence: {goal_verification.confidence:.2f})")
            if goal_verification.physical_issues:
                print(f"  Physical issues: {goal_verification.physical_issues}")

        # Stage 2: FEN comparison
        actual_state = self.perceive(overhead)

        fen_comparison = compare_fen_states(expected_fen, actual_state.fen)

        if not fen_comparison.match:
            print("  FEN comparison: FAIL")
            print(fen_comparison.summary())
        else:
            print("  FEN comparison: PASS")

        return fen_comparison.match, actual_state, fen_comparison, goal_verification

    def recover(
        self,
        intent: dict,
        expected_fen: str,
        failure_state: BoardState,
        fen_comparison: "FENComparison",
        max_attempts: int = 3,
    ) -> bool:
        """Attempt to recover from failed execution.

        Uses Cosmos-Reason2 to understand what went wrong and plan correction.

        Args:
            intent: Original intent that failed
            expected_fen: Expected FEN after move
            failure_state: Actual board state after failure
            fen_comparison: FEN comparison result
            max_attempts: Maximum correction attempts

        Returns:
            True if recovery succeeded
        """
        print("Attempting recovery...")

        for attempt in range(max_attempts):
            print(f"  Recovery attempt {attempt + 1}/{max_attempts}")

            # Get current view
            overhead, wrist = self.sense()

            # Use Cosmos to reason about the correction
            if self.game_reasoning:
                correction_plan = self.game_reasoning.plan_correction(
                    image=overhead,
                    expected_fen=expected_fen,
                    actual_fen=failure_state.fen,
                    differences=fen_comparison.differences,
                    wrist_image=wrist,
                )
                print(f"  Physical cause: {correction_plan.physical_cause}")
                print(f"  Correction needed: {correction_plan.correction_needed}")

            # Generate correction move
            correction_move = generate_correction_move(fen_comparison)

            if correction_move is None:
                print("  Cannot automatically generate correction")
                print("     (Complex case - would need human intervention)")
                return False

            print(f"  Correction move: {correction_move}")

            # Execute correction
            correction_intent = {
                "pick_square": correction_move[:2],
                "place_square": correction_move[2:4],
                "move_uci": correction_move,
                "constraints": {"approach": "from_above", "clearance": 0.05, "avoidance": []},
            }

            exec_success = self.execute(correction_intent)
            if not exec_success:
                print("  Correction execution failed")
                continue

            # Verify correction
            verify_success, new_state, new_comparison, _ = self.verify(expected_fen)

            if verify_success:
                print("  Recovery successful!")
                return True

            # Update for next attempt
            failure_state = new_state
            fen_comparison = new_comparison

        print(f"Recovery failed after {max_attempts} attempts")
        return False

    def execute_move_with_verification(
        self,
        move: str,
        current_fen: str,
        current_image=None,
        wrist_image=None,
    ) -> bool:
        """Execute a move with full verification and recovery loop.

        This is the core method demonstrating Cosmos Reason2 integration:
        1. Pre-action reasoning (obstacles, grasp strategy)
        2. Action execution
        3. Post-action verification (FEN comparison)
        4. Recovery if needed (correction reasoning)

        Args:
            move: UCI move to execute (e.g., 'e2e4')
            current_fen: Current board FEN
            current_image: Current egocentric image

        Returns:
            True if move succeeded (after corrections if needed)
        """
        # 1. Calculate expected FEN after move
        expected_fen = calculate_expected_fen(current_fen, move)
        print(f"\nExecuting move: {move}")
        print(f"   Current FEN:  {current_fen}")
        print(f"   Expected FEN: {expected_fen}")

        # 2. Pre-action reasoning (using Cosmos)
        board_state = BoardState(
            fen=current_fen,
            confidence=1.0,
            anomalies=[],
            raw_response="",
        )
        intent = self.compile_intent(move, board_state, image=current_image, wrist_image=wrist_image)

        # 3. Execute move
        print("\nExecuting action...")
        exec_success = self.execute(intent)

        if not exec_success:
            print("Execution failed at robot control level")
            return False

        # 4. Verify using visual check + FEN comparison
        print("\nVerifying result...")
        verify_success, actual_state, fen_comparison, goal_check = self.verify(
            expected_fen, intent=intent
        )

        if verify_success:
            print("Move completed successfully!")
            return True

        # 5. Recover if verification failed
        print("\nMove verification failed, attempting recovery...")
        return self.recover(intent, expected_fen, actual_state, fen_comparison)

    def run_one_move(self) -> bool:
        """Execute one complete move cycle with full reasoning loop.

        Returns:
            True if move succeeded
        """
        # 0. Park arm out of camera view before sensing
        self.park_robot()

        # 1. Sense
        print("\n" + "=" * 60)
        print("SENSING")
        print("=" * 60)
        overhead, wrist = self.sense()

        # Bootstrap board calibration on first call
        if self.board_calibration is None:
            self._init_board_calibration(overhead)

        # 2. Perceive
        print("\nPERCEIVING")
        board_state = self.perceive(overhead)
        print(f"   FEN: {board_state.fen}")
        print(f"   Confidence: {board_state.confidence:.2%}")
        if board_state.anomalies:
            print(f"   Anomalies: {board_state.anomalies}")

        # 3. Plan
        print("\nPLANNING")
        best_move = self.plan(board_state)
        print(f"   Best move: {best_move}")

        # 4. Execute with verification and recovery
        success = self.execute_move_with_verification(
            move=best_move,
            current_fen=board_state.fen,
            current_image=overhead,
            wrist_image=wrist,
        )

        # 5. Track on internal board if successful
        if success:
            try:
                self.board.push(chess.Move.from_uci(best_move))
            except (chess.InvalidMoveError, chess.IllegalMoveError):
                logger.warning("Could not push move %s to internal board", best_move)

        return success

    def capture_video_frames(self, n_frames: int = 8, interval: float = 0.2) -> list:
        """Capture a sequence of frames for video-based reasoning.

        Args:
            n_frames: Number of frames to capture
            interval: Time between frames in seconds

        Returns:
            List of PIL Images
        """
        frames = []
        for _ in range(n_frames):
            overhead, _ = self.sense()
            frames.append(overhead)
            time.sleep(interval)
        return frames

    def wait_for_opponent_turn(
        self,
        poll_interval: float = 1.0,
        timeout: float = 300.0,
    ) -> GameState:
        """Wait for the opponent to complete their turn using Cosmos video reasoning.

        Polls ``analyze_game_state()`` on captured video frames until
        ``should_robot_act`` is True.

        Args:
            poll_interval: Seconds between polls
            timeout: Maximum wait time in seconds

        Returns:
            GameState indicating the robot should act
        """
        print("\nWaiting for opponent's turn...")
        t0 = time.time()

        while time.time() - t0 < timeout:
            frames = self.capture_video_frames(n_frames=4, interval=0.15)
            game_state = self.game_reasoning.analyze_game_state(frames)

            print(f"  Turn: {game_state.whose_turn.value}, "
                  f"opponent_moving: {game_state.opponent_moving}, "
                  f"confidence: {game_state.confidence:.2f}")

            if game_state.should_robot_act:
                print("Opponent's turn complete — robot should act now.")
                return game_state

            time.sleep(poll_interval)

        # Timeout — return a fallback state
        logger.warning("Timeout waiting for opponent turn")
        from ..reasoning import GameState as GS
        from ..reasoning.game_reasoning import Turn
        return GS(
            whose_turn=Turn.UNKNOWN,
            opponent_moving=False,
            should_robot_act=True,
            reasoning="Timeout waiting for opponent",
            confidence=0.0,
        )

    def detect_opponent_move(self) -> MoveDetection:
        """Detect what move the opponent just made using Cosmos video reasoning.

        Returns:
            MoveDetection with from/to squares and piece type
        """
        print("\nDetecting opponent's move...")
        frames = self.capture_video_frames(n_frames=8, interval=0.2)
        detection = self.game_reasoning.detect_move(frames)

        if detection.move_occurred:
            print(f"  Detected: {detection.from_square} -> {detection.to_square} "
                  f"({detection.piece_type}), confidence: {detection.confidence:.2f}")
        else:
            print(f"  No move detected (confidence: {detection.confidence:.2f})")

        return detection

    def run_game(self, max_moves: Optional[int] = None) -> None:
        """Run a full game loop with turn detection and opponent move detection.

        This is the main game loop that demonstrates Cosmos Reason2's
        temporal video reasoning for multi-agent physical interaction:

        1. Wait for opponent's turn (video-based turn detection)
        2. Detect opponent's move (video-based move detection)
        3. Robot's turn: sense -> perceive -> plan -> compile -> act -> verify

        Args:
            max_moves: Maximum number of robot moves (None = play until game end)
        """
        robot_is_white = self.config.color == "white"
        move_count = 0
        limit = max_moves if max_moves is not None else float("inf")

        print("\n" + "=" * 60)
        print(f"STARTING FULL GAME  (robot plays {'white' if robot_is_white else 'black'})")
        print("=" * 60)

        while not self.board.is_game_over() and move_count < limit:
            robot_to_move = (self.board.turn == chess.WHITE) == robot_is_white

            if not robot_to_move:
                # --- Opponent's turn ---
                phase = GamePhase.WAIT_FOR_OPPONENT
                print(f"\n{'='*60}")
                print(f"[Move {self.board.fullmove_number}] OPPONENT'S TURN")
                print(f"{'='*60}")

                self.wait_for_opponent_turn()

                phase = GamePhase.DETECT_OPPONENT_MOVE
                detection = self.detect_opponent_move()

                if detection.move_occurred and detection.from_square and detection.to_square:
                    uci = f"{detection.from_square}{detection.to_square}"
                    try:
                        move = chess.Move.from_uci(uci)
                        if move in self.board.legal_moves:
                            self.board.push(move)
                            print(f"  Pushed opponent move: {uci}")
                        else:
                            logger.warning("Detected move %s is illegal, skipping push", uci)
                    except (chess.InvalidMoveError, ValueError):
                        logger.warning("Could not parse detected move: %s", uci)
                else:
                    logger.warning("Could not detect opponent move — relying on perception")

            else:
                # --- Robot's turn ---
                print(f"\n{'='*60}")
                print(f"[Move {self.board.fullmove_number}] ROBOT'S TURN")
                print(f"{'='*60}")

                success = self.run_one_move()
                move_count += 1

                if not success:
                    print("Robot move failed. Stopping game.")
                    break

        # Game ended
        if self.board.is_game_over():
            result = self.board.result()
            print(f"\nGAME OVER: {result}")
            if self.board.is_checkmate():
                print("Checkmate!")
            elif self.board.is_stalemate():
                print("Stalemate!")
            elif self.board.is_insufficient_material():
                print("Draw by insufficient material.")
        else:
            print(f"\nGame stopped after {move_count} robot moves.")
