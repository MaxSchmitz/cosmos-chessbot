"""Main control loop orchestrator."""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import chess

from ..vision import (
    Camera, CameraConfig, CosmosPerception, BoardState,
    RemoteCosmosPerception, YOLODINOFenDetector,
)
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
    policy_type: str = "cosmos"
    """Policy to use: 'pi05' or 'cosmos'"""
    policy_checkpoint: Optional[Path] = None
    """Path to policy checkpoint (None uses base model)"""
    enable_planning: bool = True
    """Enable planning for Cosmos Policy (ignored for π₀.₅)"""

    color: str = "white"
    """Robot plays as 'white' or 'black'."""

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

        # Initialize perception backend
        if config.perception_backend == "yolo":
            logger.info("Using YOLO26-DINO-MLP perception backend")
            self._yolo_detector = YOLODINOFenDetector(
                yolo_weights=config.yolo_piece_weights or "runs/detect/yolo26_chess/weights/best.pt",
                corner_weights=config.yolo_corner_weights or "runs/pose/board_corners/weights/best.pt",
                mlp_weights=config.yolo_mlp_weights,
                static_corners=config.static_corners,
            )
            self.perception = None  # FEN detection handled by _yolo_detector
        elif config.cosmos_server_url:
            logger.info("Using remote Cosmos server at %s", config.cosmos_server_url)
            self._yolo_detector = None
            self.perception = RemoteCosmosPerception(server_url=config.cosmos_server_url)
        else:
            logger.info("Using local Cosmos model: %s", config.cosmos_model)
            self._yolo_detector = None
            self.perception = CosmosPerception(model_name=config.cosmos_model)

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

        # Initialize selected policy
        self._init_policy()

    def _init_policy(self):
        """Initialize the selected manipulation policy."""
        if self.config.policy_type == "pi05":
            from ..policy.pi05_policy import PI05Policy
            print(f"Initializing π₀.₅ policy...")
            self.policy = PI05Policy(
                checkpoint_path=self.config.policy_checkpoint
            )
        elif self.config.policy_type == "cosmos":
            from ..policy.cosmos_policy import CosmosPolicy
            print(f"Initializing Cosmos Policy...")
            self.policy = CosmosPolicy(
                checkpoint_path=self.config.policy_checkpoint,
                enable_planning=self.config.enable_planning
            )
        else:
            raise ValueError(f"Unknown policy type: {self.config.policy_type}")

    def __enter__(self):
        """Initialize all components."""
        self.egocentric_camera.__enter__()
        self.wrist_camera.__enter__()
        self.engine.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup all components."""
        self.egocentric_camera.__exit__(exc_type, exc_val, exc_tb)
        self.wrist_camera.__exit__(exc_type, exc_val, exc_tb)
        self.engine.stop()

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
        if self._yolo_detector is not None:
            import numpy as np
            if not isinstance(overhead_image, np.ndarray):
                image_np = np.array(overhead_image)
            else:
                image_np = overhead_image
            fen = self._yolo_detector.detect_fen(image_np)
            return BoardState(fen=fen, confidence=1.0, anomalies=[], raw_response="")
        return self.perception.perceive(overhead_image)

    def plan(self, board_state: BoardState) -> str:
        """Get best chess move from Stockfish.

        Args:
            board_state: Current board state

        Returns:
            Best move in UCI format (e.g., 'e2e4')
        """
        return self.engine.get_best_move(fen=board_state.fen)

    def compile_intent(self, move: str, board_state: BoardState, image=None) -> dict:
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
            )
            print(f"  Obstacles: {action_reasoning.obstacles}")
            print(f"  Grasp strategy: {action_reasoning.grasp_strategy}")
            print(f"  Risks: {action_reasoning.risks}")

        # Plan trajectory using Action CoT
        trajectory_plan = None
        waypoints_3d = None
        if self.game_reasoning and image:
            print(f"Planning trajectory (Action CoT): {move}")
            trajectory_plan = self.game_reasoning.plan_trajectory(
                image=image,
                move_uci=move,
                from_square=from_square,
                to_square=to_square,
                piece_type=piece_type,
            )
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
        # Get current observations
        overhead, wrist = self.sense()
        robot_state = self._get_robot_state()

        images = {"egocentric": overhead, "wrist": wrist}

        # Create instruction for π₀.₅ (if using language)
        instruction = None
        if self.config.policy_type == "pi05":
            instruction = (
                f"Pick the piece at {intent['pick_square']} "
                f"and place it at {intent['place_square']}"
            )

        # Plan or select action
        if self.config.enable_planning and hasattr(self.policy, 'plan_action'):
            # Get multiple candidates (Cosmos Policy with planning)
            candidates = self.policy.plan_action(images, robot_state, instruction)
            action = candidates[0]  # Best candidate

            print(f"Planning: Selected action with {action.success_probability:.2%} confidence")
            print(f"  (from {len(candidates)} candidates)")
        else:
            # Direct action (π₀.₅ or Cosmos without planning)
            action = self.policy.select_action(images, robot_state, instruction)
            print(f"Direct action with {action.success_probability:.2%} confidence")

        # Execute on robot
        success = self._execute_robot_action(action.actions)

        return success

    def _get_robot_state(self):
        """Get current robot state (joint positions, gripper, etc.).

        Returns:
            numpy array of robot state

        TODO: Implement robot state reading from SO-101
        """
        import numpy as np
        # Placeholder: 7 DOF arm + gripper state
        return np.zeros(8)

    def _execute_robot_action(self, actions):
        """Execute predicted actions on the robot.

        Args:
            actions: Action tensor from policy (typically [horizon, action_dim])

        Returns:
            True if execution succeeded

        TODO: Implement robot control for SO-101
        """
        print(f"TODO: Execute robot actions with shape {actions.shape}")
        # For now, just simulate success
        return False

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
        overhead, _ = self.sense()

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
        print(f"🔧 Attempting recovery...")

        for attempt in range(max_attempts):
            print(f"  Recovery attempt {attempt + 1}/{max_attempts}")

            # Get current view
            overhead, _ = self.sense()

            # Use Cosmos to reason about the correction
            if self.game_reasoning:
                correction_plan = self.game_reasoning.plan_correction(
                    image=overhead,
                    expected_fen=expected_fen,
                    actual_fen=failure_state.fen,
                    differences=fen_comparison.differences,
                )
                print(f"  Physical cause: {correction_plan.physical_cause}")
                print(f"  Correction needed: {correction_plan.correction_needed}")

            # Generate correction move
            correction_move = generate_correction_move(fen_comparison)

            if correction_move is None:
                print(f"  ❌ Cannot automatically generate correction")
                print(f"     (Complex case - would need human intervention)")
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
                print(f"  ❌ Correction execution failed")
                continue

            # Verify correction
            verify_success, new_state, new_comparison, _ = self.verify(expected_fen)

            if verify_success:
                print(f"  ✅ Recovery successful!")
                return True

            # Update for next attempt
            failure_state = new_state
            fen_comparison = new_comparison

        print(f"❌ Recovery failed after {max_attempts} attempts")
        return False

    def execute_move_with_verification(
        self,
        move: str,
        current_fen: str,
        current_image=None,
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
        print(f"\n📋 Executing move: {move}")
        print(f"   Current FEN:  {current_fen}")
        print(f"   Expected FEN: {expected_fen}")

        # 2. Pre-action reasoning (using Cosmos)
        board_state = BoardState(
            fen=current_fen,
            confidence=1.0,
            anomalies=[],
            raw_response="",
        )
        intent = self.compile_intent(move, board_state, image=current_image)

        # 3. Execute move
        print(f"\n🤖 Executing action...")
        exec_success = self.execute(intent)

        if not exec_success:
            print(f"❌ Execution failed at robot control level")
            return False

        # 4. Verify using visual check + FEN comparison
        print(f"\n🔍 Verifying result...")
        verify_success, actual_state, fen_comparison, goal_check = self.verify(
            expected_fen, intent=intent
        )

        if verify_success:
            print(f"✅ Move completed successfully!")
            return True

        # 5. Recover if verification failed
        print(f"\n🔧 Move verification failed, attempting recovery...")
        return self.recover(intent, expected_fen, actual_state, fen_comparison)

    def run_one_move(self) -> bool:
        """Execute one complete move cycle with full reasoning loop.

        Returns:
            True if move succeeded
        """
        # 1. Sense
        print("\n" + "=" * 60)
        print("SENSING")
        print("=" * 60)
        overhead, wrist = self.sense()

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
