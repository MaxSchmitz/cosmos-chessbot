"""Main control loop orchestrator."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..vision import Camera, CameraConfig, CosmosPerception, BoardState, RemoteCosmosPerception
from ..stockfish import StockfishEngine
from ..reasoning import (
    ChessGameReasoning,
    compare_fen_states,
    calculate_expected_fen,
    generate_correction_move,
)


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

        # Initialize Cosmos perception (local or remote)
        if config.cosmos_server_url:
            print(f"Using remote Cosmos server at {config.cosmos_server_url}")
            self.perception = RemoteCosmosPerception(server_url=config.cosmos_server_url)
        else:
            print(f"Using local Cosmos model: {config.cosmos_model}")
            self.perception = CosmosPerception(model_name=config.cosmos_model)

        # Initialize Stockfish
        self.engine = StockfishEngine(engine_path=config.stockfish_path)

        # Initialize game reasoning (for pre-action, verification, recovery)
        if not config.cosmos_server_url:
            print("Initializing Cosmos game reasoning...")
            self.game_reasoning = ChessGameReasoning(model_name=config.cosmos_model)
        else:
            # For remote Cosmos, we'd need a remote reasoning client (TODO)
            print("WARNING: Remote game reasoning not yet implemented")
            self.game_reasoning = None

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
            overhead_image: PIL Image from egocentric camera

        Returns:
            BoardState with FEN, confidence, and anomalies
        """
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

        Uses Cosmos-Reason2 to reason about physical constraints.

        Args:
            move: UCI move (e.g., 'e2e4')
            board_state: Current board state
            image: Current egocentric image (for reasoning)

        Returns:
            Intent dictionary with pick/place squares and constraints
        """
        # Parse move
        from_square = move[:2]
        to_square = move[2:4]

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

        return {
            "pick_square": from_square,
            "place_square": to_square,
            "move_uci": move,
            "action_reasoning": action_reasoning,
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

    def verify(self, expected_fen: str, vision_backend="yolo") -> tuple[bool, BoardState, "FENComparison"]:
        """Verify board state after action using FEN comparison.

        Args:
            expected_fen: Expected FEN after move
            vision_backend: Which vision system to use ("cosmos" or "yolo")

        Returns:
            Tuple of (success, actual_board_state, fen_comparison)
        """
        overhead, _ = self.sense()

        # Extract FEN using selected backend
        if vision_backend == "cosmos":
            actual_state = self.perceive(overhead)
            actual_fen = actual_state.fen
        else:
            # TODO: Use YOLO-DINO detector when available
            actual_state = self.perceive(overhead)
            actual_fen = actual_state.fen

        # Compare FENs
        fen_comparison = compare_fen_states(expected_fen, actual_fen)

        if not fen_comparison.match:
            print("❌ Verification failed!")
            print(fen_comparison.summary())
        else:
            print("✅ Verification passed!")

        return fen_comparison.match, actual_state, fen_comparison

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
            verify_success, new_state, new_comparison = self.verify(expected_fen)

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

        # 4. Verify using FEN comparison
        print(f"\n🔍 Verifying result...")
        verify_success, actual_state, fen_comparison = self.verify(expected_fen)

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
        print("👁️  SENSING")
        print("=" * 60)
        overhead, wrist = self.sense()

        # 2. Perceive
        print("\n🧠 PERCEIVING")
        board_state = self.perceive(overhead)
        print(f"   FEN: {board_state.fen}")
        print(f"   Confidence: {board_state.confidence:.2%}")
        if board_state.anomalies:
            print(f"   Anomalies: {board_state.anomalies}")

        # 3. Plan
        print("\n♟️  PLANNING")
        best_move = self.plan(board_state)
        print(f"   Best move: {best_move}")

        # 4. Execute with verification and recovery
        return self.execute_move_with_verification(
            move=best_move,
            current_fen=board_state.fen,
            current_image=overhead,
        )
