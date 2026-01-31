"""Main control loop orchestrator."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..vision import Camera, CameraConfig, CosmosPerception, BoardState, RemoteCosmosPerception
from ..stockfish import StockfishEngine


@dataclass
class OrchestratorConfig:
    """Configuration for the chess orchestrator."""

    egocentric_camera_id: int = 0
    wrist_camera_id: int = 1
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

    def compile_intent(self, move: str, board_state: BoardState) -> dict:
        """Compile symbolic move into physical manipulation intent.

        Args:
            move: UCI move (e.g., 'e2e4')
            board_state: Current board state

        Returns:
            Intent dictionary with pick/place squares and constraints

        TODO: Use Cosmos-Reason2 to generate constraints and recovery strategies
        """
        # Parse move
        from_square = move[:2]
        to_square = move[2:4]

        return {
            "pick_square": from_square,
            "place_square": to_square,
            "constraints": {
                "approach": "from_above",
                "clearance": 0.05,  # meters
                "avoidance": [],  # squares to avoid
            },
            "recovery_strategy": "retry",
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

    def verify(self, expected_fen: str) -> tuple[bool, BoardState]:
        """Verify board state after action.

        Args:
            expected_fen: Expected FEN after move

        Returns:
            Tuple of (success, actual_board_state)
        """
        overhead, _ = self.sense()
        actual_state = self.perceive(overhead)

        # Simple FEN comparison (can be improved with fuzzy matching)
        success = actual_state.fen == expected_fen

        return success, actual_state

    def recover(self, intent: dict, failure_state: BoardState) -> bool:
        """Attempt to recover from failed execution.

        Args:
            intent: Original intent that failed
            failure_state: Actual board state after failure

        Returns:
            True if recovery succeeded

        TODO: Use Cosmos-Reason2 for failure diagnosis and recovery planning
        """
        print(f"TODO: Recover from failure. State: {failure_state}")
        return False

    def run_one_move(self) -> bool:
        """Execute one complete move cycle.

        Returns:
            True if move succeeded
        """
        # 1. Sense
        overhead, wrist = self.sense()

        # 2. Perceive
        board_state = self.perceive(overhead)
        print(f"Board state: {board_state.fen}")
        print(f"Confidence: {board_state.confidence}")
        if board_state.anomalies:
            print(f"Anomalies: {board_state.anomalies}")

        # 3. Plan
        best_move = self.plan(board_state)
        print(f"Best move: {best_move}")

        # 4. Compile intent
        intent = self.compile_intent(best_move, board_state)

        # 5. Execute
        exec_success = self.execute(intent)

        if not exec_success:
            return False

        # 6. Verify
        # TODO: Compute expected FEN after move
        expected_fen = board_state.fen  # Placeholder
        verify_success, actual_state = self.verify(expected_fen)

        if not verify_success:
            # 7. Recover
            return self.recover(intent, actual_state)

        return True
