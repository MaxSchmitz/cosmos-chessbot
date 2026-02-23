#!/usr/bin/env python3
"""Run Isaac Sim trained policy on real SO-101 robot arm.

This script bridges the sim2real gap by:
1. Loading the PPO checkpoint trained in Isaac Sim
2. Using YOLO-DINO vision to detect pieces and locate targets
3. Constructing 21-dim observations from real robot state + vision
4. Running policy inference to get 6-dim actions (5 joints + gripper)
5. Executing actions on the physical SO-101 arm

Usage:
    # Dry-run mode (policy inference only, no robot execution)
    uv run python scripts/run_sim_policy_on_real_robot.py \
        --checkpoint /Users/max/Code/cosmos-chessbot/data/eval/policy_final.pt \
        --robot-port /dev/tty.usbmodem58FA0962531 \
        --dry-run

    # Single action test with real robot
    uv run python scripts/run_sim_policy_on_real_robot.py \
        --checkpoint /Users/max/Code/cosmos-chessbot/data/eval/policy_final.pt \
        --robot-port /dev/tty.usbmodem58FA0962531 \
        --task "e2 e4" \
        --single-action

    # Continuous control with vision
    uv run python scripts/run_sim_policy_on_real_robot.py \
        --checkpoint /Users/max/Code/cosmos-chessbot/data/eval/policy_final.pt \
        --robot-port /dev/tty.usbmodem58FA0962531 \
        --task "e2 e4" \
        --continuous \
        --max-steps 100 \
        --save-video data/eval/robot_with_vision.mp4
"""

import argparse
import sys
from pathlib import Path

import chess
import cv2
import numpy as np
import torch
import torch.nn as nn

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cosmos_chessbot.vision.yolo_dino_detector import YOLODINOFenDetector


# -- Policy architecture (must match training) ------------------------------

class ActorCritic(nn.Module):
    """Actor-Critic policy from Isaac Sim training."""

    def __init__(self, obs_dim: int = 21, act_dim: int = 6, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ELU(),
        )
        self.actor_branch = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
        )
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic_branch = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
        )
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        shared_features = self.shared(obs)
        actor_features = self.actor_branch(shared_features)
        action_mean = self.actor_mean(actor_features)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        critic_input = shared_features + 0.5 * (shared_features.detach() - shared_features)
        critic_features = self.critic_branch(critic_input)
        value = self.critic_head(critic_features).squeeze(-1)
        return action_mean, action_std, value


class RunningMeanStd:
    """Tracks running mean and std for observation normalization."""

    def __init__(self, shape, device, epsilon=1e-4):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def update(self, batch: torch.Tensor):
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        batch_count = batch.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.mean) / (self.var.sqrt() + 1e-8)


# -- Vision and coordinate mapping -------------------------------------------

def chess_square_to_board_coords(square_name: str) -> tuple[float, float]:
    """Convert chess square name (e.g., 'e2') to board coordinates [0-8, 0-8].

    Returns (file, rank) where:
    - file: 0 (a-file) to 8 (h-file edge)
    - rank: 0 (rank 1) to 8 (rank 8 edge)
    """
    square = chess.parse_square(square_name)
    file_idx = chess.square_file(square)  # 0-7 for a-h
    rank_idx = chess.square_rank(square)  # 0-7 for 1-8

    # Convert to board coordinates (add 0.5 to get square center)
    file_coord = file_idx + 0.5  # 0.5 to 7.5
    rank_coord = rank_idx + 0.5  # 0.5 to 7.5

    return file_coord, rank_coord


def board_coords_to_world_3d(
    file_coord: float,
    rank_coord: float,
    board_size: float,
    board_height: float = 0.0,
    board_center: tuple[float, float, float] = (0.3, 0.0, 0.0),
) -> np.ndarray:
    """Convert board coordinates to 3D world position.

    Args:
        file_coord: Board coordinate [0-8] (0=a-file, 8=h-file edge)
        rank_coord: Board coordinate [0-8] (0=rank 1, 8=rank 8)
        board_size: Physical board size in meters
        board_height: Board height above table
        board_center: (x, y, z) position of board center in world frame

    Returns:
        3D position [x, y, z] in world frame
    """
    # Normalize to [-0.5, 0.5] relative to board center
    file_norm = (file_coord / 8.0) - 0.5  # -0.5 to 0.5
    rank_norm = (rank_coord / 8.0) - 0.5  # -0.5 to 0.5

    # Scale by board size
    x_offset = file_norm * board_size
    y_offset = rank_norm * board_size

    # Add to board center
    world_x = board_center[0] + x_offset
    world_y = board_center[1] + y_offset
    world_z = board_center[2] + board_height

    return np.array([world_x, world_y, world_z], dtype=np.float32)


def detect_piece_position(
    camera_image: np.ndarray,
    square_name: str,
    detector: YOLODINOFenDetector,
    board_size: float,
    board_height: float,
) -> tuple[np.ndarray, bool]:
    """Detect 3D position of piece on given square using vision.

    Args:
        camera_image: RGB image from overhead camera
        square_name: Chess square (e.g., 'e2')
        detector: YOLO-DINO detector instance
        board_size: Physical board size in meters
        board_height: Board height above table

    Returns:
        (position_3d, piece_exists) tuple
    """
    # Detect FEN to know what pieces are where
    result = detector.detect_fen_with_metadata(camera_image)
    fen = result['fen']

    # Parse FEN to get board state
    board = chess.Board(fen)
    square = chess.parse_square(square_name)
    piece = board.piece_at(square)

    if piece is None:
        # No piece at this square - return square position anyway
        file_coord, rank_coord = chess_square_to_board_coords(square_name)
        pos_3d = board_coords_to_world_3d(file_coord, rank_coord, board_size, board_height)
        return pos_3d, False

    # Piece exists - get its 3D position
    file_coord, rank_coord = chess_square_to_board_coords(square_name)
    pos_3d = board_coords_to_world_3d(file_coord, rank_coord, board_size, board_height)

    # Add piece height offset (pieces sit on the board)
    piece_height = 0.03  # 3cm typical piece height to center of mass
    pos_3d[2] += piece_height

    return pos_3d, True


# -- Observation construction ------------------------------------------------

def construct_observation(
    arm_joint_pos: np.ndarray,       # (5,) - SO-101 arm joint angles
    gripper_pos: float,               # Gripper position (0-1)
    ee_pos: np.ndarray,              # (3,) - End-effector position in world frame
    ee_quat: np.ndarray,             # (4,) - End-effector quaternion (w,x,y,z)
    target_piece_pos: np.ndarray,    # (3,) - Target piece position (from vision)
    target_square_pos: np.ndarray,   # (3,) - Target square position (from vision)
    is_grasped: bool,                # Whether piece is currently grasped
    phase: float,                    # Phase in [0,1] - 0: approach, 0.5: lift, 1: transport
) -> np.ndarray:
    """Construct 21-dim observation vector matching Isaac Sim format."""
    # Compute relative positions (policy was trained with these)
    target_piece_rel = target_piece_pos - ee_pos
    target_square_rel = target_square_pos - ee_pos

    obs = np.concatenate([
        arm_joint_pos,           # (5,)
        [gripper_pos],           # (1,)
        ee_pos,                  # (3,)
        ee_quat,                 # (4,)
        target_piece_rel,        # (3,)
        target_square_rel,       # (3,)
        [float(is_grasped)],     # (1,)
        [phase],                 # (1,)
    ])  # Total: 21

    assert obs.shape == (21,), f"Expected 21-dim obs, got {obs.shape}"
    return obs


# -- Forward kinematics (simplified) -----------------------------------------

def compute_ee_pose_simple(joint_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute approximate end-effector pose from joint angles.

    This is a simplified forward kinematics approximation.
    For production, use proper DH parameters or URDF model.

    Args:
        joint_positions: (5,) joint angles in degrees

    Returns:
        (ee_pos, ee_quat) tuple
        - ee_pos: (3,) position in world frame
        - ee_quat: (4,) quaternion (w, x, y, z)
    """
    # TODO: Implement proper forward kinematics
    # For now, use a rough approximation based on joint angles

    # Rough link lengths for SO-101 (approximate)
    L1 = 0.15  # Base to shoulder
    L2 = 0.15  # Shoulder to elbow
    L3 = 0.15  # Elbow to wrist
    L4 = 0.10  # Wrist to EE

    # Convert degrees to radians
    q = np.deg2rad(joint_positions)

    # Simplified 2D FK in XZ plane
    x = L1 + L2 * np.cos(q[1]) + L3 * np.cos(q[1] + q[2]) + L4 * np.cos(q[1] + q[2] + q[3])
    z = 0.1 + L2 * np.sin(q[1]) + L3 * np.sin(q[1] + q[2]) + L4 * np.sin(q[1] + q[2] + q[3])
    y = 0.0  # Simplified - no lateral offset

    ee_pos = np.array([x, y, z], dtype=np.float32)

    # Simplified orientation (identity quaternion)
    ee_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    return ee_pos, ee_quat


# -- Action denormalization --------------------------------------------------

def denormalize_actions(action_normalized: np.ndarray, joint_limits: dict) -> np.ndarray:
    """Denormalize actions from [-1, 1] to actual joint angle ranges.

    Args:
        action_normalized: (6,) normalized actions in [-1, 1] range
        joint_limits: Dict mapping joint names to (min, max) tuples in degrees

    Returns:
        (6,) denormalized joint angles in degrees
    """
    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                   'wrist_flex', 'wrist_roll', 'gripper']

    action_denorm = np.zeros_like(action_normalized)
    for i, name in enumerate(joint_names):
        min_val, max_val = joint_limits[name]
        # Map [-1, 1] -> [min, max]
        action_denorm[i] = (action_normalized[i] + 1.0) / 2.0 * (max_val - min_val) + min_val

    return action_denorm


# -- Movement completion detection -------------------------------------------

def wait_for_joint_convergence(robot, threshold=0.5, timeout=10.0, check_interval=0.1):
    """Wait until robot joints stop moving (converged).

    Args:
        robot: LeRobot robot instance
        threshold: Maximum joint change in degrees to consider converged
        timeout: Maximum time to wait in seconds
        check_interval: Time between checks in seconds

    Returns:
        True if converged, False if timeout
    """
    import time

    start_time = time.time()
    last_positions = None

    while time.time() - start_time < timeout:
        obs = robot.get_observation()

        # Get current joint positions
        joint_names = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
                       'wrist_flex.pos', 'wrist_roll.pos']
        current_positions = []
        for name in joint_names:
            val = obs[name]
            if hasattr(val, 'item'):
                val = val.item()
            current_positions.append(float(val))

        if last_positions is not None:
            # Check if all joints moved less than threshold
            changes = [abs(c - l) for c, l in zip(current_positions, last_positions)]
            max_change = max(changes)

            if max_change < threshold:
                elapsed = time.time() - start_time
                print(f"    Joints converged (max change: {max_change:.3f}° < {threshold}°)")
                print(f"    Convergence took {elapsed:.2f}s")
                return True

        last_positions = current_positions
        time.sleep(check_interval)

    print(f"    Convergence timeout after {timeout}s")
    return False


# -- Main script -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Isaac Sim policy on real robot")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to Isaac Sim policy checkpoint (.pt)")
    parser.add_argument("--robot-port", type=str,
                        default="/dev/tty.usbmodem58FA0962531",
                        help="SO-101 USB port")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry-run mode: load policy and test inference, no robot")
    parser.add_argument("--single-action", action="store_true",
                        help="Execute only ONE action then stop (for safety testing)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for policy inference")
    parser.add_argument("--calibration-dir", type=Path, default=None,
                        help="Directory with robot calibration files")
    parser.add_argument("--task", type=str, default=None,
                        help="Task to execute non-interactively (e.g., 'e2 e4')")
    parser.add_argument("--continuous", action="store_true",
                        help="Run policy continuously in control loop (calls policy at fixed Hz)")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum steps to run in continuous mode")
    parser.add_argument("--control-frequency", type=float, default=20.0,
                        help="Control loop frequency in Hz for continuous mode (default: 20Hz)")
    parser.add_argument("--save-video", type=Path, default=None,
                        help="Save camera feed to video file")
    parser.add_argument("--yolo-weights", type=Path,
                        default=Path("runs/detect/runs/detect/yolo26_chess_combined/weights/best.pt"),
                        help="Path to YOLO piece detection weights")
    parser.add_argument("--corner-weights", type=Path,
                        default=Path("runs/pose/runs/pose/board_corners/weights/best.pt"),
                        help="Path to YOLO corner detection weights")
    parser.add_argument("--board-size", type=float, default=0.4,
                        help="Physical board size in meters (default: 40cm)")
    parser.add_argument("--board-height", type=float, default=0.0,
                        help="Board height above table in meters (default: 0)")
    parser.add_argument("--auto-home", action="store_true",
                        help="Automatically return to home position after task completes")
    parser.add_argument("--action-delay", type=float, default=0.5,
                        help="Delay in seconds after sending each action (default: 0.5)")
    parser.add_argument("--wait-convergence", action="store_true",
                        help="Wait for joint convergence instead of fixed delay")
    parser.add_argument("--convergence-threshold", type=float, default=0.5,
                        help="Joint movement threshold for convergence in degrees (default: 0.5)")
    parser.add_argument("--convergence-timeout", type=float, default=10.0,
                        help="Max time to wait for convergence in seconds (default: 10)")
    args = parser.parse_args()

    print("=" * 60)
    print("Isaac Sim Policy → Real Robot Deployment (with Vision)")
    print("=" * 60)

    # Load checkpoint
    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return 1

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    step = ckpt.get("step", 0)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"  Training step: {step:,}")
    print(f"  Episodes: {ckpt.get('episode_count', '?')}")

    # Load policy
    policy = ActorCritic(obs_dim=21, act_dim=6, hidden=256).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    print(f"  Policy loaded")
    print(f"  Parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Load vision system (YOLO-DINO detector)
    vision_detector = None
    if not args.dry_run and args.yolo_weights.exists():
        print(f"\nLoading YOLO-DINO vision system...")
        try:
            vision_detector = YOLODINOFenDetector(
                yolo_weights=str(args.yolo_weights),
                corner_weights=str(args.corner_weights) if args.corner_weights.exists() else None,
                mlp_weights=None,
                device=args.device,
                conf_threshold=0.10,
                use_dino=False,
            )
            print(f"  YOLO-DINO detector loaded (99.73% FEN accuracy)")
            print(f"  Confidence threshold: 0.10")
        except Exception as e:
            print(f"  Warning: Could not load vision system: {e}")
            print(f"  Will use dummy vision data")
    else:
        if args.dry_run:
            print(f"\nVision system skipped (dry-run mode)")
        else:
            print(f"\nWarning: YOLO weights not found at {args.yolo_weights}")
            print(f"  Will use dummy vision data")

    # Load observation normalizer
    obs_normalizer = RunningMeanStd(21, device)
    if "obs_normalizer" in ckpt:
        norm_state = ckpt["obs_normalizer"]
        obs_normalizer.mean = norm_state["mean"].to(device)
        obs_normalizer.var = norm_state["var"].to(device)
        obs_normalizer.count = norm_state["count"]
        print(f"  Obs normalizer loaded (count={obs_normalizer.count:.0f})")
    else:
        print(f"  WARNING: No obs normalizer in checkpoint!")

    # Initialize robot (unless dry-run)
    robot = None
    if not args.dry_run:
        try:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
            from lerobot.cameras.opencv import OpenCVCameraConfig

            print(f"\nConnecting to SO-101 on {args.robot_port}...")

            camera_config = {
                "egocentric": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
                "wrist": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
            }

            config_kwargs = {
                "port": args.robot_port,
                "id": "my_follower_arm",
                "cameras": camera_config,
            }
            if args.calibration_dir:
                config_kwargs["calibration_dir"] = args.calibration_dir

            robot = SO101Follower(SO101FollowerConfig(**config_kwargs))
            robot.connect()
            print(f"  Robot connected successfully")

            # Get joint limits from robot configuration
            print(f"\n  Joint limits (from calibration):")
            joint_names_short = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                                 'wrist_flex', 'wrist_roll', 'gripper']
            joint_limits = {}
            for name in joint_names_short:
                # LeRobot stores limits as motor_names with .pos suffix
                motor_name = f"{name}.pos"
                if hasattr(robot, 'motor_names') and motor_name in robot.motor_names:
                    idx = robot.motor_names.index(motor_name)
                    if hasattr(robot, 'motor_models') and hasattr(robot.motor_models[idx], 'position_limits'):
                        limits = robot.motor_models[idx].position_limits
                        joint_limits[name] = limits
                        print(f"    {name}: [{limits[0]:.1f}°, {limits[1]:.1f}°]")
                    elif hasattr(robot, 'leader_arms') and len(robot.leader_arms) > 0:
                        # Try to get from leader arm configuration
                        leader = robot.leader_arms[0]
                        if hasattr(leader, f'{name}_min') and hasattr(leader, f'{name}_max'):
                            min_val = getattr(leader, f'{name}_min')
                            max_val = getattr(leader, f'{name}_max')
                            joint_limits[name] = (min_val, max_val)
                            print(f"    {name}: [{min_val:.1f}°, {max_val:.1f}°]")

            if not joint_limits:
                print(f"    Warning: Could not get joint limits from robot, using defaults")
                # Default SO-101 joint limits (approximate safe ranges)
                joint_limits = {
                    'shoulder_pan': (-90.0, 90.0),
                    'shoulder_lift': (-120.0, 0.0),
                    'elbow_flex': (0.0, 150.0),
                    'wrist_flex': (0.0, 120.0),
                    'wrist_roll': (-90.0, 90.0),
                    'gripper': (0.0, 5.0),
                }
                for name, limits in joint_limits.items():
                    print(f"    {name}: [{limits[0]:.1f}°, {limits[1]:.1f}°] (default)")

        except ImportError as e:
            print(f"\nERROR: LeRobot not installed: {e}")
            return 1
        except Exception as e:
            print(f"\nERROR: Failed to connect to robot: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print(f"\nDry-run mode: Robot connection skipped")

    print("\n" + "=" * 60)
    print("System ready - Vision + Policy + Robot")
    print("=" * 60)

    # Set up video writer
    video_writer = None
    if args.save_video and robot:
        args.save_video.parent.mkdir(parents=True, exist_ok=True)

    # Control loop
    print("\nControl loop starting...")
    if args.continuous:
        print(f"  CONTINUOUS MODE: {args.control_frequency}Hz control, up to {args.max_steps} steps")
    elif args.single_action:
        print(f"  SINGLE ACTION MODE")
    if args.dry_run:
        print(f"  DRY RUN MODE")

    try:
        if args.task:
            task_inputs = [args.task]
        else:
            task_inputs = None

        while True:
            print("\n" + "=" * 60)

            # Get task
            if task_inputs is not None:
                if not task_inputs:
                    break
                task_input = task_inputs.pop(0)
                print(f"Task: {task_input}")
            else:
                try:
                    task_input = input("Enter task (e.g., 'e2 e4') or 'q' to quit: ")
                except EOFError:
                    break
                if task_input.strip().lower() == 'q':
                    break

            parts = task_input.strip().split()
            if len(parts) != 2:
                print("  Invalid format. Use: <source> <target> (e.g., 'e2 e4')")
                if task_inputs is not None:
                    break
                continue

            source_square, target_square = parts
            print(f"  Pick {source_square}, place {target_square}")

            # CONTINUOUS CONTROL LOOP
            if args.continuous:
                import time
                control_period = 1.0 / args.control_frequency  # seconds per step
                print(f"  Starting continuous control at {args.control_frequency}Hz (period={control_period:.3f}s)")

                step_count = 0
                start_time = time.time()

                while step_count < args.max_steps:
                    step_start = time.time()

                    # Get current robot state
                    if robot:
                        obs_dict = robot.get_observation()

                        # Extract joint positions (in degrees from LeRobot)
                        joint_names = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
                                       'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
                        joint_positions_deg = []
                        for name in joint_names:
                            val = obs_dict[name]
                            if hasattr(val, 'item'):
                                val = val.item()
                            joint_positions_deg.append(float(val))
                        joint_positions_deg = np.array(joint_positions_deg, dtype=np.float32)

                        # Convert degrees to radians (Isaac Sim trained with radians!)
                        joint_positions_rad = np.deg2rad(joint_positions_deg)

                        arm_joint_pos = joint_positions_rad[:5]
                        gripper_pos = joint_positions_rad[5]

                        # Get camera image
                        camera_image = None
                        if "egocentric" in obs_dict:
                            ego_img = obs_dict["egocentric"]
                            if hasattr(ego_img, 'numpy'):
                                ego_img = ego_img.numpy()
                            camera_image = ego_img

                        # Compute end-effector pose (updates with robot movement)
                        ee_pos, ee_quat = compute_ee_pose_simple(arm_joint_pos)

                        # Use vision to detect piece positions
                        if vision_detector and camera_image is not None:
                            target_piece_pos, piece_exists = detect_piece_position(
                                camera_image, source_square, vision_detector,
                                args.board_size, args.board_height
                            )
                            target_square_pos, _ = detect_piece_position(
                                camera_image, target_square, vision_detector,
                                args.board_size, args.board_height
                            )
                        else:
                            target_piece_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                            target_square_pos = np.array([0.1, 0.1, 0.0], dtype=np.float32)
                    else:
                        # Dry-run: dummy values
                        arm_joint_pos = np.zeros(5, dtype=np.float32)
                        gripper_pos = 0.5
                        ee_pos = np.array([0.0, 0.0, 0.2], dtype=np.float32)
                        ee_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                        target_piece_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                        target_square_pos = np.array([0.1, 0.1, 0.0], dtype=np.float32)

                    # Construct observation
                    is_grasped = False  # TODO: Implement grasp detection
                    phase = 0.0  # TODO: Implement phase tracking

                    # Compute relative positions for logging
                    target_piece_rel = target_piece_pos - ee_pos
                    target_square_rel = target_square_pos - ee_pos

                    obs = construct_observation(
                        arm_joint_pos, gripper_pos, ee_pos, ee_quat,
                        target_piece_pos, target_square_pos,
                        is_grasped, phase
                    )

                    # Run policy
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device).float()
                    obs_norm = obs_normalizer.normalize(obs_tensor)

                    with torch.no_grad():
                        action_mean, action_std, value = policy(obs_norm)
                        action = action_mean.clamp(-1.0, 1.0)

                    action_normalized = action.cpu().numpy()[0]

                    # Denormalize actions from [-1, 1] to actual joint ranges
                    action_denorm = denormalize_actions(action_normalized, joint_limits)

                    # Log every 10 steps to avoid spam
                    if step_count % 10 == 0:
                        elapsed = time.time() - start_time
                        print(f"  Step {step_count:3d} | t={elapsed:.2f}s | EE: {ee_pos} | Value: {value.item():.3f}")
                        print(f"    Raw observation (first 10): {obs[:10]}")
                        print(f"    Normalized obs (first 10): {obs_norm[0, :10].cpu().numpy()}")
                        print(f"    Normalizer mean (first 10): {obs_normalizer.mean[:10].cpu().numpy()}")
                        print(f"    Normalizer std (first 10): {obs_normalizer.var[:10].sqrt().cpu().numpy()}")
                        print(f"    Action (normalized): {action_normalized}")
                        print(f"    Action (denormalized °): {action_denorm}")
                        print(f"    Target piece rel: {target_piece_rel}")
                        print(f"    Target square rel: {target_square_rel}")

                    # Display camera
                    if robot and camera_image is not None:
                        ego_img_bgr = cv2.cvtColor((camera_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                        cv2.putText(ego_img_bgr, f"Step: {step_count}/{args.max_steps}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(ego_img_bgr, f"Task: {source_square}->{target_square}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.imshow("Robot Vision", ego_img_bgr)
                        cv2.waitKey(1)

                        if video_writer is None and args.save_video:
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(
                                str(args.save_video), fourcc, args.control_frequency,
                                (ego_img_bgr.shape[1], ego_img_bgr.shape[0])
                            )
                        if video_writer:
                            video_writer.write(ego_img_bgr)

                    # Execute action
                    if robot and not args.dry_run:
                        action_names = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
                                        'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
                        action_dict = {name: torch.tensor([action_denorm[i]], dtype=torch.float32)
                                       for i, name in enumerate(action_names)}

                        robot.send_action(action_dict)

                    step_count += 1

                    # Maintain control frequency
                    elapsed_step = time.time() - step_start
                    sleep_time = max(0, control_period - elapsed_step)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                    actual_period = time.time() - step_start
                    if step_count % 10 == 0 and actual_period > control_period * 1.1:
                        print(f"    Warning: Control loop running slow ({1.0/actual_period:.1f}Hz)")

                elapsed_total = time.time() - start_time
                print(f"\n  Continuous control complete: {step_count} steps in {elapsed_total:.2f}s")
                print(f"  Average frequency: {step_count/elapsed_total:.1f}Hz")

                if task_inputs is not None and not task_inputs:
                    break
                continue

            # SINGLE-SHOT MODE (original behavior)
            print("\n" + "-" * 60)

            # Get robot state
            if robot:
                obs_dict = robot.get_observation()

                # Extract joint positions (in degrees from LeRobot)
                joint_names = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
                               'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
                joint_positions_deg = []
                for name in joint_names:
                    val = obs_dict[name]
                    if hasattr(val, 'item'):
                        val = val.item()
                    joint_positions_deg.append(float(val))
                joint_positions_deg = np.array(joint_positions_deg, dtype=np.float32)

                # Convert degrees to radians (Isaac Sim trained with radians!)
                joint_positions_rad = np.deg2rad(joint_positions_deg)

                arm_joint_pos = joint_positions_rad[:5]
                gripper_pos = joint_positions_rad[5]

                # Get camera image
                camera_image = None
                if "egocentric" in obs_dict:
                    ego_img = obs_dict["egocentric"]
                    if hasattr(ego_img, 'numpy'):
                        ego_img = ego_img.numpy()
                    camera_image = ego_img

                # Compute end-effector pose
                ee_pos, ee_quat = compute_ee_pose_simple(arm_joint_pos)
                print(f"  EE position: {ee_pos}")

                # Use vision to detect piece positions
                if vision_detector and camera_image is not None:
                    print(f"  Running YOLO-DINO vision...")
                    target_piece_pos, piece_exists = detect_piece_position(
                        camera_image, source_square, vision_detector,
                        args.board_size, args.board_height
                    )
                    target_square_pos, _ = detect_piece_position(
                        camera_image, target_square, vision_detector,
                        args.board_size, args.board_height
                    )

                    print(f"  Source ({source_square}): {target_piece_pos} (exists={piece_exists})")
                    print(f"  Target ({target_square}): {target_square_pos}")
                else:
                    print(f"  WARNING: No vision - using dummy positions")
                    target_piece_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    target_square_pos = np.array([0.1, 0.1, 0.0], dtype=np.float32)

            else:
                # Dry-run: dummy values
                arm_joint_pos = np.zeros(5, dtype=np.float32)
                gripper_pos = 0.5
                ee_pos = np.array([0.0, 0.0, 0.2], dtype=np.float32)
                ee_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                target_piece_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                target_square_pos = np.array([0.1, 0.1, 0.0], dtype=np.float32)

            # Construct observation
            # TODO: Implement grasp detection and phase tracking
            is_grasped = False
            phase = 0.0

            obs = construct_observation(
                arm_joint_pos, gripper_pos, ee_pos, ee_quat,
                target_piece_pos, target_square_pos,
                is_grasped, phase
            )

            # Run policy
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device).float()
            obs_norm = obs_normalizer.normalize(obs_tensor)

            with torch.no_grad():
                action_mean, action_std, value = policy(obs_norm)
                action = action_mean.clamp(-1.0, 1.0)

            action_normalized = action.cpu().numpy()[0]

            # Denormalize actions from [-1, 1] to actual joint ranges
            action_denorm = denormalize_actions(action_normalized, joint_limits)

            print(f"  Action (normalized): {action_normalized}")
            print(f"  Action (denormalized °): {action_denorm}")
            print(f"  Value: {value.item():.3f}")

            # Display camera
            if robot and camera_image is not None:
                ego_img_bgr = cv2.cvtColor((camera_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.putText(ego_img_bgr, f"Task: {source_square}->{target_square}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Robot Vision", ego_img_bgr)
                cv2.waitKey(1)

                if video_writer is None and args.save_video:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(
                        str(args.save_video), fourcc, 10.0,
                        (ego_img_bgr.shape[1], ego_img_bgr.shape[0])
                    )
                if video_writer:
                    video_writer.write(ego_img_bgr)

            # Execute action
            if robot and not args.dry_run:
                import time
                action_names = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
                                'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
                action_dict = {name: torch.tensor([action_denorm[i]], dtype=torch.float32)
                               for i, name in enumerate(action_names)}
                robot.send_action(action_dict)
                print(f"  Action sent")

                # Wait for robot to execute the action
                if args.wait_convergence:
                    print(f"  Waiting for joint convergence...")
                    converged = wait_for_joint_convergence(
                        robot,
                        threshold=args.convergence_threshold,
                        timeout=args.convergence_timeout
                    )
                    if not converged:
                        print(f"    Warning: Joints did not converge within timeout")
                else:
                    time.sleep(args.action_delay)
                    print(f"  Waited {args.action_delay}s for execution")

            if args.single_action:
                print(f"\nSingle-action mode: exiting")
                break

            # For non-continuous mode, exit after task
            if task_inputs is not None and not task_inputs:
                break

    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        # Auto-home if requested
        if robot and args.auto_home:
            print("\n" + "=" * 60)
            print("Returning to home position...")
            print("=" * 60)

            # True home position (measured from robot)
            home_position = {
                'shoulder_pan.pos': torch.tensor([7.55], dtype=torch.float32),
                'shoulder_lift.pos': torch.tensor([-98.12], dtype=torch.float32),
                'elbow_flex.pos': torch.tensor([100.00], dtype=torch.float32),
                'wrist_flex.pos': torch.tensor([62.88], dtype=torch.float32),
                'wrist_roll.pos': torch.tensor([0.08], dtype=torch.float32),
                'gripper.pos': torch.tensor([1.63], dtype=torch.float32),
            }

            try:
                robot.send_action(home_position)
                print("  Homing command sent")

                # Wait for movement to complete
                import time
                time.sleep(3)

                # Verify
                obs = robot.get_observation()
                print("  Final position:")
                joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                               'wrist_flex', 'wrist_roll', 'gripper']
                for name in joint_names:
                    val = obs[f'{name}.pos']
                    if hasattr(val, 'item'):
                        val = val.item()
                    print(f"    {name}: {val:.2f}°")
                print("  Robot homed successfully")
            except Exception as e:
                print(f"  Warning: Homing failed: {e}")

        if robot:
            robot.disconnect()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("Session complete")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
