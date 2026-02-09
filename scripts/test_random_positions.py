#!/usr/bin/env python3
"""Test robot by moving to 3 random positions then returning home.

Usage:
    uv run python scripts/test_random_positions.py \
        --checkpoint data/eval/policy_final.pt \
        --robot-port /dev/tty.usbmodem58FA0962531
"""

import argparse
import random
import sys
from pathlib import Path

import chess
import torch

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig


def wait_for_joint_convergence(robot, threshold=0.5, timeout=10.0, check_interval=0.1):
    """Wait until robot joints stop moving."""
    import time

    start_time = time.time()
    last_positions = None

    while time.time() - start_time < timeout:
        obs = robot.get_observation()

        joint_names = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
                       'wrist_flex.pos', 'wrist_roll.pos']
        current_positions = []
        for name in joint_names:
            val = obs[name]
            if hasattr(val, 'item'):
                val = val.item()
            current_positions.append(float(val))

        if last_positions is not None:
            changes = [abs(c - l) for c, l in zip(current_positions, last_positions)]
            max_change = max(changes)

            if max_change < threshold:
                elapsed = time.time() - start_time
                print(f"    Converged in {elapsed:.2f}s (max change: {max_change:.3f}°)")
                return True

        last_positions = current_positions
        time.sleep(check_interval)

    print(f"    Timeout after {timeout}s")
    return False


def send_home(robot):
    """Send robot to home position."""
    home_position = {
        'shoulder_pan.pos': torch.tensor([7.55], dtype=torch.float32),
        'shoulder_lift.pos': torch.tensor([-98.12], dtype=torch.float32),
        'elbow_flex.pos': torch.tensor([100.00], dtype=torch.float32),
        'wrist_flex.pos': torch.tensor([62.88], dtype=torch.float32),
        'wrist_roll.pos': torch.tensor([0.08], dtype=torch.float32),
        'gripper.pos': torch.tensor([1.63], dtype=torch.float32),
    }
    robot.send_action(home_position)


def main():
    parser = argparse.ArgumentParser(description="Test robot with 3 random positions")
    parser.add_argument("--robot-port", type=str, default="/dev/tty.usbmodem58FA0962531")
    parser.add_argument("--num-positions", type=int, default=3,
                        help="Number of random positions to test (default: 3)")
    args = parser.parse_args()

    print("=" * 60)
    print("Random Position Test")
    print("=" * 60)

    # Connect to robot
    print("\nConnecting to robot...")
    camera_config = {
        "egocentric": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
        "wrist": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
    }

    robot = SO101Follower(SO101FollowerConfig(
        port=args.robot_port,
        id="my_follower_arm",
        cameras=camera_config,
    ))
    robot.connect()
    print("  Robot connected")

    # Generate random chess squares
    all_squares = [chess.square_name(sq) for sq in range(64)]
    random_squares = random.sample(all_squares, args.num_positions)

    print(f"\nRandom target squares: {', '.join(random_squares)}")
    print("\n" + "=" * 60)

    try:
        for i, square in enumerate(random_squares, 1):
            print(f"\nPosition {i}/{args.num_positions}: {square}")
            print("-" * 60)

            # Generate random joint positions (safe ranges)
            # These are normalized values that the robot can safely reach
            random_action = {
                'shoulder_pan.pos': torch.tensor([random.uniform(-20, 20)], dtype=torch.float32),
                'shoulder_lift.pos': torch.tensor([random.uniform(-90, -30)], dtype=torch.float32),
                'elbow_flex.pos': torch.tensor([random.uniform(45, 120)], dtype=torch.float32),
                'wrist_flex.pos': torch.tensor([random.uniform(30, 90)], dtype=torch.float32),
                'wrist_roll.pos': torch.tensor([random.uniform(-10, 10)], dtype=torch.float32),
                'gripper.pos': torch.tensor([random.uniform(0, 2)], dtype=torch.float32),
            }

            print(f"  Target angles:")
            for name, val in random_action.items():
                print(f"    {name}: {val.item():.2f}°")

            # Send action
            robot.send_action(random_action)
            print(f"  Action sent")

            # Wait for convergence
            print(f"  Waiting for convergence...")
            wait_for_joint_convergence(robot, threshold=0.5, timeout=10.0)

            # Get final position
            obs = robot.get_observation()
            print(f"  Final position:")
            joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                           'wrist_flex', 'wrist_roll', 'gripper']
            for name in joint_names:
                val = obs[f'{name}.pos']
                if hasattr(val, 'item'):
                    val = val.item()
                print(f"    {name}: {val:.2f}°")

        # Return home
        print("\n" + "=" * 60)
        print("Returning to home position...")
        print("=" * 60)
        send_home(robot)
        print("  Homing command sent")

        import time
        time.sleep(3)

        # Verify home
        obs = robot.get_observation()
        print("  Final home position:")
        for name in joint_names:
            val = obs[f'{name}.pos']
            if hasattr(val, 'item'):
                val = val.item()
            print(f"    {name}: {val:.2f}°")

    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        robot.disconnect()
        print("\n" + "=" * 60)
        print("Test complete")
        print("=" * 60)


if __name__ == "__main__":
    main()
