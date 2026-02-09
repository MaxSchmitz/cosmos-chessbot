#!/usr/bin/env python3
"""Send SO-101 robot to home/rest position.

Usage:
    uv run python scripts/home_robot.py --port /dev/tty.usbmodem58FA0962531
"""

import argparse
import time

import torch
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig


def main():
    parser = argparse.ArgumentParser(description="Home SO-101 robot")
    parser.add_argument("--port", type=str, default="/dev/tty.usbmodem58FA0962531")
    args = parser.parse_args()

    print("Connecting to robot...")
    camera_config = {
        "egocentric": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
        "wrist": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
    }

    robot = SO101Follower(SO101FollowerConfig(
        port=args.port,
        id="my_follower_arm",
        cameras=camera_config,
    ))

    robot.connect()
    print("Robot connected")

    # Get current position
    obs = robot.get_observation()
    print("\nCurrent joint positions:")
    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
    for name in joint_names:
        val = obs[f'{name}.pos']
        if hasattr(val, 'item'):
            val = val.item()
        print(f"  {name}: {val:.2f}°")

    # Define home position (TRUE home state measured from robot)
    home_position = {
        'shoulder_pan.pos': torch.tensor([7.55], dtype=torch.float32),
        'shoulder_lift.pos': torch.tensor([-98.12], dtype=torch.float32),
        'elbow_flex.pos': torch.tensor([100.00], dtype=torch.float32),
        'wrist_flex.pos': torch.tensor([62.88], dtype=torch.float32),
        'wrist_roll.pos': torch.tensor([0.08], dtype=torch.float32),
        'gripper.pos': torch.tensor([1.63], dtype=torch.float32),
    }

    print("\nMoving to home position...")
    print("  shoulder_pan: 7.55°")
    print("  shoulder_lift: -98.12°")
    print("  elbow_flex: 100.00°")
    print("  wrist_flex: 62.88°")
    print("  wrist_roll: 0.08°")
    print("  gripper: 1.63°")

    # Send home position
    robot.send_action(home_position)

    # Wait for movement to complete
    print("\nWaiting for robot to reach home position...")
    time.sleep(3)

    # Verify final position
    obs = robot.get_observation()
    print("\nFinal joint positions:")
    for name in joint_names:
        val = obs[f'{name}.pos']
        if hasattr(val, 'item'):
            val = val.item()
        print(f"  {name}: {val:.2f}°")

    robot.disconnect()
    print("\nRobot homed and disconnected")


if __name__ == "__main__":
    main()
