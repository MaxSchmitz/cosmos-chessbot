#!/usr/bin/env python3
"""Calibrate SO-101 robot arm and save calibration.

This script connects to the SO-101 and guides you through calibration,
then saves the calibration data so you don't need to recalibrate every time.

Usage:
    uv run python scripts/calibrate_so101.py \
        --port /dev/tty.usbmodem58FA0962531 \
        --output-dir .lerobot/calibration/so101
"""

import argparse
from pathlib import Path

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig


def main():
    parser = argparse.ArgumentParser(description="Calibrate SO-101 robot")
    parser.add_argument("--port", type=str, required=True,
                        help="USB port for SO-101 (e.g., /dev/tty.usbmodem58FA0962531)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path.home() / ".lerobot" / "calibration" / "so101",
                        help="Directory to save calibration files")
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SO-101 Robot Calibration")
    print("=" * 60)
    print(f"Port: {args.port}")
    print(f"Calibration will be saved to: {args.output_dir}")
    print()
    print("INSTRUCTIONS:")
    print("  1. Move robot to middle of range, press ENTER")
    print("  2. Move each joint through its full range of motion")
    print("  3. Press ENTER when done")
    print("=" * 60)
    print()

    # Create robot (will trigger calibration)
    robot = SO101Follower(SO101FollowerConfig(
        port=args.port,
        id="calibration_robot",
        calibration_dir=args.output_dir,
    ))

    # Connect (this triggers calibration if needed)
    robot.connect()

    print()
    print("=" * 60)
    print("Calibration complete!")
    print(f"Calibration saved to: {args.output_dir}")
    print()
    print("You can now use this calibration with:")
    print(f"  --calibration-dir {args.output_dir}")
    print("=" * 60)

    # Disconnect
    robot.disconnect()


if __name__ == "__main__":
    main()
