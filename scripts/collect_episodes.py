#!/usr/bin/env python3
"""Collect teleoperation episodes using LeRobot.

This script records demonstrations of chess piece manipulation using the leader arm.
Episodes are saved in LeRobot format for later training.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.control_robot import record


def main():
    parser = argparse.ArgumentParser(
        description="Collect chess manipulation episodes with teleoperation"
    )
    parser.add_argument(
        "--robot-path",
        type=str,
        default="so100",
        help="Robot configuration path or type",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="cosmos-chessbot/chess-manipulation",
        help="Dataset repository ID",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/episodes"),
        help="Root directory for episode storage",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to collect in this session",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Recording frequency (Hz)",
    )
    parser.add_argument(
        "--warmup-time-s",
        type=float,
        default=3.0,
        help="Warmup time before recording starts (seconds)",
    )
    parser.add_argument(
        "--episode-time-s",
        type=float,
        default=30.0,
        help="Maximum episode duration (seconds)",
    )
    parser.add_argument(
        "--display-cameras",
        action="store_true",
        help="Display camera feeds during recording",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("LeRobot Chess Episode Collector")
    print("=" * 60)
    print(f"Robot: {args.robot_path}")
    print(f"Dataset: {args.repo_id}")
    print(f"Episodes to collect: {args.num_episodes}")
    print(f"Recording frequency: {args.fps} Hz")
    print(f"Episode duration: {args.episode_time_s}s")
    print()

    # Create dataset directory
    args.root.mkdir(parents=True, exist_ok=True)

    print("Initializing robot and cameras...")
    print()
    print("Instructions:")
    print("1. Use the leader arm to demonstrate chess piece manipulation")
    print("2. Pick up a piece and place it on a target square")
    print("3. Each episode should be one complete pick-and-place action")
    print("4. Try to vary:")
    print("   - Different pieces (pawns, knights, bishops, etc.)")
    print("   - Different source and target squares")
    print("   - Different approach angles")
    print("   - Captures (removing opponent pieces)")
    print()
    print(f"Recording will start after {args.warmup_time_s}s warmup.")
    print("Press Enter to start session, Ctrl+C to stop early.")
    print()

    input("Press Enter to begin...")

    # Use LeRobot's built-in recording function
    try:
        record(
            robot_path=args.robot_path,
            robot_overrides=[],
            fps=args.fps,
            root=str(args.root),
            repo_id=args.repo_id,
            warmup_time_s=args.warmup_time_s,
            episode_time_s=args.episode_time_s,
            reset_time_s=5.0,
            num_episodes=args.num_episodes,
            video=True,
            run_compute_stats=True,
            push_to_hub=False,  # Set to True if you want to upload
            tags=["chess", "manipulation", "cosmos-chessbot"],
            display_cameras=args.display_cameras,
        )
    except KeyboardInterrupt:
        print("\n\nRecording interrupted by user.")
        return 0

    print()
    print("=" * 60)
    print("Recording Complete!")
    print("=" * 60)
    print(f"Episodes saved to: {args.root / args.repo_id}")
    print()
    print("Next steps:")
    print("1. Review episodes: Check data quality and consistency")
    print("2. Collect more data: Aim for 100-200 episodes total")
    print("3. Train policy: Fine-tune π₀.₅ on your collected data")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
