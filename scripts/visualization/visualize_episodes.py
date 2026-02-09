#!/usr/bin/env python3
"""Visualize collected episodes to verify data quality."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.visualize_dataset import visualize_dataset


def main():
    parser = argparse.ArgumentParser(description="Visualize collected episodes")
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
        "--episode-index",
        type=int,
        default=0,
        help="Episode to visualize (default: 0 = first episode)",
    )

    args = parser.parse_args()

    dataset_path = args.root / args.repo_id

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("\nCollect episodes first:")
        print("  uv run python scripts/collect_episodes.py")
        return 1

    print(f"Loading dataset from {dataset_path}...")
    dataset = LeRobotDataset(args.repo_id, root=str(args.root))

    print(f"Dataset info:")
    print(f"  Total episodes: {dataset.num_episodes}")
    print(f"  Total frames: {dataset.num_frames}")
    print(f"  FPS: {dataset.fps}")
    print()

    if args.episode_index >= dataset.num_episodes:
        print(f"Error: Episode {args.episode_index} not found.")
        print(f"Dataset has {dataset.num_episodes} episodes (0-{dataset.num_episodes-1})")
        return 1

    print(f"Visualizing episode {args.episode_index}...")
    visualize_dataset(
        repo_id=args.repo_id,
        root=str(args.root),
        episode_index=args.episode_index,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
