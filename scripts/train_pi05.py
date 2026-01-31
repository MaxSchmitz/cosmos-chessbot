#!/usr/bin/env python3
"""Train π₀.₅ policy on collected chess manipulation data.

This script fine-tunes the π₀.₅ base model on chess piece manipulation
demonstrations collected via LeRobot teleoperation.

Usage:
    # Basic training
    python scripts/train_pi05.py

    # Custom dataset path
    python scripts/train_pi05.py --dataset-path data/episodes/my-dataset

    # Adjust training parameters
    python scripts/train_pi05.py --steps 100000 --batch-size 8
"""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Train π₀.₅ policy on chess manipulation data"
    )
    parser.add_argument(
        "--dataset-repo",
        type=str,
        default="cosmos-chessbot/chess-manipulation",
        help="Dataset repository ID (default: cosmos-chessbot/chess-manipulation)",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/episodes"),
        help="Local dataset root path (default: data/episodes)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/pi05_chess"),
        help="Output directory for checkpoints (default: checkpoints/pi05_chess)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="lerobot/pi05_base",
        help="Base model to fine-tune (default: lerobot/pi05_base)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50000,
        help="Training steps (default: 50000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5000,
        help="Evaluation frequency in steps (default: 5000)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("π₀.₅ Training Configuration")
    print("=" * 60)
    print(f"Dataset: {args.dataset_repo}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Base model: {args.base_model}")
    print(f"Output: {args.output_dir}")
    print(f"Steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)
    print()

    # Check if dataset exists
    if not args.dataset_path.exists():
        print(f"ERROR: Dataset path not found: {args.dataset_path}")
        print("Please collect data first using scripts/collect_episodes.py")
        return 1

    # Build lerobot-train command
    # Note: This assumes lerobot is installed in a separate environment
    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={args.dataset_repo}",
        f"--dataset.root={args.dataset_path}",
        f"--policy.path={args.base_model}",
        f"--output_dir={args.output_dir}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--learning_rate={args.lr}",
        f"--eval_freq={args.eval_freq}",
        "--save_checkpoint",
        "--save_model",
    ]

    print("Running lerobot-train...")
    print("Command:", " ".join(cmd))
    print()

    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("=" * 60)
        print("Training completed successfully!")
        print(f"Checkpoints saved to: {args.output_dir}")
        print("=" * 60)
        return result.returncode

    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print("ERROR: Training failed")
        print(f"Exit code: {e.returncode}")
        print("=" * 60)
        print()
        print("Troubleshooting:")
        print("1. Ensure lerobot is installed: pip install lerobot")
        print("2. Check dataset format and location")
        print("3. Verify GPU is available if using CUDA")
        return e.returncode

    except FileNotFoundError:
        print()
        print("=" * 60)
        print("ERROR: lerobot-train command not found")
        print("=" * 60)
        print()
        print("Setup instructions:")
        print("1. Create separate environment: python3 -m venv lerobot_env")
        print("2. Activate: source lerobot_env/bin/activate")
        print("3. Install: pip install lerobot")
        print("4. Run this script from that environment")
        return 1


if __name__ == "__main__":
    exit(main())
