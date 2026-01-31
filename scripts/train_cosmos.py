#!/usr/bin/env python3
"""Train Cosmos Policy on collected chess manipulation data.

This script trains Cosmos Policy using NVIDIA's Cosmos world model
for robot manipulation tasks. Cosmos Policy combines vision, action
prediction, and future state forecasting.

Usage:
    # Basic training
    python scripts/train_cosmos.py

    # Custom dataset path
    python scripts/train_cosmos.py --dataset-path data/episodes/my-dataset

    # Adjust training parameters
    python scripts/train_cosmos.py --steps 30000 --batch-size 4
"""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Train Cosmos Policy on chess manipulation data"
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
        default=Path("checkpoints/cosmos_chess"),
        help="Output directory for checkpoints (default: checkpoints/cosmos_chess)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="nvidia/cosmos-predict2-2b",
        help="Base Cosmos model to fine-tune (default: nvidia/cosmos-predict2-2b)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30000,
        help="Training steps (default: 30000)",
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
        default=3000,
        help="Evaluation frequency in steps (default: 3000)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Action horizon for planning (default: 10)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Cosmos Policy Training Configuration")
    print("=" * 60)
    print(f"Dataset: {args.dataset_repo}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Base model: {args.base_model}")
    print(f"Output: {args.output_dir}")
    print(f"Steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Horizon: {args.horizon}")
    print("=" * 60)
    print()

    # Check if dataset exists
    if not args.dataset_path.exists():
        print(f"ERROR: Dataset path not found: {args.dataset_path}")
        print("Please collect data first using scripts/collect_episodes.py")
        return 1

    # Build cosmos-policy training command
    # Note: This uses NVIDIA's Cosmos Policy training script
    # The exact command may vary based on Cosmos package version
    cmd = [
        "python", "-m", "cosmos_policy.train",
        f"--dataset={args.dataset_repo}",
        f"--data_root={args.dataset_path}",
        f"--base_model={args.base_model}",
        f"--output_dir={args.output_dir}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--learning_rate={args.lr}",
        f"--eval_freq={args.eval_freq}",
        f"--horizon={args.horizon}",
        "--save_checkpoints",
    ]

    print("Running Cosmos Policy training...")
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
        print()
        print("Next steps:")
        print("1. Test policy: cosmos-chessbot --policy cosmos --policy-checkpoint", args.output_dir / "final.pt")
        print("2. Run comparison: python scripts/compare_policies.py")
        return result.returncode

    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print("ERROR: Training failed")
        print(f"Exit code: {e.returncode}")
        print("=" * 60)
        print()
        print("Troubleshooting:")
        print("1. Ensure Cosmos Policy is installed")
        print("2. Check dataset format (should be LeRobot format)")
        print("3. Verify GPU is available")
        print("4. Check NVIDIA Cosmos documentation for latest training API")
        return e.returncode

    except FileNotFoundError:
        print()
        print("=" * 60)
        print("ERROR: Cosmos Policy package not found")
        print("=" * 60)
        print()
        print("Setup instructions:")
        print("1. Install Cosmos Policy from NVIDIA:")
        print("   git clone https://github.com/NVIDIA/Cosmos.git")
        print("   cd Cosmos")
        print("   pip install -e .")
        print()
        print("2. Or install from PyPI when available:")
        print("   pip install cosmos-policy")
        print()
        print("3. See plan Phase 4 for detailed setup")
        return 1


if __name__ == "__main__":
    exit(main())
