#!/usr/bin/env python3
"""
Combine ChessReD2k and synthetic corner datasets into a single YOLO pose dataset.

Creates symlinks from both source datasets into a combined directory,
following the same pattern as data/combined_pieces/.

Prerequisites:
    uv run python scripts/data/convert_chessred2k_corners.py
    uv run python scripts/data/convert_synthetic_corners.py data/value_hybrid/ -o data/synthetic_pose

Usage:
    uv run python scripts/data/combine_corner_datasets.py
    uv run python scripts/data/combine_corner_datasets.py -o data/combined_corners
"""

import argparse
import os
from pathlib import Path


def symlink_dataset(src_dir: Path, dst_images: Path, dst_labels: Path,
                    prefix: str = "", splits=("train", "val")):
    """Symlink images and labels from src to dst with optional filename prefix."""
    count = 0
    for split in splits:
        src_img_dir = src_dir / "images" / split
        src_lbl_dir = src_dir / "labels" / split

        if not src_img_dir.exists():
            print(f"  Skipping {src_img_dir} (not found)")
            continue

        dst_img_split = dst_images / split
        dst_lbl_split = dst_labels / split
        dst_img_split.mkdir(parents=True, exist_ok=True)
        dst_lbl_split.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(src_img_dir.glob("*.jpg")):
            stem = img_path.stem
            name = f"{prefix}{stem}" if prefix else stem

            # Symlink image
            dst_img = dst_img_split / f"{name}.jpg"
            if not dst_img.exists():
                os.symlink(img_path.resolve(), dst_img)

            # Symlink label
            src_lbl = src_lbl_dir / f"{stem}.txt"
            if src_lbl.exists():
                dst_lbl = dst_lbl_split / f"{name}.txt"
                if not dst_lbl.exists():
                    os.symlink(src_lbl.resolve(), dst_lbl)
                count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Combine ChessReD2k and synthetic corner datasets"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("data/combined_corners"),
        help="Output directory (default: data/combined_corners)",
    )
    parser.add_argument(
        "--chessred", type=Path, default=Path("data/chessred2k_pose"),
        help="ChessReD2k pose dataset directory",
    )
    parser.add_argument(
        "--synthetic", type=Path, default=Path("data/synthetic_pose"),
        help="Synthetic pose dataset directory",
    )
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    dst_images = output / "images"
    dst_labels = output / "labels"

    print("Combining corner datasets")
    print(f"  ChessReD2k: {args.chessred}")
    print(f"  Synthetic:  {args.synthetic}")
    print(f"  Output:     {output}")
    print()

    # ChessReD2k: keep original filenames, include train/val (skip test)
    n_chessred = 0
    if args.chessred.exists():
        print("Linking ChessReD2k corners...")
        n_chessred = symlink_dataset(
            args.chessred, dst_images, dst_labels,
            prefix="", splits=("train", "val"),
        )
        print(f"  {n_chessred} images linked")
    else:
        print(f"  WARNING: {args.chessred} not found -- run convert_chessred2k_corners.py first")

    # Synthetic: prefix with "synth_" to avoid name collisions
    n_synthetic = 0
    if args.synthetic.exists():
        print("Linking synthetic corners...")
        n_synthetic = symlink_dataset(
            args.synthetic, dst_images, dst_labels,
            prefix="synth_", splits=("train", "val"),
        )
        print(f"  {n_synthetic} images linked")
    else:
        print(f"  WARNING: {args.synthetic} not found -- run convert_synthetic_corners.py first")

    total = n_chessred + n_synthetic
    print(f"\nTotal: {total} images")

    # Count per split
    for split in ("train", "val"):
        d = dst_images / split
        if d.exists():
            n = len(list(d.glob("*.jpg")))
            print(f"  {split}: {n}")

    # Write YAML
    yaml_content = f"""path: {output.resolve()}
train: images/train
val: images/val

# Keypoint config: 4 corners, 2D (x, y)
kpt_shape: [4, 2]

# Horizontal flip index mapping:
#   top_left(0) <-> top_right(1)
#   bottom_right(2) <-> bottom_left(3)
flip_idx: [1, 0, 3, 2]

names:
  0: chessboard
"""

    yaml_path = output / "board_corners.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\nDataset YAML: {yaml_path}")
    print("\nTo train:")
    print(f"  uv run python scripts/training/train_yolo26_corners.py "
          f"--data {yaml_path} --device cuda --epochs 60 --batch 16")


if __name__ == "__main__":
    main()
