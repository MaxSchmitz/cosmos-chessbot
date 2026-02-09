#!/usr/bin/env python3
"""
Convert synthetic render annotations to YOLO pose format for board corner detection.

Reads annotations.json from a synthetic render output directory and creates
YOLO pose format labels compatible with the ChessReD2k pose dataset.

Output label format (one line per image, class 0 = chessboard):
    <class> <x_center> <y_center> <w> <h> <px1> <py1> <px2> <py2> <px3> <py3> <px4> <py4>

Keypoint order (clockwise from top-left, matching ChessReD2k convention):
    0: top_left
    1: top_right
    2: bottom_right
    3: bottom_left

Usage:
    uv run python scripts/data/convert_synthetic_corners.py data/synthetic_renders/
    uv run python scripts/data/convert_synthetic_corners.py data/synthetic_renders/ -o data/synthetic_pose
    uv run python scripts/data/convert_synthetic_corners.py data/synthetic_renders/ --val-split 0.1
"""

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert synthetic render annotations to YOLO pose format"
    )
    parser.add_argument(
        "input_dir", type=Path,
        help="Directory containing annotations.json and rendered images"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output directory (default: data/synthetic_pose)"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1,
        help="Fraction of images for validation (default: 0.1)"
    )
    args = parser.parse_args()

    annotations_path = args.input_dir / "annotations.json"
    if not annotations_path.exists():
        print(f"Error: {annotations_path} not found")
        return

    output_dir = args.output or Path("data/synthetic_pose")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {annotations_path}...")
    with open(annotations_path) as f:
        annotations = json.load(f)

    # Filter to entries that have board_corners
    valid = [a for a in annotations if "board_corners" in a and a["board_corners"]]
    print(f"Total annotations: {len(annotations)}, with board corners: {len(valid)}")

    if not valid:
        print("No annotations with board_corners found. "
              "Make sure you rendered with a version that exports corners.")
        return

    # Split into train/val
    n_val = max(1, int(len(valid) * args.val_split))
    n_train = len(valid) - n_val
    train_set = valid[:n_train]
    val_set = valid[n_train:]
    print(f"Split: {len(train_set)} train, {len(val_set)} val")

    # Create output directories
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    total = 0
    for split_name, split_data in [("train", train_set), ("val", val_set)]:
        for entry in split_data:
            image_path = Path(entry["image"])
            if not image_path.exists():
                print(f"  Warning: image not found: {image_path}")
                continue

            corners = entry["board_corners"]["corners"]
            bbox = entry["board_corners"]["bbox"]

            tl = corners["top_left"]
            tr = corners["top_right"]
            br = corners["bottom_right"]
            bl = corners["bottom_left"]

            # YOLO pose format: class x_center y_center w h kp1_x kp1_y ...
            kpts = [
                tl[0], tl[1],
                tr[0], tr[1],
                br[0], br[1],
                bl[0], bl[1],
            ]

            line = (
                f"0 {bbox['x_center']:.6f} {bbox['y_center']:.6f} "
                f"{bbox['width']:.6f} {bbox['height']:.6f} "
                + " ".join(f"{v:.6f}" for v in kpts)
            )

            # Copy image and write label
            stem = image_path.stem
            dst_image = output_dir / "images" / split_name / image_path.name
            dst_label = output_dir / "labels" / split_name / f"{stem}.txt"

            if not dst_image.exists():
                shutil.copy2(image_path, dst_image)

            with open(dst_label, "w") as f:
                f.write(line + "\n")

            total += 1

    print(f"Labels written: {total}")

    # Create dataset YAML
    yaml_content = f"""path: {output_dir.resolve()}
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

    yaml_path = output_dir / "board_corners.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Dataset YAML: {yaml_path}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
