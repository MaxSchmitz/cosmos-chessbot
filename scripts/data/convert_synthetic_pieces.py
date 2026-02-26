#!/usr/bin/env python3
"""
Convert synthetic render annotations to YOLO detection format for piece detection.

Reads annotations.json from a synthetic render output directory and creates
YOLO detection format labels matching the ChessReD2k class mapping (12 classes).

Output label format (one line per piece):
    <class_id> <x_center> <y_center> <width> <height>

Class mapping (matches chessred2k_yolo/chess_dataset.yaml):
    0: white-pawn     6: black-pawn
    1: white-rook     7: black-rook
    2: white-knight   8: black-knight
    3: white-bishop   9: black-bishop
    4: white-queen   10: black-queen
    5: white-king    11: black-king

Usage:
    uv run python scripts/data/convert_synthetic_pieces.py data/value_hybrid/
    uv run python scripts/data/convert_synthetic_pieces.py data/value_hybrid/ -o data/synthetic_pieces
    uv run python scripts/data/convert_synthetic_pieces.py data/value_hybrid/ --val-split 0.1
"""

import argparse
import json
import shutil
from pathlib import Path


# Map annotation piece_type (e.g. "pawn_w") to YOLO class ID
PIECE_TYPE_TO_CLASS = {
    "pawn_w": 0,    "rook_w": 1,    "knight_w": 2,
    "bishop_w": 3,  "queen_w": 4,   "king_w": 5,
    "pawn_b": 6,    "rook_b": 7,    "knight_b": 8,
    "bishop_b": 9,  "queen_b": 10,  "king_b": 11,
}

CLASS_NAMES = {
    0: "white-pawn",   1: "white-rook",   2: "white-knight",
    3: "white-bishop", 4: "white-queen",  5: "white-king",
    6: "black-pawn",   7: "black-rook",   8: "black-knight",
    9: "black-bishop", 10: "black-queen", 11: "black-king",
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert synthetic render annotations to YOLO detection format"
    )
    parser.add_argument(
        "input_dir", type=Path,
        help="Directory containing annotations.json and rendered images"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output directory (default: data/synthetic_pieces)"
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

    output_dir = args.output or Path("data/synthetic_pieces")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {annotations_path}...")
    with open(annotations_path) as f:
        annotations = json.load(f)

    # Filter to entries that have bounding boxes
    valid = [a for a in annotations if a.get("bounding_boxes")]
    print(f"Total annotations: {len(annotations)}, with bounding boxes: {len(valid)}")

    if not valid:
        print("No annotations with bounding_boxes found.")
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

    total_images = 0
    total_boxes = 0
    skipped_types = set()

    for split_name, split_data in [("train", train_set), ("val", val_set)]:
        for entry in split_data:
            image_path = Path(entry["image"])
            if not image_path.exists():
                print(f"  Warning: image not found: {image_path}")
                continue

            lines = []
            for bb in entry["bounding_boxes"]:
                piece_type = bb["piece_type"]
                class_id = PIECE_TYPE_TO_CLASS.get(piece_type)
                if class_id is None:
                    skipped_types.add(piece_type)
                    continue

                bbox = bb["bbox"]
                lines.append(
                    f"{class_id} {bbox['x_center']:.6f} {bbox['y_center']:.6f} "
                    f"{bbox['width']:.6f} {bbox['height']:.6f}"
                )

            if not lines:
                continue

            stem = image_path.stem
            dst_image = output_dir / "images" / split_name / image_path.name
            dst_label = output_dir / "labels" / split_name / f"{stem}.txt"

            if not dst_image.exists():
                shutil.copy2(image_path, dst_image)

            with open(dst_label, "w") as f:
                f.write("\n".join(lines) + "\n")

            total_images += 1
            total_boxes += len(lines)

    if skipped_types:
        print(f"Warning: skipped unknown piece types: {skipped_types}")

    print(f"Images: {total_images}, total bounding boxes: {total_boxes}")
    print(f"Average pieces per image: {total_boxes / total_images:.1f}")

    # Create dataset YAML
    yaml_path = output_dir / "chess_dataset.yaml"
    yaml_content = f"""path: {output_dir.resolve()}
train: images/train
val: images/val

names:
  0: white-pawn
  1: white-rook
  2: white-knight
  3: white-bishop
  4: white-queen
  5: white-king
  6: black-pawn
  7: black-rook
  8: black-knight
  9: black-bishop
  10: black-queen
  11: black-king
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Dataset YAML: {yaml_path}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
