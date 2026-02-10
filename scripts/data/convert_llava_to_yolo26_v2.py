#!/usr/bin/env python3
"""
Convert Llava Chess Dataset to YOLO26 Format (Version 2)

This version uses pre-computed bounding boxes from the rendering metadata,
making conversion fast and accurate.

Usage:
    python scripts/convert_llava_to_yolo26_v2.py --input data/chess --output data/yolo26_chess
"""

import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List
import random

# FEN piece to YOLO class mapping (12 classes)
PIECE_TYPE_TO_CLASS = {
    # White pieces
    'pawn_w': 0,
    'knight_w': 1,
    'bishop_w': 2,
    'rook_w': 3,
    'queen_w': 4,
    'king_w': 5,
    # Black pieces
    'pawn_b': 6,
    'knight_b': 7,
    'bishop_b': 8,
    'rook_b': 9,
    'queen_b': 10,
    'king_b': 11,
}

CLASS_NAMES = [
    'white_pawn', 'white_knight', 'white_bishop', 'white_rook', 'white_queen', 'white_king',
    'black_pawn', 'black_knight', 'black_bishop', 'black_rook', 'black_queen', 'black_king'
]


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    train_split: float = 0.9,
    seed: int = 42,
):
    """
    Convert Llava dataset to YOLO26 format using pre-computed bounding boxes.

    Args:
        input_dir: Directory containing annotations.json and images
        output_dir: Output directory for YOLO dataset
        train_split: Fraction of data for training (default 0.9)
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    # Load annotations
    annotations_path = input_dir / 'annotations.json'
    print(f"Loading annotations from {annotations_path}")

    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    with open(annotations_path) as f:
        annotations = json.load(f)

    print(f"Found {len(annotations)} annotated images")

    # Check if bounding boxes are available
    has_bboxes = False
    if annotations and 'bounding_boxes' in annotations[0]:
        has_bboxes = True
        print("✓ Bounding boxes found in annotations")
    else:
        print("✗ No bounding boxes in annotations - dataset needs to be re-generated")
        print("  Run the rendering script with the updated version to include bounding boxes")
        return

    # Shuffle and split
    random.shuffle(annotations)
    split_idx = int(len(annotations) * train_split)
    train_data = annotations[:split_idx]
    val_data = annotations[split_idx:]

    print(f"Split: {len(train_data)} train, {len(val_data)} val")

    # Create output directories
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Process datasets
    stats = {'train': {'images': 0, 'pieces': 0}, 'val': {'images': 0, 'pieces': 0}}

    for split_name, split_data in [('train', train_data), ('val', val_data)]:
        print(f"\nProcessing {split_name} set...")

        for i, item in enumerate(split_data):
            # Get image path
            image_path = Path(item['image'])
            if not image_path.is_absolute():
                image_path = input_dir / image_path.name

            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue

            # Copy image
            output_image_path = output_dir / 'images' / split_name / image_path.name
            shutil.copy(image_path, output_image_path)

            # Convert bounding boxes to YOLO format
            bboxes = item.get('bounding_boxes', [])
            labels = []

            for bbox_data in bboxes:
                piece_type = bbox_data['piece_type']
                bbox = bbox_data['bbox']

                # Get YOLO class ID
                if piece_type not in PIECE_TYPE_TO_CLASS:
                    print(f"Warning: Unknown piece type '{piece_type}' in {image_path.name}")
                    continue

                class_id = PIECE_TYPE_TO_CLASS[piece_type]

                # YOLO format: class_id x_center y_center width height (all normalized)
                x_center = bbox['x_center']
                y_center = bbox['y_center']
                width = bbox['width']
                height = bbox['height']

                # Validate bbox (must be within [0, 1])
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                       0 < width <= 1 and 0 < height <= 1):
                    print(f"Warning: Invalid bbox in {image_path.name}: {bbox}")
                    continue

                labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Write labels file
            output_label_path = output_dir / 'labels' / split_name / f"{image_path.stem}.txt"
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(labels))

            # Update stats
            stats[split_name]['images'] += 1
            stats[split_name]['pieces'] += len(labels)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(split_data)} images")

        print(f"Completed {split_name}: {stats[split_name]['images']} images, "
              f"{stats[split_name]['pieces']} pieces")

    # Create dataset YAML
    yaml_content = f"""# YOLO26 Chess Piece Detection Dataset
path: {output_dir.absolute()}
train: images/train
val: images/val

# Classes (12 piece types)
names:
  0: white_pawn
  1: white_knight
  2: white_bishop
  3: white_rook
  4: white_queen
  5: white_king
  6: black_pawn
  7: black_knight
  8: black_bishop
  9: black_rook
  10: black_queen
  11: black_king
"""

    yaml_path = output_dir / 'chess_dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"{'='*60}")
    print(f"Dataset YAML: {yaml_path}")
    print(f"Train: {stats['train']['images']} images, {stats['train']['pieces']} pieces")
    print(f"Val: {stats['val']['images']} images, {stats['val']['pieces']} pieces")
    print(f"Total: {stats['train']['pieces'] + stats['val']['pieces']} piece annotations")
    print(f"\nNext steps:")
    print(f"1. Train YOLO26: python scripts/train_yolo26_pieces.py")
    print(f"2. Validate labels: Check {output_dir / 'labels/train/'}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Llava chess dataset to YOLO26 format (using pre-computed bboxes)"
    )
    parser.add_argument('--input', type=Path, default=Path('data/chess'),
                       help='Input directory with annotations.json and images')
    parser.add_argument('--output', type=Path, default=Path('data/yolo26_chess'),
                       help='Output directory for YOLO dataset')
    parser.add_argument('--train-split', type=float, default=0.9,
                       help='Fraction of data for training (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    convert_dataset(
        input_dir=args.input,
        output_dir=args.output,
        train_split=args.train_split,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
