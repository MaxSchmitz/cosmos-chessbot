#!/usr/bin/env python3
"""
Convert ChessReD2k dataset to YOLO format for training YOLO26.

ChessReD2k format:
- COCO-style annotations
- Bounding boxes: [x, y, width, height] in pixels
- 13 categories (12 pieces + empty)

YOLO format:
- Text files with one line per object
- Format: <class_id> <x_center> <y_center> <width> <height>
- All values normalized to [0, 1]
- 12 classes (pieces only, no empty)
"""

import json
import shutil
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, desc=None):
        return iterable


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert COCO bbox [x, y, w, h] to YOLO format [x_center, y_center, w, h] (normalized).

    Args:
        bbox: [x, y, width, height] in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        [x_center, y_center, width, height] normalized to [0, 1]
    """
    x, y, w, h = bbox

    # Calculate center
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height

    # Normalize width and height
    w_norm = w / img_width
    h_norm = h / img_height

    return [x_center, y_center, w_norm, h_norm]


def convert_chessred2k_to_yolo(
    annotations_path: str,
    images_dir: str,
    output_dir: str,
    use_split: str = "train"
):
    """
    Convert ChessReD2k dataset to YOLO format.

    Args:
        annotations_path: Path to annotations.json
        images_dir: Path to images directory
        output_dir: Output directory for YOLO format dataset
        use_split: Which split to use ("train", "val", "test", or "all")
    """
    print(f"Loading annotations from {annotations_path}")
    with open(annotations_path) as f:
        data = json.load(f)

    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine which images to process
    if use_split == "all":
        splits_to_process = [("train", data['splits']['train']['image_ids']),
                              ("val", data['splits']['val']['image_ids']),
                              ("test", data['splits']['test']['image_ids'])]
    else:
        splits_to_process = [(use_split, data['splits'][use_split]['image_ids'])]

    # Check which game directories exist
    images_path = Path(images_dir) / "images"
    existing_game_ids = set(
        int(d.name) for d in images_path.iterdir()
        if d.is_dir() and d.name.isdigit()
    )
    print(f"Found {len(existing_game_ids)} game directories: {sorted(existing_game_ids)}")

    # Create image_id to image info mapping (only for available games)
    image_map = {}
    available_count = 0
    for img in data['images']:
        if img['game_id'] in existing_game_ids:
            image_map[img['id']] = img
            available_count += 1
    print(f"Available images: {available_count} / {len(data['images'])}")

    # Create image_id to annotations mapping
    print("Building annotation index...")
    image_to_pieces = {}
    for piece_ann in data['annotations']['pieces']:
        img_id = piece_ann['image_id']
        if img_id not in image_to_pieces:
            image_to_pieces[img_id] = []
        image_to_pieces[img_id].append(piece_ann)

    # Process each split
    for split_name, image_ids in splits_to_process:
        print(f"\nProcessing {split_name} split ({len(image_ids)} images)...")

        # Create split directories
        images_out = output_path / "images" / split_name
        labels_out = output_path / "labels" / split_name
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        skipped = 0
        processed = 0

        for img_id in tqdm(image_ids, desc=f"Converting {split_name}"):
            if img_id not in image_map:
                print(f"Warning: Image ID {img_id} not found in images list")
                skipped += 1
                continue

            img_info = image_map[img_id]
            img_width = img_info['width']
            img_height = img_info['height']

            # Source and destination paths
            src_img_path = Path(images_dir) / img_info['path']
            dst_img_path = images_out / img_info['file_name']

            if not src_img_path.exists():
                print(f"Warning: Image not found: {src_img_path}")
                skipped += 1
                continue

            # Copy image
            shutil.copy2(src_img_path, dst_img_path)

            # Convert annotations
            label_lines = []
            if img_id in image_to_pieces:
                for piece_ann in image_to_pieces[img_id]:
                    cat_id = piece_ann['category_id']

                    # Skip "empty" category (id 12)
                    if cat_id == 12:
                        continue

                    # Convert bbox to YOLO format
                    bbox = piece_ann['bbox']
                    yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)

                    # YOLO format: <class_id> <x_center> <y_center> <width> <height>
                    line = f"{cat_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
                    label_lines.append(line)

            # Write label file
            label_path = labels_out / f"{Path(img_info['file_name']).stem}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(label_lines))

            processed += 1

        print(f"  Processed: {processed} images")
        if skipped > 0:
            print(f"  Skipped: {skipped} images")

    # Create dataset.yaml
    yaml_content = f"""# ChessReD2k dataset in YOLO format
path: {output_path.absolute()}
train: images/train
val: images/val
test: images/test

# Classes (12 piece types)
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

    yaml_path = output_path / "chess_dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\nDataset configuration saved to: {yaml_path}")
    print(f"Dataset ready for YOLO26 training!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert ChessReD2k to YOLO format")
    parser.add_argument(
        "--annotations",
        default="data/chessred2k/annotations.json",
        help="Path to annotations.json"
    )
    parser.add_argument(
        "--images-dir",
        default="data/chessred2k",
        help="Path to ChessReD2k images directory"
    )
    parser.add_argument(
        "--output-dir",
        default="data/chessred2k_yolo",
        help="Output directory for YOLO format"
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["train", "val", "test", "all"],
        help="Which split to convert (default: all)"
    )

    args = parser.parse_args()

    convert_chessred2k_to_yolo(
        annotations_path=args.annotations,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        use_split=args.split
    )
