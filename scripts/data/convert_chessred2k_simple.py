#!/usr/bin/env python3
"""
Simple ChessReD2k to YOLO converter - processes only available images.
"""

import json
import shutil
from pathlib import Path


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """Convert COCO bbox to YOLO format (normalized)."""
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return [x_center, y_center, w_norm, h_norm]


def main():
    # Load annotations
    print("Loading annotations...")
    with open('data/chessred2k/annotations.json') as f:
        data = json.load(f)

    # Get available game directories
    images_path = Path('data/chessred2k/images')
    existing_game_ids = set(
        int(d.name) for d in images_path.iterdir()
        if d.is_dir() and d.name.isdigit()
    )
    print(f"Found {len(existing_game_ids)} game directories")

    # Filter available images
    available_images = [img for img in data['images'] if img['game_id'] in existing_game_ids]
    print(f"Available images: {len(available_images)}")

    # Create splits (80/10/10)
    train_size = int(len(available_images) * 0.80)
    val_size = int(len(available_images) * 0.10)

    splits = {
        'train': available_images[:train_size],
        'val': available_images[train_size:train_size + val_size],
        'test': available_images[train_size + val_size:]
    }

    print(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Build annotation index
    print("Indexing annotations...")
    image_to_pieces = {}
    for piece_ann in data['annotations']['pieces']:
        img_id = piece_ann['image_id']
        if img_id not in image_to_pieces:
            image_to_pieces[img_id] = []
        image_to_pieces[img_id].append(piece_ann)

    # Create output directory
    output_path = Path('data/chessred2k_yolo')
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split_name, images in splits.items():
        print(f"\nProcessing {split_name} ({len(images)} images)...")

        images_out = output_path / "images" / split_name
        labels_out = output_path / "labels" / split_name
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for i, img_info in enumerate(images):
            if i % 100 == 0:
                print(f"  {i}/{len(images)}")

            img_id = img_info['id']
            img_width = img_info['width']
            img_height = img_info['height']

            # Copy image
            src_img = Path('data/chessred2k') / img_info['path']
            dst_img = images_out / img_info['file_name']

            if not src_img.exists():
                continue

            shutil.copy2(src_img, dst_img)

            # Convert annotations
            label_lines = []
            if img_id in image_to_pieces:
                for piece_ann in image_to_pieces[img_id]:
                    cat_id = piece_ann['category_id']
                    if cat_id == 12:  # Skip empty
                        continue

                    bbox = piece_ann['bbox']
                    yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)
                    line = f"{cat_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
                    label_lines.append(line)

            # Write labels
            label_path = labels_out / f"{Path(img_info['file_name']).stem}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(label_lines))

        print(f"  Done: {len(images)} images")

    # Create dataset.yaml
    yaml_content = f"""path: {output_path.absolute()}
train: images/train
val: images/val
test: images/test

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

    print(f"\nDataset ready: {yaml_path}")


if __name__ == "__main__":
    main()
