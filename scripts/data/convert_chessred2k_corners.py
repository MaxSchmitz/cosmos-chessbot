#!/usr/bin/env python3
"""
Convert ChessReD2k corner annotations to YOLO pose format for board corner detection.

Output label format (one line per image, class 0 = chessboard):
    <class> <x_center> <y_center> <w> <h> <px1> <py1> <px2> <py2> <px3> <py3> <px4> <py4>

Keypoint order (clockwise from top-left):
    0: top_left
    1: top_right
    2: bottom_right
    3: bottom_left

Usage:
    uv run python scripts/convert_chessred2k_corners.py
"""

import json
import os
from pathlib import Path


def main():
    print("Loading annotations...")
    with open('data/chessred2k/annotations.json') as f:
        data = json.load(f)

    # Index corners by image_id
    corners_by_image = {}
    for corner_ann in data['annotations']['corners']:
        corners_by_image[corner_ann['image_id']] = corner_ann['corners']

    # Index images by id
    images_by_id = {img['id']: img for img in data['images']}

    print(f"Total corner annotations: {len(corners_by_image)}")

    # Output directory - symlink images, create pose labels
    output_path = Path('data/chessred2k_pose')
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy images from the existing resized dataset
    # Note: symlinks won't work here -- YOLO resolves them and looks for
    # labels relative to the resolved path, not the symlink location.
    images_dst = output_path / 'images'
    if not images_dst.exists():
        import shutil
        src = Path('data/chessred2k_yolo/images')
        shutil.copytree(src, images_dst)
        print(f"Copied images from {src}")

    # Determine which images are in each split by scanning existing image dirs
    splits = {}
    for split in ('train', 'val', 'test'):
        split_dir = Path('data/chessred2k_yolo/images') / split
        splits[split] = sorted(f.stem for f in split_dir.glob('*.jpg'))
    print(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Build filename -> image_id mapping
    filename_to_imgid = {}
    for img in data['images']:
        filename_to_imgid[Path(img['file_name']).stem] = img['id']

    # Process each split
    total_written = 0
    total_missing = 0

    for split_name, filenames in splits.items():
        labels_out = output_path / 'labels' / split_name
        labels_out.mkdir(parents=True, exist_ok=True)

        for fname in filenames:
            img_id = filename_to_imgid.get(fname)
            if img_id is None:
                total_missing += 1
                continue

            if img_id not in corners_by_image:
                total_missing += 1
                continue

            img_info = images_by_id[img_id]
            img_width = img_info['width']
            img_height = img_info['height']
            corners = corners_by_image[img_id]

            # Extract corners in clockwise order: TL, TR, BR, BL
            tl = corners['top_left']
            tr = corners['top_right']
            br = corners['bottom_right']
            bl = corners['bottom_left']

            # Normalize all coordinates
            kpts = [
                tl[0] / img_width, tl[1] / img_height,   # top_left
                tr[0] / img_width, tr[1] / img_height,   # top_right
                br[0] / img_width, br[1] / img_height,   # bottom_right
                bl[0] / img_width, bl[1] / img_height,   # bottom_left
            ]

            # Bounding box around all corners
            xs = [tl[0], tr[0], br[0], bl[0]]
            ys = [tl[1], tr[1], br[1], bl[1]]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            x_center = ((x_min + x_max) / 2) / img_width
            y_center = ((y_min + y_max) / 2) / img_height
            w = (x_max - x_min) / img_width
            h = (y_max - y_min) / img_height

            # YOLO pose format: class x_center y_center w h px1 py1 px2 py2 ...
            line = f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} " + \
                   " ".join(f"{v:.6f}" for v in kpts)

            label_path = labels_out / f"{fname}.txt"
            with open(label_path, 'w') as f:
                f.write(line + '\n')

            total_written += 1

    print(f"\nLabels written: {total_written}")
    print(f"Missing corners: {total_missing}")

    # Create dataset YAML
    yaml_content = f"""path: {output_path.resolve()}
train: images/train
val: images/val
test: images/test

# Keypoint config: 4 corners, 2D (x, y)
kpt_shape: [4, 2]

# Horizontal flip index mapping:
#   top_left(0) <-> top_right(1)
#   bottom_right(2) <-> bottom_left(3)
flip_idx: [1, 0, 3, 2]

names:
  0: chessboard
"""

    yaml_path = output_path / 'board_corners.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Dataset YAML: {yaml_path}")


if __name__ == '__main__':
    main()
