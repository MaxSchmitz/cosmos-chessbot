#!/usr/bin/env python3
"""
Visualize ground truth board corner annotations on images.

Draws the 4 keypoints and board outline on a sample of images
to verify the corner conversion is correct before training.

Usage:
    uv run python scripts/visualize_corners.py
    uv run python scripts/visualize_corners.py --n 10 --split val
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


# Keypoint order: top_left(0), top_right(1), bottom_right(2), bottom_left(3)
CORNER_COLORS = {
    0: (0, 255, 0),    # top_left - green
    1: (0, 0, 255),    # top_right - red
    2: (255, 0, 0),    # bottom_right - blue
    3: (255, 255, 0),  # bottom_left - yellow
}
CORNER_LABELS = ['TL', 'TR', 'BR', 'BL']

# Draw board outline in this order to trace the perimeter
OUTLINE_ORDER = [0, 1, 2, 3, 0]


def parse_pose_label(label_path, img_w, img_h):
    """Parse a YOLO pose label file into denormalized corner points."""
    with open(label_path) as f:
        parts = f.read().strip().split()

    # Format: class x_center y_center w h px1 py1 px2 py2 px3 py3 px4 py4
    kpts_norm = [float(v) for v in parts[5:]]
    corners = []
    for i in range(4):
        x = int(kpts_norm[i * 2] * img_w)
        y = int(kpts_norm[i * 2 + 1] * img_h)
        corners.append((x, y))
    return corners


def draw_board(img, corners):
    """Draw board outline and corner markers on image."""
    # Board outline
    for i in range(len(OUTLINE_ORDER) - 1):
        cv2.line(img, corners[OUTLINE_ORDER[i]], corners[OUTLINE_ORDER[i + 1]], (255, 255, 255), 2)

    # Corner dots and labels
    for idx, pt in enumerate(corners):
        color = CORNER_COLORS[idx]
        cv2.circle(img, pt, 8, color, -1)
        cv2.circle(img, pt, 10, (255, 255, 255), 2)
        cv2.putText(img, CORNER_LABELS[idx], (pt[0] + 14, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=6, help='Number of images to visualize')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--output', type=Path, default=Path('data/chessred2k_pose/corner_viz'))
    args = parser.parse_args()

    images_dir = Path('data/chessred2k_pose/images') / args.split
    labels_dir = Path('data/chessred2k_pose/labels') / args.split
    args.output.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob('*.jpg'))[:args.n]
    print(f"Visualizing {len(image_files)} images from {args.split}")

    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"  Skipping {img_path.name} - no label")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        corners = parse_pose_label(label_path, w, h)
        img = draw_board(img, corners)

        out_path = args.output / img_path.name
        cv2.imwrite(str(out_path), img)
        print(f"  Saved {out_path.name}  corners={corners}")

    print(f"\nOutput: {args.output}")


if __name__ == '__main__':
    main()
