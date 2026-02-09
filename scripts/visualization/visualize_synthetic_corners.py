#!/usr/bin/env python3
"""
Visualize ground truth board corner annotations on synthetic renders.

Reads annotations.json from a render output directory and draws the 4 board
corners + bounding box on each image, saving annotated copies.

Usage:
    uv run python scripts/visualization/visualize_synthetic_corners.py data/value_hybrid/
    uv run python scripts/visualization/visualize_synthetic_corners.py data/value_hybrid/ -n 5
    uv run python scripts/visualization/visualize_synthetic_corners.py data/value_hybrid/ -o viz_output/
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


CORNER_COLORS = {
    "top_left": (0, 255, 0),      # green
    "top_right": (0, 165, 255),    # orange
    "bottom_right": (0, 0, 255),   # red
    "bottom_left": (255, 0, 0),    # blue
}

CORNER_LABELS = {
    "top_left": "TL",
    "top_right": "TR",
    "bottom_right": "BR",
    "bottom_left": "BL",
}


def draw_corners(image: np.ndarray, board_corners: dict) -> np.ndarray:
    """Draw board corners and bounding box on a BGR image."""
    vis = image.copy()
    h, w = vis.shape[:2]

    scale = max(w, h) / 1000
    thick = max(2, int(3 * scale))
    thin = max(1, int(2 * scale))
    radius = max(8, int(14 * scale))
    font = 0.6 * scale

    corners = board_corners["corners"]
    bbox = board_corners["bbox"]

    # Draw bounding box
    bx = bbox["x_center"] * w
    by = bbox["y_center"] * h
    bw = bbox["width"] * w
    bh = bbox["height"] * h
    x1 = int(bx - bw / 2)
    y1 = int(by - bh / 2)
    x2 = int(bx + bw / 2)
    y2 = int(by + bh / 2)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), thin)

    # Draw corner polygon (board outline)
    order = ["top_left", "top_right", "bottom_right", "bottom_left"]
    pts = []
    for name in order:
        cx, cy = corners[name]
        pts.append([int(cx * w), int(cy * h)])
    pts_arr = np.array(pts, dtype=np.int32)
    cv2.polylines(vis, [pts_arr], isClosed=True, color=(0, 255, 0), thickness=thick)

    # Draw corner keypoints with labels
    for name in order:
        cx, cy = corners[name]
        px, py = int(cx * w), int(cy * h)
        color = CORNER_COLORS[name]
        label = CORNER_LABELS[name]

        cv2.circle(vis, (px, py), radius, color, -1)
        cv2.circle(vis, (px, py), radius + 2, (255, 255, 255), thin)
        cv2.putText(
            vis, label, (px + radius + 4, py - radius // 2),
            cv2.FONT_HERSHEY_SIMPLEX, font, color, thin + 1, cv2.LINE_AA
        )

    return vis


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ground truth board corners on synthetic renders"
    )
    parser.add_argument("input_dir", type=Path, help="Render output directory")
    parser.add_argument("-n", "--num", type=int, default=None,
                        help="Max images to visualize (default: all)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output directory (default: <input_dir>/corner_viz)")
    args = parser.parse_args()

    annotations_path = args.input_dir / "annotations.json"
    if not annotations_path.exists():
        print(f"Error: {annotations_path} not found")
        return

    with open(annotations_path) as f:
        annotations = json.load(f)

    valid = [a for a in annotations if "board_corners" in a and a["board_corners"]]
    if args.num:
        valid = valid[:args.num]

    print(f"Visualizing {len(valid)} images with board corners...")

    output_dir = args.output or args.input_dir / "corner_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    for entry in valid:
        image_path = Path(entry["image"])
        if not image_path.exists():
            print(f"  Warning: {image_path} not found, skipping")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  Warning: could not read {image_path}")
            continue

        vis = draw_corners(image, entry["board_corners"])

        out_path = output_dir / f"{image_path.stem}_corners.jpg"
        cv2.imwrite(str(out_path), vis)

        corners = entry["board_corners"]["corners"]
        print(f"  {image_path.name}: "
              f"TL=({corners['top_left'][0]:.3f},{corners['top_left'][1]:.3f}) "
              f"TR=({corners['top_right'][0]:.3f},{corners['top_right'][1]:.3f}) "
              f"BR=({corners['bottom_right'][0]:.3f},{corners['bottom_right'][1]:.3f}) "
              f"BL=({corners['bottom_left'][0]:.3f},{corners['bottom_left'][1]:.3f})")

    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    main()
