#!/usr/bin/env python3
"""
Interactive board corner calibration.

Captures a frame from the camera, displays it, and records 4 mouse clicks
as the board corners.  Saves them to a JSON file that YOLODINOFenDetector
can load via the static_corners parameter -- no pose model needed.

Corner click order (must match keypoint convention):
    1st click: top_left     (a8 corner -- rank 8, file a)
    2nd click: top_right    (h8 corner -- rank 8, file h)
    3rd click: bottom_right (h1 corner -- rank 1, file h)
    4th click: bottom_left  (a1 corner -- rank 1, file a)

"Top" and "bottom" are from the chess-board perspective:
    top    = rank 8 (the side closer to the black pieces at start)
    bottom = rank 1 (the side closer to the white pieces at start)

Usage:
    uv run python scripts/calibrate_corners.py
    uv run python scripts/calibrate_corners.py --camera 0 --output data/calibration/corners.json
"""

import argparse
import json
import sys
import cv2
import numpy as np
from pathlib import Path


CORNER_LABELS = ['top_left (a8)', 'top_right (h8)', 'bottom_right (h1)', 'bottom_left (a1)']
CORNER_COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]  # BGR for OpenCV
OUTLINE_ORDER = [0, 1, 2, 3, 0]

# Global state for mouse callback
_clicks: list = []
_current_frame: np.ndarray = None


def mouse_callback(event, x, y, flags, param):
    global _clicks, _current_frame
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if len(_clicks) >= 4:
        return
    _clicks.append((x, y))
    idx = len(_clicks) - 1
    print(f"  Click {idx + 1}/4: {CORNER_LABELS[idx]} at ({x}, {y})")

    # Redraw
    _redraw()


def _redraw():
    vis = _current_frame.copy()
    h, w = vis.shape[:2]

    # Instructions
    if len(_clicks) < 4:
        next_label = CORNER_LABELS[len(_clicks)]
        cv2.putText(vis, f"Click {len(_clicks)+1}/4: {next_label}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw placed corners
    for i, (x, y) in enumerate(_clicks):
        cv2.circle(vis, (x, y), 8, CORNER_COLORS[i], -1)
        cv2.circle(vis, (x, y), 10, (255, 255, 255), 2)
        cv2.putText(vis, CORNER_LABELS[i].split(' ')[0], (x + 12, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CORNER_COLORS[i], 2, cv2.LINE_AA)

    # Draw outline once all 4 corners are placed
    if len(_clicks) == 4:
        pts = np.array(_clicks, dtype=np.int32)
        cv2.polylines(vis, [pts], True, (255, 255, 255), 2)
        cv2.putText(vis, "All corners set. Press ENTER to save, 'r' to redo.",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Corner Calibration', vis)


def main():
    global _clicks, _current_frame

    parser = argparse.ArgumentParser(description="Interactive board corner calibration")
    parser.add_argument('--camera', type=int, default=1, help='Camera index (default: 1)')
    parser.add_argument('--output', type=Path, default=Path('data/calibration/corners.json'),
                        help='Output JSON path')
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Open camera and capture a single frame
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution: {w}x{h}")
    print(f"Captured frame. Now click the 4 corners in order:")
    for i, lbl in enumerate(CORNER_LABELS):
        print(f"  {i + 1}. {lbl}")
    print("Press 'r' to reset clicks, ENTER to save, 'q' to quit.\n")

    # Capture frame (grab a few to let the camera warm up)
    for _ in range(5):
        cap.grab()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Failed to capture frame")
        sys.exit(1)

    _current_frame = frame
    _clicks = []

    cv2.namedWindow('Corner Calibration')
    cv2.setMouseCallback('Corner Calibration', mouse_callback)
    _redraw()

    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            print("Quit.")
            break
        if key == ord('r'):
            print("  Reset.")
            _clicks = []
            _redraw()
            continue
        if key == 13 or key == 10:  # ENTER
            if len(_clicks) < 4:
                print("  Need 4 corners first.")
                continue
            # Save
            corners_data = {
                'corners': [list(pt) for pt in _clicks],
                'resolution': [w, h],
                'labels': CORNER_LABELS,
            }
            with open(args.output, 'w') as f:
                json.dump(corners_data, f, indent=2)
            print(f"\nSaved corners to {args.output}")
            print(f"  Corners: {_clicks}")

            # Also save the annotated frame
            img_path = args.output.parent / 'calibration_frame.jpg'
            vis = frame.copy()
            pts = np.array(_clicks, dtype=np.int32)
            cv2.polylines(vis, [pts], True, (255, 255, 255), 2)
            for i, (x, y) in enumerate(_clicks):
                cv2.circle(vis, (x, y), 8, CORNER_COLORS[i], -1)
                cv2.circle(vis, (x, y), 10, (255, 255, 255), 2)
            cv2.imwrite(str(img_path), vis)
            print(f"  Calibration frame: {img_path}")
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
