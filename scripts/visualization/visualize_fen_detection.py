#!/usr/bin/env python3
"""
Visualize FEN detection on any input image.

Usage:
    uv run python scripts/visualization/visualize_fen_detection.py path/to/image.jpg
    uv run python scripts/visualization/visualize_fen_detection.py path/to/image.jpg -o output.jpg
    uv run python scripts/visualization/visualize_fen_detection.py path/to/image.jpg --verbose
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from cosmos_chessbot.vision.yolo_dino_detector import YOLODINOFenDetector

PIECE_WEIGHTS = "runs/detect/runs/detect/yolo26_chess/weights/best.pt"
CORNER_WEIGHTS = "runs/pose/runs/pose/board_corners/weights/best.pt"


def draw_annotations(image, corners, pieces, fen):
    """Draw board pose, piece bounding boxes, and FEN onto a BGR image."""
    vis = image.copy()
    h, w = vis.shape[:2]

    # Scale line widths and font sizes relative to image size
    scale = max(w, h) / 1000
    thick = max(2, int(3 * scale))
    thin = max(1, int(2 * scale))
    radius = max(6, int(12 * scale))
    font_big = 0.7 * scale
    font_med = 0.55 * scale
    font_sm = 0.45 * scale

    # -- Board pose: outline + corner keypoints --
    if corners is not None:
        pts = corners.astype(np.int32)
        # Draw board outline with semi-transparent overlay
        overlay = vis.copy()
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=thick + 2)
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
        # Solid outline on top
        for i in range(4):
            cv2.line(vis, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), thick)

        # Corner keypoints
        colors = [(0, 255, 0), (0, 165, 255), (0, 0, 255), (255, 0, 0)]
        labels = ["TL", "TR", "BR", "BL"]
        for pt, col, lbl in zip(pts, colors, labels):
            cv2.circle(vis, tuple(pt), radius, col, -1)
            cv2.circle(vis, tuple(pt), radius + 2, (255, 255, 255), thin)
            cv2.putText(vis, lbl, (pt[0] + radius + 4, pt[1] - radius // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_med, col, thin + 1, cv2.LINE_AA)

    # -- Piece bounding boxes --
    for det in pieces:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        piece = det["piece"]
        conf = det["confidence"]
        color = (0, 220, 255) if piece.isupper() else (255, 100, 100)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thin + 1)
        label = f"{det['class']} {conf:.2f} -> {det['square']}"
        # Text background for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_sm, thin)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_sm, (0, 0, 0), thin, cv2.LINE_AA)

    # -- FEN text at bottom --
    fen_label = f"FEN: {fen}"
    (tw, th), _ = cv2.getTextSize(fen_label, cv2.FONT_HERSHEY_SIMPLEX, font_big, thick)
    cv2.rectangle(vis, (5, h - th - 20), (tw + 15, h - 5), (0, 0, 0), -1)
    cv2.putText(vis, fen_label, (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, font_big, (0, 255, 0), thick, cv2.LINE_AA)

    return vis


def main():
    parser = argparse.ArgumentParser(description="Visualize FEN detection on an image")
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path (default: <input>_fen_viz.jpg)")
    parser.add_argument("--piece-weights", type=str, default=PIECE_WEIGHTS)
    parser.add_argument("--corner-weights", type=str, default=CORNER_WEIGHTS)
    parser.add_argument("--corners", type=Path, default=None,
                        help="Path to calibrated corners JSON (skips pose model)")
    parser.add_argument("--conf", type=float, default=0.10,
                        help="YOLO confidence threshold (default: 0.10)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    use_static_corners = args.corners is not None and args.corners.exists()

    detector = YOLODINOFenDetector(
        yolo_weights=args.piece_weights,
        corner_weights=None if use_static_corners else args.corner_weights,
        mlp_weights=None,
        device=args.device,
        conf_threshold=args.conf,
        use_dino=False,
        static_corners=str(args.corners) if use_static_corners else None,
    )

    frame = cv2.imread(str(args.image))
    if frame is None:
        print(f"Error: could not read image: {args.image}")
        sys.exit(1)

    image_rgb = frame[:, :, ::-1]
    corners = detector._detect_corners(image_rgb)
    result = detector.detect_fen_with_metadata(image_rgb)

    fen = result["fen"]
    pieces = result["pieces"]

    print(f"FEN: {fen}")
    print(f"Pieces: {len(pieces)}")
    if args.verbose:
        for p in sorted(pieces, key=lambda x: x["square"]):
            print(f"  {p['square']:>3s}  {p['class']:<16s} conf={p['confidence']:.3f}")

    vis = draw_annotations(frame, corners, pieces, fen)

    out_path = args.output or args.image.with_stem(args.image.stem + "_fen_viz")
    cv2.imwrite(str(out_path), vis)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
