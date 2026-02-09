#!/usr/bin/env python3
"""
Visualize FEN detection on any input image.

Always shows raw YOLO bounding boxes (piece detections) and board pose
(corner keypoints), even when one or both models find nothing.

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

from cosmos_chessbot.vision.yolo_dino_detector import (
    YOLODINOFenDetector, CLASS_NAMES, CLASS_TO_PIECE,
)

PIECE_WEIGHTS = "runs/detect/runs/detect/yolo26_chess/weights/best.pt"
CORNER_WEIGHTS = "runs/pose/runs/pose/board_corners/weights/best.pt"


def draw_annotations(image, corners, raw_detections, mapped_pieces, fen):
    """Draw board pose, all raw YOLO bounding boxes, and FEN onto a BGR image.

    Args:
        image: BGR numpy array
        corners: (4, 2) corner array or None
        raw_detections: list of dicts with bbox/class/confidence from YOLO
            (always shown, even without corners)
        mapped_pieces: list of dicts with bbox/class/confidence/square
            (only pieces that mapped to a square via homography)
        fen: detected FEN string
    """
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
        overlay = vis.copy()
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=thick + 2)
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
        for i in range(4):
            cv2.line(vis, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), thick)

        colors = [(0, 255, 0), (0, 165, 255), (0, 0, 255), (255, 0, 0)]
        labels = ["TL", "TR", "BR", "BL"]
        for pt, col, lbl in zip(pts, colors, labels):
            cv2.circle(vis, tuple(pt), radius, col, -1)
            cv2.circle(vis, tuple(pt), radius + 2, (255, 255, 255), thin)
            cv2.putText(vis, lbl, (pt[0] + radius + 4, pt[1] - radius // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_med, col, thin + 1, cv2.LINE_AA)

    # Set of mapped bboxes so we can distinguish mapped vs unmapped
    mapped_bboxes = set()
    for det in mapped_pieces:
        mapped_bboxes.add(tuple(int(v) for v in det["bbox"]))

    # -- All raw YOLO bounding boxes --
    for det in raw_detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        bbox_key = (x1, y1, x2, y2)
        piece_char = det["piece"]
        conf = det["confidence"]
        is_mapped = bbox_key in mapped_bboxes

        if is_mapped:
            # Find the square name from mapped_pieces
            square = "?"
            for mp in mapped_pieces:
                if tuple(int(v) for v in mp["bbox"]) == bbox_key:
                    square = mp["square"]
                    break
            color = (0, 220, 255) if piece_char.isupper() else (255, 100, 100)
            label = f"{det['class']} {conf:.2f} -> {square}"
        else:
            # Unmapped: detected by YOLO but not placed on board
            color = (128, 128, 128)
            label = f"{det['class']} {conf:.2f} (unmapped)"

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thin + 1)
        (tw, th_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_sm, thin)
        cv2.rectangle(vis, (x1, y1 - th_text - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_sm, (0, 0, 0), thin, cv2.LINE_AA)

    # -- Status + FEN text at bottom --
    lines = []
    if corners is None:
        lines.append("Board pose: NOT DETECTED")
    else:
        lines.append("Board pose: OK")
    lines.append(f"Detections: {len(raw_detections)} raw, {len(mapped_pieces)} mapped")
    lines.append(f"FEN: {fen}")

    y_pos = h - 14
    for line in reversed(lines):
        (tw, th_text), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_big, thick)
        cv2.rectangle(vis, (5, y_pos - th_text - 8), (tw + 15, y_pos + 5), (0, 0, 0), -1)
        text_color = (0, 255, 0) if "NOT" not in line else (0, 0, 255)
        cv2.putText(vis, line, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, font_big, text_color, thick, cv2.LINE_AA)
        y_pos -= th_text + 14

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

    # Step 1: Run corner detection independently
    corners = detector._detect_corners(image_rgb)

    # Step 2: Run piece detection independently (always, regardless of corners)
    piece_results = detector.yolo.predict(image_rgb, conf=args.conf, verbose=False)
    raw_detections = []
    if len(piece_results) > 0 and len(piece_results[0].boxes) > 0:
        for box in piece_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            raw_detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "class": CLASS_NAMES[cls],
                "piece": CLASS_TO_PIECE[cls],
                "confidence": conf,
            })

    # Step 3: Run full FEN pipeline (maps pieces to squares if corners found)
    result = detector.detect_fen_with_metadata(image_rgb)
    fen = result["fen"]
    mapped_pieces = result["pieces"]

    print(f"Corners: {'detected' if corners is not None else 'NOT detected'}")
    print(f"Raw YOLO detections: {len(raw_detections)}")
    print(f"Mapped to squares: {len(mapped_pieces)}")
    print(f"FEN: {fen}")
    if args.verbose:
        for det in raw_detections:
            x1, y1, x2, y2 = det["bbox"]
            print(f"  [{det['class']:<16s} conf={det['confidence']:.3f}] "
                  f"bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        if mapped_pieces:
            print("Mapped pieces:")
            for p in sorted(mapped_pieces, key=lambda x: x["square"]):
                print(f"  {p['square']:>3s}  {p['class']:<16s} conf={p['confidence']:.3f}")

    vis = draw_annotations(frame, corners, raw_detections, mapped_pieces, fen)

    out_path = args.output or args.image.with_stem(args.image.stem + "_fen_viz")
    cv2.imwrite(str(out_path), vis)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
