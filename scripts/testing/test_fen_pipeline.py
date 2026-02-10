#!/usr/bin/env python3
"""
End-to-end FEN detection pipeline test.

Captures a frame from the board camera, runs YOLO26 corner detection +
piece detection, maps pieces to squares via homography, and prints the
resulting FEN.  Saves an annotated image for visual verification.

Usage:
    uv run python scripts/test_fen_pipeline.py
    uv run python scripts/test_fen_pipeline.py --camera 0
    uv run python scripts/test_fen_pipeline.py --interactive   # live loop
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from cosmos_chessbot.vision.yolo_dino_detector import YOLODINOFenDetector, CLASS_NAMES

# ---------------------------------------------------------------------------
# Paths (ultralytics nested project/name, so the path is doubled)
# ---------------------------------------------------------------------------
PIECE_WEIGHTS = 'runs/detect/runs/detect/yolo26_chess/weights/best.pt'
CORNER_WEIGHTS = 'runs/pose/runs/pose/board_corners/weights/best.pt'


def draw_annotations(image: np.ndarray, corners, detections, fen: str) -> np.ndarray:
    """Draw corners, piece boxes, and FEN onto a BGR image."""
    vis = image.copy()
    h, w = vis.shape[:2]

    # --- Board outline & corner dots ---
    if corners is not None:
        # corners: (4,2) in order TL, TR, BR, BL
        pts = corners.astype(np.int32)
        # Draw perimeter
        order = [0, 1, 2, 3, 0]
        for i in range(len(order) - 1):
            cv2.line(vis, tuple(pts[order[i]]), tuple(pts[order[i + 1]]), (255, 255, 255), 2)
        # Corner dots with labels
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]
        labels = ['TL', 'TR', 'BR', 'BL']
        for idx, (pt, col, lbl) in enumerate(zip(pts, colors, labels)):
            cv2.circle(vis, tuple(pt), 8, col, -1)
            cv2.circle(vis, tuple(pt), 10, (255, 255, 255), 2)
            cv2.putText(vis, lbl, (pt[0] + 12, pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

    # --- Piece bounding boxes ---
    if detections is not None:
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            piece = det['piece']
            conf = det['confidence']
            color = (0, 200, 255) if piece.isupper() else (100, 100, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']} {conf:.2f} -> {det['square']}"
            cv2.putText(vis, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # --- FEN string at bottom ---
    cv2.putText(vis, f"FEN: {fen}", (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    return vis


def main():
    parser = argparse.ArgumentParser(description="End-to-end FEN pipeline test")
    parser.add_argument('--camera', type=int, default=1, help='Camera index (default: 1)')
    parser.add_argument('--piece-weights', type=str, default=PIECE_WEIGHTS)
    parser.add_argument('--corner-weights', type=str, default=CORNER_WEIGHTS)
    parser.add_argument('--corners', type=Path, default=None,
                        help='Path to calibrated corners JSON (skips pose model). '
                             'Generate with: uv run python scripts/calibrate_corners.py')
    parser.add_argument('--conf', type=float, default=0.10, help='YOLO confidence threshold')
    parser.add_argument('--output', type=Path, default=Path('data/fen_pipeline_test'))
    parser.add_argument('--interactive', action='store_true', help='Live loop with display window')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Validate weights exist
    if not Path(args.piece_weights).exists():
        print(f"Error: piece weights not found: {args.piece_weights}")
        sys.exit(1)

    use_static_corners = args.corners is not None and args.corners.exists()

    print("=" * 60)
    print("FEN Detection Pipeline Test")
    print("=" * 60)
    print(f"Piece weights:  {args.piece_weights}")
    if use_static_corners:
        print(f"Corners:        {args.corners} (static calibration)")
    else:
        print(f"Corner weights: {args.corner_weights}")
        if not Path(args.corner_weights).exists():
            print(f"Error: corner weights not found: {args.corner_weights}")
            sys.exit(1)
    print(f"Camera:         {args.camera}")
    print(f"Conf threshold: {args.conf}")
    print("=" * 60)

    # Load detector (use_dino=False -- DINO-MLP not trained yet)
    detector = YOLODINOFenDetector(
        yolo_weights=args.piece_weights,
        corner_weights=None if use_static_corners else args.corner_weights,
        mlp_weights=None,
        device='cpu',
        conf_threshold=args.conf,
        use_dino=False,
        static_corners=str(args.corners) if use_static_corners else None,
    )

    # Open camera and force 1920x1080
    print(f"\nOpening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {w}x{h}")

    # Warmup: discard frames until we get a bright one
    for _ in range(30):
        cap.grab()

    if args.interactive:
        print("\nLive mode -- press 'q' to quit, 's' to save frame")
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")
                break

            # BGR -> RGB for detector
            image_rgb = frame[:, :, ::-1]

            # Run pipeline
            corners = detector._detect_corners(image_rgb)
            result = detector.detect_fen_with_metadata(image_rgb)

            # Annotate
            vis = draw_annotations(frame, corners, result['pieces'], result['fen'])
            cv2.imshow('FEN Pipeline', vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                out_path = args.output / f"frame_{frame_idx:04d}.jpg"
                cv2.imwrite(str(out_path), vis)
                print(f"  Saved {out_path}  FEN: {result['fen']}")
                frame_idx += 1

        cv2.destroyAllWindows()
    else:
        # Single-shot: capture one frame, run, save
        print("Capturing frame...")
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            sys.exit(1)

        image_rgb = frame[:, :, ::-1]

        # Corner detection (verbose)
        corners = detector._detect_corners(image_rgb)
        if corners is not None:
            print(f"\nCorners detected:")
            labels = ['TL', 'TR', 'BR', 'BL']
            for lbl, pt in zip(labels, corners):
                print(f"  {lbl}: ({pt[0]:.1f}, {pt[1]:.1f})")
        else:
            print("\nNo board corners detected")

        # Full FEN with metadata (verbose mode)
        print("\nRunning FEN detection...")
        result = detector.detect_fen_with_metadata(image_rgb)

        fen = result['fen']
        pieces = result['pieces']
        confidence = result['confidence']

        print(f"\nFEN:        {fen}")
        print(f"Pieces:     {len(pieces)}")
        print(f"Confidence: {confidence:.3f}")
        if pieces:
            print("\nPiece map:")
            # Sort by square name for readability
            for p in sorted(pieces, key=lambda x: x['square']):
                print(f"  {p['square']:>3s}  {p['class']:<16s} conf={p['confidence']:.3f}")

        # Also run detect_fen with verbose for the debug printout
        print("\n--- detect_fen verbose ---")
        fen2 = detector.detect_fen(image_rgb, verbose=True)
        assert fen == fen2, f"FEN mismatch: {fen} vs {fen2}"

        # Save annotated image
        vis = draw_annotations(frame, corners, pieces, fen)
        out_path = args.output / 'annotated.jpg'
        cv2.imwrite(str(out_path), vis)
        print(f"\nAnnotated image saved: {out_path}")

        # Also save raw frame for reference
        raw_path = args.output / 'raw.jpg'
        cv2.imwrite(str(raw_path), frame)

    cap.release()


if __name__ == '__main__':
    main()
