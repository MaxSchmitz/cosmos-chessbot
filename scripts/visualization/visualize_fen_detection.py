#!/usr/bin/env python3
"""
Visualize FEN detection on an image or live camera feed.

Usage:
    # Single image:
    uv run python scripts/visualization/visualize_fen_detection.py path/to/image.jpg
    uv run python scripts/visualization/visualize_fen_detection.py path/to/image.jpg -o output.jpg

    # Live camera feed:
    uv run python scripts/visualization/visualize_fen_detection.py --live
    uv run python scripts/visualization/visualize_fen_detection.py --live --camera 1

    # Without DINO-MLP:
    uv run python scripts/visualization/visualize_fen_detection.py --live --no-dino
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from cosmos_chessbot.vision.yolo_dino_detector import (
    YOLODINOFenDetector, CLASS_NAMES, CLASS_TO_PIECE,
)

PIECE_WEIGHTS = "models/yolo_pieces.pt"
CORNER_WEIGHTS = "models/yolo_corners.pt"


def draw_annotations(image, corners, raw_detections, mapped_pieces, fen, fps=None):
    """Draw board pose, mapped piece boxes, and FEN onto a BGR image."""
    vis = image.copy()
    h, w = vis.shape[:2]

    scale = max(w, h) / 1000
    thick = max(2, int(3 * scale))
    thin = max(1, int(2 * scale))
    radius = max(6, int(12 * scale))
    font_big = 0.7 * scale
    font_med = 0.55 * scale

    # -- Board pose: outline + corner keypoints --
    if corners is not None:
        pts = corners.astype(np.int32)
        for i in range(4):
            cv2.line(vis, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), thick)

        colors = [(0, 255, 0), (0, 165, 255), (0, 0, 255), (255, 0, 0)]
        labels = ["TL", "TR", "BR", "BL"]
        for pt, col, lbl in zip(pts, colors, labels):
            cv2.circle(vis, tuple(pt), radius, col, -1)
            cv2.circle(vis, tuple(pt), radius + 2, (255, 255, 255), thin)
            cv2.putText(vis, lbl, (pt[0] + radius + 4, pt[1] - radius // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_med, col, thin + 1, cv2.LINE_AA)

    # -- Mapped pieces: green box, FEN char + conf + square --
    for det in mapped_pieces:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        piece_char = det["piece"]
        conf = det["confidence"]
        square = det["square"]
        color = (0, 255, 0)
        label = f"{piece_char} {conf:.2f} {square}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thin + 1)
        (tw, th_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_med, thin + 1)
        cv2.rectangle(vis, (x1, y1 - th_text - 8), (x1 + tw + 6, y1), (0, 0, 0), -1)
        cv2.putText(vis, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_med, color, thin + 1, cv2.LINE_AA)

    # -- Status + FEN text at bottom --
    lines = []
    if fps is not None:
        lines.append(f"{fps:.1f} fps")
    if corners is None:
        lines.append("Board pose: NOT DETECTED")
    lines.append(f"{len(mapped_pieces)} pieces")
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


def run_detection(detector, image_rgb, conf):
    """Run full pipeline: corners + pieces + FEN mapping."""
    corners = detector._detect_corners(image_rgb)

    piece_results = detector.yolo.predict(image_rgb, conf=conf, verbose=False)
    raw_detections = []
    if len(piece_results) > 0 and len(piece_results[0].boxes) > 0:
        for box in piece_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0])
            c = float(box.conf[0])
            raw_detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "class": CLASS_NAMES[cls],
                "piece": CLASS_TO_PIECE[cls],
                "confidence": c,
            })

    result = detector.detect_fen_with_metadata(image_rgb)
    return corners, raw_detections, result["pieces"], result["fen"]


def run_live(detector, args):
    """Live camera feed with FEN overlay."""
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print(f"Error: could not open camera {args.camera}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {args.camera}: {w}x{h}")
    print("Press 'q' to quit, 's' to save frame")

    # Video writer
    writer = None
    frame_count = 0
    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, 30, (w, h))
        print(f"Recording to: {args.record}")

    # Warmup
    for _ in range(10):
        cap.grab()

    frame_idx = 0
    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")
                break

            image_rgb = frame[:, :, ::-1]
            corners, raw, mapped, fen = run_detection(detector, image_rgb, args.conf)
            dt = time.perf_counter() - t0
            fps = 1.0 / dt if dt > 0 else 0

            vis = draw_annotations(frame, corners, raw, mapped, fen, fps=fps)

            if writer:
                writer.write(vis)
                frame_count += 1

            cv2.imshow("FEN Detection", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                out = f"/tmp/fen_live_{frame_idx:04d}.jpg"
                cv2.imwrite(out, vis)
                print(f"Saved {out}  FEN: {fen}")
                frame_idx += 1
    finally:
        if writer:
            writer.release()
            print(f"Saved {frame_count} frames to {args.record}")
        cap.release()
        cv2.destroyAllWindows()


def run_image(detector, args):
    """Single image detection."""
    frame = cv2.imread(str(args.image))
    if frame is None:
        print(f"Error: could not read image: {args.image}")
        sys.exit(1)

    image_rgb = frame[:, :, ::-1]
    corners, raw, mapped, fen = run_detection(detector, image_rgb, args.conf)

    print(f"Corners: {'detected' if corners is not None else 'NOT detected'}")
    print(f"Raw YOLO detections: {len(raw)}")
    print(f"Mapped to squares: {len(mapped)}")
    print(f"FEN: {fen}")
    if args.verbose:
        for det in raw:
            x1, y1, x2, y2 = det["bbox"]
            print(f"  [{det['class']:<16s} conf={det['confidence']:.3f}] "
                  f"bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        if mapped:
            print("Mapped pieces:")
            for p in sorted(mapped, key=lambda x: x["square"]):
                print(f"  {p['square']:>3s}  {p['class']:<16s} conf={p['confidence']:.3f}")

    vis = draw_annotations(frame, corners, raw, mapped, fen)

    out_path = args.output or args.image.with_stem(args.image.stem + "_fen_viz")
    cv2.imwrite(str(out_path), vis)
    print(f"Saved: {out_path}")


def run_benchmark(detector, args):
    """Run N detections from camera and log results."""
    from collections import Counter

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print(f"Error: could not open camera {args.camera}")
        sys.exit(1)

    # Warmup
    for _ in range(10):
        cap.grab()

    n = args.benchmark
    fen_counts = Counter()
    all_mapped = []

    print(f"Running {n} detections...")
    print("-" * 70)

    for i in range(n):
        ret, frame = cap.read()
        if not ret:
            print(f"  Frame {i}: camera read failed")
            continue

        image_rgb = frame[:, :, ::-1]
        t0 = time.perf_counter()
        corners, raw, mapped, fen = run_detection(detector, image_rgb, args.conf)
        dt = time.perf_counter() - t0

        fen_counts[fen] += 1
        all_mapped.append(mapped)

        pieces_str = " ".join(f"{p['piece']}{p['square']}" for p in
                              sorted(mapped, key=lambda x: x["square"]))
        print(f"  [{i+1:2d}] {dt:.2f}s  {len(mapped):2d} pieces  {fen}")
        if args.verbose:
            print(f"       {pieces_str}")

    cap.release()

    # Summary
    print("-" * 70)
    print(f"\nFEN frequency ({len(fen_counts)} unique across {n} frames):")
    for fen, count in fen_counts.most_common():
        print(f"  {count:3d}x  {fen}")

    # Per-square consistency
    square_pieces = {}  # square -> Counter of piece chars
    for mapped in all_mapped:
        for p in mapped:
            sq = p["square"]
            if sq not in square_pieces:
                square_pieces[sq] = Counter()
            square_pieces[sq][p["piece"]] += 1

    print(f"\nPer-square detection rate (across {n} frames):")
    for sq in sorted(square_pieces.keys()):
        counts = square_pieces[sq]
        total = sum(counts.values())
        parts = " ".join(f"{piece}={cnt}" for piece, cnt in counts.most_common())
        print(f"  {sq}: {total:2d}/{n} frames  ({parts})")


def main():
    parser = argparse.ArgumentParser(description="Visualize FEN detection")
    parser.add_argument("image", type=Path, nargs="?", default=None,
                        help="Input image path (omit for --live mode)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path (default: <input>_fen_viz.jpg)")
    parser.add_argument("--live", action="store_true",
                        help="Live camera feed with FEN overlay")
    parser.add_argument("--benchmark", type=int, default=None, metavar="N",
                        help="Run N detections from camera and log results")
    parser.add_argument("--camera", type=int, default=1,
                        help="Camera index for live mode (default: 1)")
    parser.add_argument("--piece-weights", type=str, default=PIECE_WEIGHTS)
    parser.add_argument("--corner-weights", type=str, default=CORNER_WEIGHTS)
    parser.add_argument("--corners", type=Path, default=None,
                        help="Path to calibrated corners JSON (skips pose model)")
    parser.add_argument("--conf", type=float, default=0.10,
                        help="YOLO confidence threshold (default: 0.10)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--record", type=str, default=None,
                        help="Output video file (e.g. shots/yolo_demo.mp4)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-dino", action="store_true",
                        help="Disable DINO-MLP classifier (use YOLO classes only)")
    args = parser.parse_args()

    if not args.live and args.benchmark is None and args.image is None:
        parser.error("provide an image path, --live, or --benchmark N")

    use_static_corners = args.corners is not None and args.corners.exists()

    mlp_path = Path("models/dino_mlp/dino_mlp_best.pth")
    use_dino = mlp_path.exists() and not args.no_dino

    detector = YOLODINOFenDetector(
        yolo_weights=args.piece_weights,
        corner_weights=None if use_static_corners else args.corner_weights,
        mlp_weights=str(mlp_path) if use_dino else None,
        device=args.device,
        conf_threshold=args.conf,
        use_dino=use_dino,
        static_corners=str(args.corners) if use_static_corners else None,
    )

    if args.benchmark:
        run_benchmark(detector, args)
    elif args.live:
        run_live(detector, args)
    else:
        run_image(detector, args)


if __name__ == "__main__":
    main()
