#!/usr/bin/env python3
"""
Train YOLO26 pose model for chessboard corner detection.

Detects 4 keypoints (board corners) used to compute a homography
for mapping piece pixel positions to board squares.

Keypoint order (clockwise from top-left):
    0: top_left
    1: top_right
    2: bottom_right
    3: bottom_left

Usage:
    uv run python scripts/train_yolo26_corners.py

    # Resume from checkpoint:
    uv run python scripts/train_yolo26_corners.py --resume runs/pose/board_corners/weights/last.pt
"""

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not found")
    print("Install with: uv add ultralytics")
    exit(1)


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26 pose for board corner detection")
    parser.add_argument('--data', type=Path, default=Path('data/chessred2k_pose/board_corners.yaml'))
    parser.add_argument('--model', type=str, default='yolo26n-pose.pt')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Error: Dataset YAML not found: {args.data}")
        print("Run: uv run python scripts/convert_chessred2k_corners.py")
        exit(1)

    print("=" * 60)
    print("YOLO26 Board Corner Detection Training")
    print("=" * 60)
    print(f"Dataset:  {args.data}")
    print(f"Model:    {args.model}")
    print(f"Epochs:   {args.epochs}")
    print(f"Batch:    {args.batch}")
    print(f"Device:   {args.device}")
    print(f"Resume:   {args.resume or 'No'}")
    print("=" * 60)

    if args.resume:
        print(f"\nResuming from {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"\nLoading pretrained {args.model}")
        model = YOLO(args.model)

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        resume=bool(args.resume),
        optimizer='auto',
        patience=15,
        save=True,
        save_period=10,
        project='runs/pose',
        name='board_corners',
        exist_ok=True,
        verbose=True,
        seed=42,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best weights: runs/pose/board_corners/weights/best.pt")
    print(f"Last weights: runs/pose/board_corners/weights/last.pt")


if __name__ == '__main__':
    main()
