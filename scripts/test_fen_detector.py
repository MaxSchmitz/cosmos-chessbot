#!/usr/bin/env python3
"""Test FEN detection with Ultimate V2 + YOLO pipeline.

Usage:
    python scripts/test_fen_detector.py
    python scripts/test_fen_detector.py --image path/to/chess_board.jpg
"""

import argparse
from pathlib import Path
from PIL import Image

from cosmos_chessbot.vision import FENDetector


def main():
    parser = argparse.ArgumentParser(
        description="Test FEN detection pipeline"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to chess board image (default: use first image in data/raw/)"
    )
    parser.add_argument(
        "--save-viz",
        type=Path,
        default=None,
        help="Path to save visualization (default: don't save)"
    )

    args = parser.parse_args()

    # Find test image
    if args.image is None:
        # Use first PNG or JPG in data/raw
        data_dir = Path("data/raw")
        for pattern in ["*.png", "*.jpg", "*.jpeg"]:
            images = list(data_dir.glob(pattern))
            if images:
                args.image = images[0]
                break

        if args.image is None:
            print("ERROR: No test images found in data/raw/")
            print("Please specify an image with --image")
            return 1

    if not args.image.exists():
        print(f"ERROR: Image not found: {args.image}")
        return 1

    print("=" * 80)
    print("FEN Detection Test")
    print("=" * 80)
    print(f"Image: {args.image}")
    print()

    # Initialize detector
    print("Loading models...")
    try:
        detector = FENDetector()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print()
        print("Make sure you've downloaded the models:")
        print("  cd models/")
        print("  wget https://huggingface.co/yamero999/ultimate-v2-chess-onnx/resolve/main/ultimate_v2_breakthrough_accurate.onnx")
        print("  wget https://huggingface.co/dopaul/chessboard-detector/resolve/main/best.pt -O chess_piece_yolo.pt")
        return 1

    print("Models loaded!")
    print()

    # Load image
    print("Loading image...")
    image = Image.open(args.image)
    print(f"Image size: {image.size}")
    print()

    # Detect FEN
    print("Running FEN detection...")
    try:
        fen = detector.detect_fen(image)

        print("=" * 80)
        print("RESULT")
        print("=" * 80)
        print(f"FEN: {fen}")
        print()

        # Parse FEN components
        parts = fen.split()
        if len(parts) >= 1:
            print(f"Position:     {parts[0]}")
        if len(parts) >= 2:
            print(f"Active color: {parts[1]} ({'White' if parts[1] == 'w' else 'Black'})")
        if len(parts) >= 3:
            print(f"Castling:     {parts[2]}")
        if len(parts) >= 4:
            print(f"En passant:   {parts[3]}")
        print()

        # Visualize
        if args.save_viz:
            print(f"Saving visualization to {args.save_viz}...")
            vis_image, _ = detector.visualize_detection(image, save_path=args.save_viz)
            print("Done!")
        else:
            print("Tip: Use --save-viz to save visualization")

        print()
        print("=" * 80)
        print("✓ FEN detection successful!")
        print("=" * 80)

        return 0

    except Exception as e:
        print()
        print("=" * 80)
        print("ERROR during FEN detection")
        print("=" * 80)
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
