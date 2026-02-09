#!/usr/bin/env python3
"""Test LiveChess2FEN on our chess board images."""

import sys
from pathlib import Path

# Add LiveChess2FEN to path
lc2fen_path = Path(__file__).parent.parent / "external" / "LiveChess2FEN"
sys.path.insert(0, str(lc2fen_path))

from keras.applications.xception import preprocess_input as prein_xception
from lc2fen.predict_board import predict_board_onnx
import cv2


def filename_to_fen(filename: str) -> str:
    """Extract ground truth FEN from filename."""
    fen = filename.replace(".png", "").replace(".jpg", "")
    fen = fen.replace(":", "/")
    return fen


def test_livechess2fen():
    """Test LiveChess2FEN on our test images."""
    # Configuration
    model_path = str(lc2fen_path / "data" / "models" / "Xception_last.onnx")
    img_size = 299
    pre_input = prein_xception

    # Get test images
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    test_images = sorted(data_dir.glob("*.png"))
    test_images = [img for img in test_images if ":" in img.stem]

    if not test_images:
        print("No test images found in data/raw/")
        return

    print(f"Found {len(test_images)} test images\n")
    print(f"Using model: {model_path}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Preprocessing: Xception\n")

    results = []
    for img_path in test_images:
        print(f"Testing: {img_path.name}")
        print("-" * 80)

        # Get ground truth
        ground_truth = filename_to_fen(img_path.stem)
        print(f"Ground Truth: {ground_truth}")

        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"❌ ERROR: Could not read image\n")
                continue

            # Predict FEN using LiveChess2FEN
            # Note: This function expects the full prediction pipeline
            # We'll need to adapt this based on the actual API
            print(f"Image loaded: {img.shape}")
            print("Note: LiveChess2FEN requires full pipeline setup")
            print("We'll need to integrate the board detection + piece classification\n")

        except Exception as e:
            print(f"❌ ERROR: {e}\n")

    print("\nNote: Full LiveChess2FEN integration requires:")
    print("1. Board detection (finds chessboard in image)")
    print("2. Square extraction (splits board into 64 squares)")
    print("3. Piece classification (identifies pieces using ONNX model)")
    print("4. FEN generation")


if __name__ == "__main__":
    test_livechess2fen()
