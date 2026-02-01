#!/usr/bin/env python3
"""Test LLM-based FEN detection on test images."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cosmos_chessbot.vision.llm_fen_detector import LLMFenDetector
from PIL import Image


def filename_to_fen(filename: str) -> str:
    """Extract ground truth FEN from filename."""
    # Remove extension
    fen = filename.replace(".png", "").replace(".jpg", "")
    # Replace colons with slashes
    fen = fen.replace(":", "/")
    return fen


def test_llm_fen_detection():
    """Test LLM FEN detection on all test images."""
    # Get test images
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    test_images = sorted(data_dir.glob("*.png"))

    if not test_images:
        print("No test images found in data/raw/")
        return

    # Filter images with FEN in filename
    test_images = [img for img in test_images if ":" in img.stem]

    if not test_images:
        print("No test images with FEN notation in filename")
        return

    print(f"Found {len(test_images)} test images with FEN notation\n")

    # Initialize LLM detector with GPT-5.2
    try:
        detector = LLMFenDetector(provider="openai", model="gpt-5.2", detail="high")
        print(f"Using GPT-5.2\n")
    except ValueError as e:
        print(f"Error: {e}")
        print("\nTo use this script, set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-key-here'")
        return

    # Test each image
    results = []
    for img_path in test_images:
        print(f"Testing: {img_path.name}")
        print("-" * 80)

        # Get ground truth from filename
        ground_truth = filename_to_fen(img_path.stem)
        print(f"Ground Truth: {ground_truth}")

        # Detect FEN
        try:
            detected_fen = detector.detect_fen_from_path(str(img_path))
            print(f"Detected:     {detected_fen}")

            # Compare (just the piece placement part for now)
            gt_pieces = ground_truth.split()[0]
            det_pieces = detected_fen.split()[0]
            match = gt_pieces == det_pieces

            if match:
                print("✅ MATCH!\n")
            else:
                print("❌ MISMATCH\n")

            results.append({
                "image": img_path.name,
                "ground_truth": ground_truth,
                "detected": detected_fen,
                "match": match,
            })

        except Exception as e:
            print(f"❌ ERROR: {e}\n")
            results.append({
                "image": img_path.name,
                "ground_truth": ground_truth,
                "detected": None,
                "match": False,
                "error": str(e),
            })

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    matches = sum(1 for r in results if r["match"])
    total = len(results)

    print(f"Accuracy: {matches}/{total} ({100*matches/total:.1f}%)")

    if matches < total:
        print("\nFailed images:")
        for r in results:
            if not r["match"]:
                print(f"  - {r['image']}")
                if "error" in r:
                    print(f"    Error: {r['error']}")


if __name__ == "__main__":
    test_llm_fen_detection()
