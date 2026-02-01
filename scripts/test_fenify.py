#!/usr/bin/env python3
"""Test Fenify-3D on our chess board images."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cosmos_chessbot.vision.fenify_detector import FenifyDetector
from PIL import Image


def filename_to_fen(filename: str) -> str:
    """Extract ground truth FEN from filename."""
    fen = filename.replace(".png", "").replace(".jpg", "")
    fen = fen.replace(":", "/")
    return fen


def test_fenify():
    """Test Fenify-3D on all test images."""
    # Get test images
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    test_images = sorted(data_dir.glob("*.png"))
    test_images = [img for img in test_images if ":" in img.stem]

    if not test_images:
        print("No test images found in data/raw/")
        return

    print(f"Found {len(test_images)} test images\n")

    # Initialize Fenify detector
    try:
        detector = FenifyDetector()
        print()
    except Exception as e:
        print(f"Error initializing Fenify detector: {e}")
        return

    # Test each image
    results = []
    for img_path in test_images:
        print(f"Testing: {img_path.name}")
        print("-" * 80)

        # Get ground truth
        ground_truth = filename_to_fen(img_path.stem)
        ground_truth_position = ground_truth.split()[0]  # Just piece placement
        print(f"Ground Truth: {ground_truth}")

        try:
            # Detect FEN
            detected_fen = detector.detect_fen_from_path(str(img_path))
            detected_position = detected_fen.split()[0] if detected_fen else ""
            print(f"Detected:     {detected_fen}")

            # Compare piece placement
            match = ground_truth_position == detected_position
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
    test_fenify()
