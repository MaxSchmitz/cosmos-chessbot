#!/usr/bin/env python3
"""Test Cosmos perception on a static chess board image."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from cosmos_chessbot.vision import CosmosPerception
from PIL import Image


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_perception.py <image_path>")
        print("\nExample:")
        print("  python test_perception.py assets/test_board.jpg")
        return 1

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return 1

    print("Loading Cosmos-Reason2 model...")
    perception = CosmosPerception()

    print(f"Loading image from {image_path}...")
    image = Image.open(image_path)

    print("Running perception...")
    board_state = perception.perceive(image)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"FEN: {board_state.fen}")
    print(f"Confidence: {board_state.confidence:.2%}")

    if board_state.anomalies:
        print("\nAnomalies detected:")
        for anomaly in board_state.anomalies:
            print(f"  - {anomaly}")
    else:
        print("\nNo anomalies detected")

    print("\nRaw response:")
    print("-" * 60)
    print(board_state.raw_response)
    print("-" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
