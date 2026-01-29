#!/usr/bin/env python3
"""Test remote Cosmos perception via server."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from cosmos_chessbot.vision import RemoteCosmosPerception
from PIL import Image


def main():
    if len(sys.argv) < 3:
        print("Usage: python test_remote_perception.py <server_url> <image_path>")
        print("\nExample:")
        print("  python test_remote_perception.py http://gpu-server:8000 assets/test_board.jpg")
        return 1

    server_url = sys.argv[1]
    image_path = Path(sys.argv[2])

    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return 1

    print(f"Connecting to Cosmos server at {server_url}...")
    perception = RemoteCosmosPerception(server_url=server_url)

    print("Checking server health...")
    if not perception.health_check():
        print("Error: Server is not healthy or model not loaded")
        return 1

    print("Server is healthy!")

    print(f"Loading image from {image_path}...")
    image = Image.open(image_path)

    print("Running remote perception...")
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
