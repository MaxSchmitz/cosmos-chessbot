#!/usr/bin/env python3
"""Capture a frame from camera and save to disk."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from cosmos_chessbot.vision import Camera, CameraConfig


def main():
    camera_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/raw/frame.jpg")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Capturing from camera {camera_id}...")

    config = CameraConfig(device_id=camera_id, name=f"camera_{camera_id}")

    with Camera(config) as camera:
        camera.save_frame(output_path)

    print(f"Frame saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
