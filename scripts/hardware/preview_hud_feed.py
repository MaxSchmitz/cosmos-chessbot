#!/usr/bin/env python3
"""Live preview of the HUD overlay as it would be sent to the pi0.5 server.

Shows a cv2 window with the egocentric camera feed + HUD circles drawn
in real time. This is exactly what the policy sees during inference.

Usage:
    uv run python scripts/hardware/preview_hud_feed.py --source b2 --target b4
    uv run python scripts/hardware/preview_hud_feed.py --source e2 --target e4 --cam 1

Press 'q' to quit.
"""

import argparse

import cv2
import numpy as np
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

from cosmos_chessbot.vision.hud_overlay import (
    apply_hud,
    compute_homography,
    detect_corners,
)


def main():
    parser = argparse.ArgumentParser(description="Live HUD overlay preview")
    parser.add_argument("--source", required=True, help="Source square (e.g. e2)")
    parser.add_argument("--target", required=True, help="Target square (e.g. e4)")
    parser.add_argument("--cam", type=int, default=1, help="Camera index (default: 1 = overhead)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    # Open camera via lerobot (same path as run_pi05_episode.py)
    cam = OpenCVCamera(OpenCVCameraConfig(
        index_or_path=args.cam,
        width=args.width,
        height=args.height,
        fps=args.fps,
    ))
    cam.connect()
    print(f"Camera {args.cam} connected ({args.width}x{args.height} @ {args.fps}fps)")

    # Detect corners once from first frame
    frame = cam.async_read()
    if isinstance(frame, dict):
        frame = frame["frame"]
    frame = np.array(frame, dtype=np.uint8)
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = np.transpose(frame, (1, 2, 0))

    # detect_corners uses YOLO which handles RGB/BGR internally
    corners = detect_corners(frame)
    if corners is None:
        print("ERROR: could not detect board corners")
        cam.disconnect()
        return
    homography = compute_homography(corners)
    print(f"Board corners detected. Showing HUD: {args.source} -> {args.target}")
    print("Press 'q' to quit.")

    while True:
        frame = cam.async_read()
        if isinstance(frame, dict):
            frame = frame["frame"]
        frame = np.array(frame, dtype=np.uint8)
        if frame.ndim == 3 and frame.shape[0] == 3:
            frame = np.transpose(frame, (1, 2, 0))

        # This is exactly what run_pi05_episode.py does before sending to server
        apply_hud(frame, args.source, args.target, corners=corners, homography=homography)

        # frame is RGB (lerobot default), convert to BGR for cv2.imshow display
        cv2.imshow("HUD Feed (what policy sees)", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.disconnect()
    cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    main()
