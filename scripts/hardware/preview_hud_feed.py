#!/usr/bin/env python3
"""Live preview of the HUD overlay with optional video recording.

Shows a cv2 window with the egocentric camera feed + HUD circles drawn
in real time. This is exactly what the policy sees during inference.

Usage:
    # Preview only
    uv run python scripts/hardware/preview_hud_feed.py --source b2 --target b4

    # Record to file
    uv run python scripts/hardware/preview_hud_feed.py --source e2 --target e4 --record demo.mp4

    # No HUD (raw camera)
    uv run python scripts/hardware/preview_hud_feed.py --no-hud --record raw.mp4

    # With move label overlay
    uv run python scripts/hardware/preview_hud_feed.py --source e2 --target e4 --label "e2 -> e4"

Press 'q' to quit, 'r' to toggle recording mid-session, SPACE to pause/resume recording.
"""

import argparse
import time

import cv2
import numpy as np
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

from cosmos_chessbot.vision.hud_overlay import (
    apply_hud,
    compute_homography,
    detect_corners,
)


def draw_label(frame, text, position="top-left"):
    """Draw a text label with background on the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    color = (255, 255, 255)
    bg_color = (0, 0, 0)

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 8

    if position == "top-left":
        x, y = pad, pad + th
    elif position == "top-right":
        x, y = frame.shape[1] - tw - pad * 2, pad + th
    elif position == "bottom-left":
        x, y = pad, frame.shape[0] - pad - baseline
    else:
        x, y = pad, pad + th

    # Background rectangle
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_rec_indicator(frame, recording):
    """Draw a red recording indicator in top-right corner."""
    if not recording:
        return
    h, w = frame.shape[:2]
    cv2.circle(frame, (w - 25, 25), 10, (255, 0, 0), -1, cv2.LINE_AA)
    cv2.putText(frame, "REC", (w - 70, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Live HUD overlay preview + recorder")
    parser.add_argument("--source", default=None, help="Source square (e.g. e2)")
    parser.add_argument("--target", default=None, help="Target square (e.g. e4)")
    parser.add_argument("--cam", type=int, default=1, help="Camera index (default: 1 = overhead)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--record", type=str, default=None, help="Output video file (e.g. demo.mp4)")
    parser.add_argument("--no-hud", action="store_true", help="Disable HUD overlay (raw camera)")
    parser.add_argument("--label", type=str, default=None, help="Text label to overlay (e.g. 'e2 -> e4')")
    args = parser.parse_args()

    if not args.no_hud and (args.source is None or args.target is None):
        parser.error("--source and --target required unless --no-hud is set")

    # Open camera via lerobot (same path as run_pi05_episode.py)
    cam = OpenCVCamera(OpenCVCameraConfig(
        index_or_path=args.cam,
        width=args.width,
        height=args.height,
        fps=args.fps,
    ))
    cam.connect()
    print(f"Camera {args.cam} connected ({args.width}x{args.height} @ {args.fps}fps)")

    if not args.no_hud:
        print(f"HUD: {args.source} -> {args.target}")
    print("Keys: 'q'=quit, 'r'=toggle recording, SPACE=pause recording")

    # Video writer
    writer = None
    recording = False
    if args.record:
        from pathlib import Path
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, args.fps, (args.width, args.height))
        recording = True
        print(f"Recording to: {args.record}")

    # Sticky corners
    last_corners = None
    last_homography = None
    min_update_conf = 0.97
    frame_count = 0

    try:
        while True:
            try:
                frame = cam.async_read()
            except Exception:
                break
            if isinstance(frame, dict):
                frame = frame["frame"]
            frame = np.array(frame, dtype=np.uint8)
            if frame.ndim == 3 and frame.shape[0] == 3:
                frame = np.transpose(frame, (1, 2, 0))

            # Corner detection + HUD
            if not args.no_hud:
                corners, detection_conf = detect_corners(frame, return_conf=True)
                if corners is not None:
                    if frame_count % 30 == 0:
                        print(f"  corner conf: {detection_conf:.3f} {'*' if detection_conf >= min_update_conf else ''}")
                    if detection_conf >= min_update_conf:
                        last_corners = corners
                        last_homography = compute_homography(corners)

                if last_corners is not None:
                    apply_hud(frame, args.source, args.target,
                              corners=last_corners, homography=last_homography)

            # Write frame BEFORE display overlays (record clean feed)
            if writer and recording:
                # Write RGB->BGR for mp4
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_count += 1

            # Display-only overlays (not recorded)
            display = frame.copy()
            if args.label:
                draw_label(display, args.label, position="bottom-left")
            draw_rec_indicator(display, recording)

            cv2.imshow("HUD Feed", cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                if writer is None:
                    # Start new recording
                    path = args.record or f"recording_{int(time.time())}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(path, fourcc, args.fps, (args.width, args.height))
                    recording = True
                    print(f"Started recording: {path}")
                else:
                    recording = not recording
                    print(f"Recording {'resumed' if recording else 'paused'}")
            elif key == ord(" "):
                if writer:
                    recording = not recording
                    print(f"Recording {'resumed' if recording else 'paused'}")

    finally:
        if writer:
            writer.release()
            duration = frame_count / args.fps if args.fps > 0 else 0
            print(f"Saved {frame_count} frames ({duration:.1f}s)")
        cam.disconnect()
        cv2.destroyAllWindows()
        print("Done")


if __name__ == "__main__":
    main()
