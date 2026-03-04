#!/usr/bin/env python3
"""Compare corner detection between fine-tuned and base YOLO pose models."""

import cv2
import numpy as np
from pathlib import Path

FINETUNED = "models/yolo_corners.pt"
BASE = "yolo26n-pose.pt"

def draw_corners(img, corners, label, conf=None):
    """Draw detected corners on image copy."""
    vis = img.copy()
    labels = ["TL", "TR", "BR", "BL"]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
    if corners is not None:
        for i, (x, y) in enumerate(corners):
            cv2.circle(vis, (int(x), int(y)), 8, colors[i], -1)
            cv2.putText(vis, labels[i], (int(x)+10, int(y)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)
        # Draw board outline
        pts = corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
    status = "DETECTED" if corners is not None else "FAILED"
    text = f"{label}: {status}"
    if conf is not None:
        text += f" (conf={conf:.3f})"
    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if corners is not None else (0, 0, 255), 2)
    return vis


def detect_with_model(image, weights, conf=0.25):
    """Run detection and return corners + confidence."""
    from ultralytics import YOLO
    model = YOLO(weights)
    results = model.predict(image, verbose=False, conf=conf)
    if len(results) == 0 or results[0].keypoints is None:
        return None, None
    kpts = results[0].keypoints
    if len(kpts.xy) == 0:
        return None, None
    # Pick detection with largest keypoint area
    best_corners = None
    best_area = 0.0
    best_conf = 0.0
    confs = results[0].boxes.conf if results[0].boxes is not None else []
    for i in range(len(kpts.xy)):
        pts = kpts.xy[i].cpu().numpy().astype(np.float32)
        c = float(confs[i]) if i < len(confs) else 0.0
        if pts.shape[0] < 4:
            continue
        xs, ys = pts[:, 0], pts[:, 1]
        area = (xs.max() - xs.min()) * (ys.max() - ys.min())
        if area > best_area:
            best_area = area
            best_corners = pts[:4]
            best_conf = c
    return best_corners, best_conf


def main():
    # Capture frame from overhead camera
    print("Capturing from overhead camera (index 1)...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: Could not open camera 1")
        return 1
    # Warm up
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: Failed to capture frame")
        return 1
    print(f"Captured frame: {frame.shape}, mean brightness: {frame.mean():.0f}")
    cv2.imwrite("/tmp/corner_test_raw.png", frame)

    # Also try gamma-corrected version
    gamma = 0.5
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    bright = cv2.LUT(frame, table)
    print(f"Gamma-corrected: mean brightness: {bright.mean():.0f}")

    # Test both models on both raw and gamma-corrected
    for label, weights in [("FINETUNED", FINETUNED), ("BASE_YOLO26N", BASE)]:
        if not Path(weights).exists() and not weights.startswith("yolo26"):
            print(f"SKIP {label}: {weights} not found")
            continue
        print(f"\n--- {label} ({weights}) ---")

        # Raw
        corners, conf = detect_with_model(frame, weights)
        vis = draw_corners(frame, corners, f"{label}_raw", conf)
        path = f"/tmp/corner_test_{label.lower()}_raw.png"
        cv2.imwrite(path, vis)
        print(f"  Raw: {'DETECTED' if corners is not None else 'FAILED'}"
              + (f" conf={conf:.3f}" if conf else ""))
        if corners is not None:
            for i, (name) in enumerate(["TL", "TR", "BR", "BL"]):
                print(f"    {name}: ({corners[i][0]:.0f}, {corners[i][1]:.0f})")

        # Gamma
        corners_g, conf_g = detect_with_model(bright, weights)
        vis_g = draw_corners(bright, corners_g, f"{label}_gamma", conf_g)
        path_g = f"/tmp/corner_test_{label.lower()}_gamma.png"
        cv2.imwrite(path_g, vis_g)
        print(f"  Gamma: {'DETECTED' if corners_g is not None else 'FAILED'}"
              + (f" conf={conf_g:.3f}" if conf_g else ""))
        if corners_g is not None:
            for i, (name) in enumerate(["TL", "TR", "BR", "BL"]):
                print(f"    {name}: ({corners_g[i][0]:.0f}, {corners_g[i][1]:.0f})")

    print(f"\nSaved images to /tmp/corner_test_*.png")
    return 0


if __name__ == "__main__":
    exit(main())
