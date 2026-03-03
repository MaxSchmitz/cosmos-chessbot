#!/usr/bin/env python3
"""Generate a contact sheet of HUD-overlaid first frames for quick QA.

For each episode, grabs the first frame from the egocentric camera,
applies HUD markers from moves.json, annotates with episode number and
move text, and tiles everything into a single image.

Usage:
    uv run python scripts/data/review_hud_episodes.py \
        --repo-id maux/chess-random-raw \
        --output review_hud.png
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

from cosmos_chessbot.vision.hud_overlay import (
    apply_hud,
    compute_homography,
    detect_corners,
)


def tensor_to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) float32 [0,1] tensor to (H, W, C) uint8 numpy array."""
    if img_tensor.dtype == torch.float32:
        arr = (img_tensor * 255).clamp(0, 255).byte()
    else:
        arr = img_tensor
    return np.ascontiguousarray(arr.permute(1, 2, 0).numpy())


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Review HUD overlays per episode")
    parser.add_argument("--repo-id", required=True, help="Dataset repo ID")
    parser.add_argument("--root", type=str, default=None, help="Dataset root")
    parser.add_argument("--output", type=str, default="review_hud.png", help="Output image path")
    parser.add_argument("--corner-weights", type=str, default=None, help="YOLO pose model path")
    parser.add_argument("--cols", type=int, default=10, help="Columns in contact sheet")
    parser.add_argument("--thumb-width", type=int, default=320, help="Thumbnail width")
    args = parser.parse_args()

    root = Path(args.root) if args.root else HF_LEROBOT_HOME / args.repo_id
    moves_path = root / "moves.json"
    if not moves_path.exists():
        raise FileNotFoundError(f"No moves.json at {moves_path}")

    with open(moves_path) as f:
        moves_data = json.load(f)

    ds = LeRobotDataset(args.repo_id, root=root)
    print(f"Dataset: {ds.num_episodes} episodes, {ds.num_frames} frames")

    # Find egocentric camera key
    ego_key = None
    for key in ds.features:
        if "egocentric" in key and ds.features[key]["dtype"] in ("image", "video"):
            ego_key = key
            break
    if ego_key is None:
        raise RuntimeError("No egocentric camera found")
    print(f"Camera key: {ego_key}")

    thumbs = []
    for ep_idx in range(ds.num_episodes):
        ep_key = str(ep_idx)
        move = moves_data.get(ep_key)
        if move is None:
            print(f"  Episode {ep_idx}: no move in moves.json, skipping")
            continue

        src_sq = move["source"]
        dst_sq = move["target"]

        # Get first frame of episode
        ep_meta = ds.meta.episodes[ep_idx]
        ep_start = ep_meta["dataset_from_index"]
        frame = ds[ep_start]
        img = tensor_to_uint8(frame[ego_key])

        # Detect corners and apply HUD
        detect_kwargs = {"corner_weights": args.corner_weights} if args.corner_weights else {}
        corners = detect_corners(img, **detect_kwargs)
        if corners is not None:
            homography = compute_homography(corners)
            apply_hud(img, src_sq, dst_sq, corners=corners, homography=homography)
            status = ""
        else:
            status = " [NO CORNERS]"

        # Annotate with episode number and move
        label = f"Ep {ep_idx}: {src_sq}->{dst_sq}{status}"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

        # Resize to thumbnail
        h, w = img.shape[:2]
        thumb_h = int(args.thumb_width * h / w)
        thumb = cv2.resize(img, (args.thumb_width, thumb_h))
        thumbs.append(thumb)

    if not thumbs:
        print("No episodes processed")
        return

    # Build contact sheet
    cols = min(args.cols, len(thumbs))
    rows = math.ceil(len(thumbs) / cols)
    th, tw = thumbs[0].shape[:2]

    sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = thumb

    cv2.imwrite(args.output, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"\nSaved contact sheet: {args.output} ({cols}x{rows}, {len(thumbs)} episodes)")


if __name__ == "__main__":
    main()
