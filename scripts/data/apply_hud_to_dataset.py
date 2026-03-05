#!/usr/bin/env python3
"""Apply HUD overlays to an existing lerobot dataset.

Reads a raw dataset recorded with --random-moves (no HUD) plus its
moves.json, applies source/target markers to the egocentric camera
images, and writes a new dataset ready for pi0.5 training.

Usage:
    uv run python scripts/data/apply_hud_to_dataset.py \
        --src-repo-id maux/chess-random-raw \
        --dst-repo-id maux/chess-random-hud
"""

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.utils.constants import HF_LEROBOT_HOME

from cosmos_chessbot.vision.hud_overlay import (
    apply_hud,
    compute_homography,
    detect_corners,
)

DEFAULT_TASK = "Move the small object from the green circle to the magenta circle"


def tensor_to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) float32 [0,1] tensor to (H, W, C) uint8 numpy array."""
    if img_tensor.dtype == torch.float32:
        arr = (img_tensor * 255).clamp(0, 255).byte()
    else:
        arr = img_tensor
    # (C, H, W) -> (H, W, C), contiguous for OpenCV
    return np.ascontiguousarray(arr.permute(1, 2, 0).numpy())


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Apply HUD overlays to a lerobot dataset")
    parser.add_argument("--src-repo-id", required=True, help="Source dataset repo ID")
    parser.add_argument("--dst-repo-id", required=True, help="Destination dataset repo ID")
    parser.add_argument("--src-root", type=str, default=None,
                        help="Source dataset root (default: HF_LEROBOT_HOME/src-repo-id)")
    parser.add_argument("--dst-root", type=str, default=None,
                        help="Destination dataset root (default: HF_LEROBOT_HOME/dst-repo-id)")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        help="Task string for the output dataset")
    parser.add_argument("--corner-weights", type=str, default=None,
                        help="Path to YOLO pose model for corner detection")
    args = parser.parse_args()

    # Resolve paths
    src_root = Path(args.src_root) if args.src_root else HF_LEROBOT_HOME / args.src_repo_id
    dst_root = Path(args.dst_root) if args.dst_root else HF_LEROBOT_HOME / args.dst_repo_id

    # Load moves.json
    moves_path = src_root / "moves.json"
    if not moves_path.exists():
        raise FileNotFoundError(f"No moves.json found at {moves_path}")
    with open(moves_path) as f:
        moves_data = json.load(f)
    print(f"Loaded {len(moves_data)} moves from {moves_path}")

    # Load source dataset
    print(f"Loading source dataset: {args.src_repo_id}")
    src_ds = LeRobotDataset(args.src_repo_id, root=src_root)
    print(f"  Episodes: {src_ds.num_episodes}, Frames: {src_ds.num_frames}, FPS: {src_ds.fps}")

    # Build destination features: copy from source, preserving video format
    dst_features = {}
    for key, feat in src_ds.features.items():
        if key in DEFAULT_FEATURES:
            continue
        dst_features[key] = dict(feat)

    # Check destination doesn't already exist
    if dst_root.exists():
        raise FileExistsError(f"Destination already exists: {dst_root}")

    # Create destination dataset (use_videos=True to keep compressed format)
    print(f"Creating destination dataset: {args.dst_repo_id}")
    dst_ds = LeRobotDataset.create(
        repo_id=args.dst_repo_id,
        fps=src_ds.fps,
        root=dst_root,
        features=dst_features,
        robot_type=src_ds.meta.info.get("robot_type"),
        use_videos=True,
        image_writer_threads=4,
    )

    # Identify the egocentric camera key
    ego_key = None
    for key in src_ds.features:
        if "egocentric" in key and src_ds.features[key]["dtype"] in ("image", "video"):
            ego_key = key
            break
    if ego_key is None:
        raise RuntimeError("Could not find egocentric camera feature in source dataset")
    print(f"  Egocentric key: {ego_key}")

    # Process each episode
    feature_keys = [k for k in dst_features if k not in DEFAULT_FEATURES]
    total_frames = 0

    for ep_idx in range(src_ds.num_episodes):
        ep_key = str(ep_idx)
        if ep_key not in moves_data:
            print(f"  WARNING: episode {ep_idx} not in moves.json, skipping")
            continue

        move = moves_data[ep_key]
        src_sq = move["source"]
        dst_sq = move["target"]

        # Get episode frame range
        ep_meta = src_ds.meta.episodes[ep_idx]
        ep_start = ep_meta["dataset_from_index"]
        ep_end = ep_meta["dataset_to_index"]
        ep_len = ep_end - ep_start

        # Corner detection: use first frame of episode
        first_frame = src_ds[ep_start]
        first_img = tensor_to_uint8(first_frame[ego_key])

        detect_kwargs = {"corner_weights": args.corner_weights} if args.corner_weights else {}
        corners = detect_corners(first_img, **detect_kwargs)
        if corners is not None:
            homography = compute_homography(corners)
        else:
            print(f"  WARNING: could not detect corners for episode {ep_idx}, skipping HUD")
            homography = None

        desc = f"  Episode {ep_idx} ({src_sq}->{dst_sq}, {ep_len} frames)"
        for frame_offset in tqdm(range(ep_len), desc=desc, leave=False):
            abs_idx = ep_start + frame_offset
            frame = src_ds[abs_idx]

            # Build output frame
            out_frame = {"task": args.task}
            for key in feature_keys:
                if key == ego_key:
                    img = tensor_to_uint8(frame[key])
                    if corners is not None:
                        apply_hud(img, src_sq, dst_sq, corners=corners, homography=homography)
                    out_frame[key] = img
                elif src_ds.features[key]["dtype"] in ("image", "video"):
                    # Other camera: convert tensor to uint8 numpy
                    out_frame[key] = tensor_to_uint8(frame[key])
                else:
                    # Numeric features: keep as numpy
                    val = frame[key]
                    if isinstance(val, torch.Tensor):
                        val = val.numpy()
                    out_frame[key] = val

            dst_ds.add_frame(out_frame)

        dst_ds.save_episode()
        total_frames += ep_len
        print(f"  Episode {ep_idx}: {ep_len} frames saved ({src_sq} -> {dst_sq})")

    # Finalize
    dst_ds.finalize()
    print(f"\nDone: {src_ds.num_episodes} episodes, {total_frames} frames")
    print(f"Output: {dst_root}")

    # Copy moves.json for reference
    shutil.copy2(moves_path, dst_root / "moves.json")
    print(f"Copied moves.json to {dst_root / 'moves.json'}")


if __name__ == "__main__":
    main()
