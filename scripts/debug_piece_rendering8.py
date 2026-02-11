#!/usr/bin/env python3
"""Verify piece rendering fix: geometry child reference (primPath="/root/P0")."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
from cosmos_chessbot.isaac.chess_env import ChessPickPlaceEnv
from cosmos_chessbot.isaac.chess_env_cfg import ChessPickPlaceEnvCfg
import omni.usd

cfg = ChessPickPlaceEnvCfg()
cfg.scene.num_envs = 1
env = ChessPickPlaceEnv(cfg)

stage = omni.usd.get_context().get_stage()

# Check Pieces container type
pieces_prim = stage.GetPrimAtPath("/World/envs/env_0/Pieces")
print(f"Pieces container: valid={pieces_prim.IsValid()}, type='{pieces_prim.GetTypeName()}'")

# Check first few piece prims
for i in range(4):
    p = stage.GetPrimAtPath(f"/World/envs/env_0/Pieces/piece_{i}")
    children = [c.GetName() for c in p.GetChildren()] if p.IsValid() else []
    print(f"piece_{i}: valid={p.IsValid()}, type='{p.GetTypeName()}', children={children}")

# Reset and render (warmup)
obs, info = env.reset()
for i in range(60):
    action = torch.zeros((1, cfg.action_space), device="cuda:0")
    obs, _, _, _, _ = env.step(action)

# Count pieces at table height
count = 0
for i in range(32):
    p = stage.GetPrimAtPath(f"/World/envs/env_0/Pieces/piece_{i}")
    if p.IsValid():
        from pxr import UsdGeom, Gf
        xf = UsdGeom.Xformable(p)
        mtx = xf.ComputeLocalToWorldTransform(0)
        z = mtx[3][2]
        if z > 0.5:
            count += 1
print(f"Pieces at Z > 0.5 (on table): {count}")

# Capture frame
camera = env.scene["front"]
rgb_data = camera.data.output.get("rgb")
if rgb_data is not None:
    frame = rgb_data[0].cpu().numpy()
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[:, :, :3]
    frame = np.flipud(frame)
    import imageio
    imageio.imwrite("outputs/debug_rendering8.png", frame)
    print(f"Frame saved: outputs/debug_rendering8.png (shape={frame.shape})")

env.close()
simulation_app.close()
