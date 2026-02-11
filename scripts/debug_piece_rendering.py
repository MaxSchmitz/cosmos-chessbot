#!/usr/bin/env python3
"""Debug script to check piece rendering in the chess environment."""
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
from cosmos_chessbot.isaac.chess_env import ChessPickPlaceEnv
from cosmos_chessbot.isaac.chess_env_cfg import ChessPickPlaceEnvCfg
import omni.usd
from pxr import UsdGeom

cfg = ChessPickPlaceEnvCfg()
cfg.scene.num_envs = 1
env = ChessPickPlaceEnv(cfg)

print("\n" + "=" * 60)
print("DEBUG: After env creation, before reset")
print("=" * 60)

stage = omni.usd.get_context().get_stage()

# Check piece_0 prim tree
p0 = stage.GetPrimAtPath("/World/envs/env_0/Pieces/piece_0")
print(f"piece_0 valid={p0.IsValid()} type={p0.GetTypeName()} active={p0.IsActive()}")

# List ALL descendants
def print_tree(prim, indent=0):
    prefix = "  " * indent
    info = f"{prefix}{prim.GetName()} (type={prim.GetTypeName()}, active={prim.IsActive()}"
    xf = UsdGeom.Xformable(prim)
    if xf:
        ops = [(op.GetOpName(), op.GetOpType()) for op in xf.GetOrderedXformOps()]
        if ops:
            info += f", xformOps={[o[0] for o in ops]}"
    info += ")"
    print(info)
    for child in prim.GetChildren():
        print_tree(child, indent + 1)

print("\nPiece 0 tree:")
print_tree(p0)

print("\nBoard tree:")
board = stage.GetPrimAtPath("/World/envs/env_0/Board")
print_tree(board)

# Do a reset
print("\n" + "=" * 60)
print("DEBUG: After reset")
print("=" * 60)
obs, info = env.reset()

# Check physics positions
all_transforms = env._piece_rigid_view.get_transforms()
positions = all_transforms[:, :3]
active_mask = positions[:, 2] > -50.0
print(f"Physics view count: {env._piece_rigid_view.count}")
print(f"Pieces with Z > -50: {active_mask.sum().item()}")
for i in range(min(32, all_transforms.shape[0])):
    if active_mask[i]:
        pos = positions[i].cpu().numpy()
        print(f"  piece_{i}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

# Re-check piece_0 tree after reset
print("\nPiece 0 tree after reset:")
p0 = stage.GetPrimAtPath("/World/envs/env_0/Pieces/piece_0")
print_tree(p0)

# Step a few times and check if positions change
for i in range(5):
    action = torch.zeros((1, cfg.action_space), device="cuda:0")
    obs, _, _, _, _ = env.step(action)

all_transforms = env._piece_rigid_view.get_transforms()
positions = all_transforms[:, :3]
active_mask = positions[:, 2] > -50.0
print(f"\nAfter 5 steps, pieces with Z > -50: {active_mask.sum().item()}")
for i in range(min(32, all_transforms.shape[0])):
    if active_mask[i]:
        pos = positions[i].cpu().numpy()
        print(f"  piece_{i}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

env.close()
simulation_app.close()
