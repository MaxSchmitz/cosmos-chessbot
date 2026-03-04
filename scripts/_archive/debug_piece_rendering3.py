#!/usr/bin/env python3
"""Debug: test piece rendering at different positions and scales."""
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
from pxr import UsdGeom, Gf, UsdPhysics, Sdf

cfg = ChessPickPlaceEnvCfg()
cfg.scene.num_envs = 1
env = ChessPickPlaceEnv(cfg)

stage = omni.usd.get_context().get_stage()
usd_dir = PROJECT_ROOT / "data" / "usd"
board_scale = env._board_scale
print(f"Board scale: {board_scale}")

# Check what the board's actual world extent is
board = stage.GetPrimAtPath("/World/envs/env_0/Board")
bxf = UsdGeom.Xformable(board)
for op in bxf.GetOrderedXformOps():
    print(f"Board xformOp: {op.GetOpName()} = {op.Get()}")

# Test A: Bare pawn at board center, same scale as board
path_a = "/World/envs/env_0/test_center"
prim_a = stage.DefinePrim(path_a, "Xform")
pawn_usd = usd_dir / "pawn_w.usd"
prim_a.GetReferences().AddReference(str(pawn_usd))
xf_a = UsdGeom.Xformable(prim_a)
xf_a.ClearXformOpOrder()
xf_a.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.2, 0.75))  # board center
xf_a.AddScaleOp().Set(Gf.Vec3f(board_scale, board_scale, board_scale))
print(f"Test A: center of board, scale={board_scale:.4f}")

# Test B: Bare pawn at board center, 5x scale
path_b = "/World/envs/env_0/test_big"
prim_b = stage.DefinePrim(path_b, "Xform")
prim_b.GetReferences().AddReference(str(pawn_usd))
xf_b = UsdGeom.Xformable(prim_b)
xf_b.ClearXformOpOrder()
xf_b.AddTranslateOp().Set(Gf.Vec3d(0.15, 0.2, 0.75))  # offset right
xf_b.AddScaleOp().Set(Gf.Vec3f(board_scale * 5, board_scale * 5, board_scale * 5))
print(f"Test B: right of center, scale={board_scale * 5:.4f}")

# Test C: Bare pawn, NO scale at all (raw USD size)
path_c = "/World/envs/env_0/test_raw"
prim_c = stage.DefinePrim(path_c, "Xform")
prim_c.GetReferences().AddReference(str(pawn_usd))
xf_c = UsdGeom.Xformable(prim_c)
xf_c.ClearXformOpOrder()
xf_c.AddTranslateOp().Set(Gf.Vec3d(-0.15, 0.2, 0.75))
print(f"Test C: raw USD scale (no scale op)")

# Test D: Use the BOARD usd as a test (we know this renders)
path_d = "/World/envs/env_0/test_board2"
prim_d = stage.DefinePrim(path_d, "Xform")
board_usd = usd_dir / "board.usd"
prim_d.GetReferences().AddReference(str(board_usd))
xf_d = UsdGeom.Xformable(prim_d)
xf_d.ClearXformOpOrder()
xf_d.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.2, 0.85))  # above the board
xf_d.AddScaleOp().Set(Gf.Vec3f(board_scale * 0.3, board_scale * 0.3, board_scale * 0.3))
print(f"Test D: small board above the main board")

# Check pawn USD structure directly
print(f"\nPawn USD path: {pawn_usd}")
print(f"Pawn USD exists: {pawn_usd.exists()}")
pawn_layer = Sdf.Layer.FindOrOpen(str(pawn_usd))
if pawn_layer:
    root_path = pawn_layer.defaultPrim
    print(f"Pawn USD defaultPrim: '{root_path}'")
    # List top-level prims
    for p in pawn_layer.rootPrims:
        print(f"  Root prim: {p.name} (type={p.typeName})")
else:
    print("ERROR: Could not open pawn USD")

# Reset and render
obs, info = env.reset()
for i in range(60):
    action = torch.zeros((1, cfg.action_space), device="cuda:0")
    obs, _, _, _, _ = env.step(action)

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
    imageio.imwrite("outputs/debug_rendering3.png", frame)
    print(f"\nFrame saved: outputs/debug_rendering3.png")
    print(f"Frame shape: {frame.shape}, min={frame.min()}, max={frame.max()}")

env.close()
simulation_app.close()
