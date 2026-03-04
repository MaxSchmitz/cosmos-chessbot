#!/usr/bin/env python3
"""Deep debug: check if piece mesh data is present in runtime scene."""
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
from pxr import UsdGeom, Gf

cfg = ChessPickPlaceEnvCfg()
cfg.scene.num_envs = 1
env = ChessPickPlaceEnv(cfg)

stage = omni.usd.get_context().get_stage()

# Inspect TestPiece prim (king_b at 5x scale)
for path in ["/World/envs/env_0/Board",
             "/World/envs/env_0/TestBoardAsPiece",
             "/World/envs/env_0/TestPiece",
             "/World/envs/env_0/Pieces/piece_0"]:
    prim = stage.GetPrimAtPath(path)
    print(f"\n{'=' * 50}")
    print(f"PRIM: {path}")
    print(f"  valid={prim.IsValid()}, type={prim.GetTypeName()}, active={prim.IsActive()}")
    if not prim.IsValid():
        continue

    img = UsdGeom.Imageable(prim)
    print(f"  purpose={img.GetPurposeAttr().Get()}")
    print(f"  visibility={img.GetVisibilityAttr().Get()}")
    xf = UsdGeom.Xformable(prim)
    local_mtx = xf.ComputeLocalToWorldTransform(0)
    print(f"  localToWorld: {local_mtx}")
    ops = [(op.GetOpName(), str(op.Get())) for op in xf.GetOrderedXformOps()]
    print(f"  xformOps: {ops}")

    # Check boundable extent
    boundable = UsdGeom.Boundable(prim)
    if boundable:
        extent = boundable.GetExtentAttr().Get()
        print(f"  extent: {extent}")

    def walk(p, depth=1):
        prefix = "  " * depth
        for child in p.GetChildren():
            ctype = child.GetTypeName()
            active = child.IsActive()
            info = f"{prefix}{child.GetName()} (type={ctype}, active={active}"
            if ctype == "Mesh":
                mesh = UsdGeom.Mesh(child)
                pts = mesh.GetPointsAttr().Get()
                ext = mesh.GetExtentAttr().Get()
                info += f", points={len(pts) if pts else 0}, extent={ext}"
                # Check doubleSided
                ds = mesh.GetDoubleSidedAttr().Get()
                orient = mesh.GetOrientationAttr().Get()
                info += f", doubleSided={ds}, orient={orient}"
            info += ")"
            print(info)
            walk(child, depth + 1)
    walk(prim)

# Reset and render
obs, info = env.reset()
for i in range(60):
    action = torch.zeros((1, cfg.action_space), device="cuda:0")
    obs, _, _, _, _ = env.step(action)

# Re-check piece_0 position in world after reset
p0 = stage.GetPrimAtPath("/World/envs/env_0/Pieces/piece_0")
xf = UsdGeom.Xformable(p0)
print(f"\nAfter reset piece_0 localToWorld: {xf.ComputeLocalToWorldTransform(0)}")

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
    imageio.imwrite("outputs/debug_rendering5.png", frame)
    print(f"\nFrame saved: outputs/debug_rendering5.png")

env.close()
simulation_app.close()
