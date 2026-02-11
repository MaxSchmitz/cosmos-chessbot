#!/usr/bin/env python3
"""Debug: test different piece rendering configurations."""
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

# Create 3 test pieces with different configurations:
# A: Bare reference, no physics (like the board)
# B: Reference + RigidBodyAPI, no collision
# C: Reference + RigidBodyAPI + collision on mesh

for label, test_name, add_physics, add_collision in [
    ("A", "test_bare", False, False),
    ("B", "test_phys", True, False),
    ("C", "test_phys_col", True, True),
]:
    path = f"/World/envs/env_0/{test_name}"
    prim = stage.DefinePrim(path, "Xform")
    pawn_usd = usd_dir / "pawn_w.usd"
    prim.GetReferences().AddReference(str(pawn_usd))

    if add_physics:
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        rb.CreateRigidBodyEnabledAttr(True)
        rb.CreateKinematicEnabledAttr(True)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(0.05)

    if add_collision:
        # Apply collision to mesh children
        for child in prim.GetChildren():
            if child.GetName() in ("env_light", "_materials"):
                continue
            for gc in child.GetChildren():
                if gc.GetTypeName() == "Mesh":
                    UsdPhysics.CollisionAPI.Apply(gc)
                    mc = UsdPhysics.MeshCollisionAPI.Apply(gc)
                    mc.CreateApproximationAttr("convexHull")

    # Zero internal translate
    for child in prim.GetChildren():
        if child.GetName() in ("env_light", "_materials"):
            continue
        child_xf = UsdGeom.Xformable(child)
        if child_xf:
            for op in child_xf.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
                    break

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    # Place them in a row next to the board
    x_pos = -0.3 + 0.15 * ["A", "B", "C"].index(label)
    xform.AddTranslateOp().Set(Gf.Vec3d(x_pos, 0.0, 0.75))
    xform.AddScaleOp().Set(Gf.Vec3f(board_scale, board_scale, board_scale))

    print(f"Test {label} ({test_name}): physics={add_physics}, collision={add_collision}, pos=({x_pos:.2f}, 0.0, 0.75)")

# Reset env and step to render
obs, info = env.reset()
for i in range(60):  # enough for renderer warmup
    action = torch.zeros((1, cfg.action_space), device="cuda:0")
    obs, _, _, _, _ = env.step(action)

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
    imageio.imwrite("outputs/debug_rendering_test.png", frame)
    print(f"\nFrame saved: outputs/debug_rendering_test.png")
    print(f"Frame shape: {frame.shape}, min={frame.min()}, max={frame.max()}")

env.close()
simulation_app.close()
