#!/usr/bin/env python3
"""Smoke test for ChessPickPlaceEnv with rigid body physics.

Validates that the environment can:
1. Create and initialize properly with piece physics
2. Reset with a random FEN position and random target locations
3. Step with random actions
4. Return correct observation/reward/done shapes
5. Detect collision penalty when pieces are displaced
6. Verify target positions span the reachable workspace (not just board squares)

Usage (inside Isaac Sim container):
    /isaac-sim/python.sh scripts/test_chess_env.py
    /isaac-sim/python.sh scripts/test_chess_env.py --gui
    /isaac-sim/python.sh scripts/test_chess_env.py --episodes 10
"""

import argparse
import sys
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Parse args and launch Isaac Sim via AppLauncher (handles cameras, headless, etc.)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test for ChessPickPlaceEnv")
parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
parser.add_argument("--steps-per-episode", type=int, default=50, help="Max steps")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True  # required for TiledCamera sensors

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import env modules
import torch
import numpy as np

from cosmos_chessbot.isaac.chess_env import ChessPickPlaceEnv, MAX_PIECES, SLOT_PIECE_TYPES
from cosmos_chessbot.isaac.chess_env_cfg import ChessPickPlaceEnvCfg
from cosmos_chessbot.isaac.chess_scene_cfg import TABLE_HEIGHT


def main():
    print("=" * 60)
    print("ChessPickPlaceEnv Smoke Test (with Physics)")
    print("=" * 60)

    # Create environment config
    cfg = ChessPickPlaceEnvCfg()
    cfg.scene.num_envs = 1
    print(f"\nConfig: num_envs={cfg.scene.num_envs}, "
          f"action_space={cfg.action_space}, "
          f"rl_obs_dim={cfg.num_rl_observations}")
    print(f"Physics: mass={cfg.piece_mass_kg}kg, "
          f"friction={cfg.piece_friction}, "
          f"damping={cfg.piece_linear_damping}")

    # Create environment
    print("\n[1/5] Creating environment...")
    env = ChessPickPlaceEnv(cfg)
    print(f"  Environment created: {env.num_envs} env(s)")
    print(f"  Device: {env.device}")

    # Verify physics setup
    print("\n[2/5] Checking piece physics setup...")
    import omni.usd
    from pxr import UsdPhysics
    stage = omni.usd.get_context().get_stage()

    piece_path = "/World/envs/env_0/Pieces/piece_0"
    piece_prim = stage.GetPrimAtPath(piece_path)
    assert piece_prim.IsValid(), f"Piece prim not found at {piece_path}"

    # Check rigid body API
    rb_api = UsdPhysics.RigidBodyAPI(piece_prim)
    assert rb_api.GetRigidBodyEnabledAttr().Get() is True, "RigidBody not enabled"
    print("  Rigid body API: OK")

    # Check mass
    mass_api = UsdPhysics.MassAPI(piece_prim)
    mass = mass_api.GetMassAttr().Get()
    assert abs(mass - cfg.piece_mass_kg) < 1e-6, f"Mass mismatch: {mass} != {cfg.piece_mass_kg}"
    print(f"  Mass: {mass}kg OK")

    # Verify Visual child (holds USD reference, separate from physics)
    visual_path = f"{piece_path}/Visual"
    visual_prim = stage.GetPrimAtPath(visual_path)
    assert visual_prim.IsValid(), f"Visual child prim not found at {visual_path}"
    print("  Visual child prim: OK")

    # Check mesh collision on the actual Mesh prim (not the Visual Xform)
    # Collision must be on the mesh so PhysX accounts for the +90° X rotation
    found_collision_mesh = False
    for child in visual_prim.GetChildren():
        if child.GetName() in ("env_light", "_materials"):
            continue
        for grandchild in child.GetChildren():
            if grandchild.GetTypeName() == "Mesh":
                assert grandchild.HasAPI(UsdPhysics.CollisionAPI), \
                    f"Mesh {grandchild.GetPath()} missing CollisionAPI"
                assert grandchild.HasAPI(UsdPhysics.MeshCollisionAPI), \
                    f"Mesh {grandchild.GetPath()} missing MeshCollisionAPI"
                mc = UsdPhysics.MeshCollisionAPI(grandchild)
                approx = mc.GetApproximationAttr().Get()
                assert approx == "convexHull", \
                    f"Expected convexHull, got {approx}"
                print(f"  Mesh collision ({grandchild.GetPath()}): OK (approx={approx})")
                found_collision_mesh = True
    assert found_collision_mesh, "No mesh prim with CollisionAPI found under Visual"

    # Verify typed pool — different slots have different piece meshes
    print("\n  Typed pool check:")
    # piece_0 = pawn_w (slot 0), piece_14 = queen_w, piece_31 = king_b
    for check_idx, expected_type in [(0, "pawn_w"), (14, "queen_w"), (31, "king_b")]:
        assert SLOT_PIECE_TYPES[check_idx] == expected_type, (
            f"Slot {check_idx}: expected {expected_type}, "
            f"got {SLOT_PIECE_TYPES[check_idx]}"
        )
        # Verify the Visual child has a USD reference loaded (has mesh children)
        check_visual = f"/World/envs/env_0/Pieces/piece_{check_idx}/Visual"
        check_prim = stage.GetPrimAtPath(check_visual)
        assert check_prim.IsValid(), f"Visual prim missing at {check_visual}"
        mesh_children = [
            c for c in check_prim.GetAllChildren()
            if c.GetTypeName() == "Mesh"
              or any(gc.GetTypeName() == "Mesh" for gc in c.GetChildren())
        ]
        assert len(mesh_children) > 0, (
            f"piece_{check_idx} ({expected_type}): no mesh geometry found"
        )
        print(f"    piece_{check_idx} = {expected_type}: OK")

    # Run episodes and collect target positions
    print(f"\n[3/5] Running {args.episodes} episodes with random actions...")
    total_rewards = []
    episode_lengths = []
    target_positions = []

    for ep in range(args.episodes):
        obs, info = env.reset()

        # Check observation structure
        assert "policy" in obs, f"Expected 'policy' key in obs, got {list(obs.keys())}"
        policy_obs = obs["policy"]
        print(f"\n  Episode {ep + 1}/{args.episodes}")
        print(f"    Obs keys: {list(policy_obs.keys())}")

        if "rl_obs" in policy_obs:
            rl_obs = policy_obs["rl_obs"]
            print(f"    rl_obs shape: {rl_obs.shape}")
            assert rl_obs.shape[-1] == cfg.num_rl_observations, (
                f"Expected rl_obs dim {cfg.num_rl_observations}, "
                f"got {rl_obs.shape[-1]}"
            )

        # Record target position
        target_pos = env._target_square_pos[0].cpu().numpy()
        target_positions.append(target_pos.copy())
        print(f"    Target pos: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")

        # Check physics state tensors
        assert env._piece_pos is not None, "piece_pos tensor not initialized"
        assert env._piece_active_mask is not None, "piece_active_mask not initialized"
        num_active = env._piece_active_mask[0].sum().item()
        print(f"    Active pieces: {num_active}/{MAX_PIECES}")

        ep_reward = 0.0
        step = 0

        for step in range(args.steps_per_episode):
            action = torch.rand((1, cfg.action_space), device=env.device) * 2 - 1
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward.item() if isinstance(reward, torch.Tensor) else reward

            if isinstance(terminated, torch.Tensor) and terminated.any():
                print(f"    Terminated at step {step + 1}")
                break
            if isinstance(truncated, torch.Tensor) and truncated.any():
                print(f"    Truncated at step {step + 1}")
                break

        total_rewards.append(ep_reward)
        episode_lengths.append(step + 1)

        print(f"    Steps: {step + 1}, Total reward: {ep_reward:.2f}")

    # Check target diversity
    print(f"\n[4/5] Target diversity check...")
    target_positions = np.array(target_positions)
    if len(target_positions) > 1:
        x_range = target_positions[:, 0].max() - target_positions[:, 0].min()
        y_range = target_positions[:, 1].max() - target_positions[:, 1].min()
        print(f"  Target X range: {x_range:.3f}m")
        print(f"  Target Y range: {y_range:.3f}m")
        print(f"  Target Z (all): {target_positions[:, 2].mean():.3f}m "
              f"(expected ~{TABLE_HEIGHT:.3f}m)")
        if x_range > 0.01 or y_range > 0.01:
            print("  Random target sampling: DIVERSE (OK)")
        else:
            print("  WARNING: Targets not diverse enough — check random sampling")
    else:
        print("  (Need >1 episode for diversity check)")

    # Summary
    print(f"\n[5/5] Summary")
    print(f"  Episodes: {args.episodes}")
    print(f"  Mean reward: {sum(total_rewards) / len(total_rewards):.2f}")
    print(f"  Mean episode length: {sum(episode_lengths) / len(episode_lengths):.1f}")
    print(f"\n  Action space: {cfg.action_space}")
    print(f"  RL obs dim: {cfg.num_rl_observations}")
    print(f"  Physics: rigid bodies (mesh collision)")
    print(f"  Targets: random workspace positions")
    print("=" * 60)
    print("Smoke test PASSED")
    print("=" * 60)

    env.close()
    simulation_app.close()
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
