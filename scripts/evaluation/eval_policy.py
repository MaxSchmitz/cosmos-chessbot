#!/usr/bin/env python3
"""Evaluate a trained chess pick-and-place policy and record video.

Loads a checkpoint, runs the policy deterministically (using the action
mean, no sampling), and records overhead camera frames to an MP4 video.

Usage (inside Isaac Sim container):
    # Record 3 episodes from a checkpoint
    /isaac-sim/python.sh scripts/eval_policy.py \
        --checkpoint outputs/checkpoints/policy_step_96000.pt \
        --episodes 3

    # Record from latest checkpoint
    /isaac-sim/python.sh scripts/eval_policy.py \
        --checkpoint outputs/checkpoints/policy_final.pt \
        --episodes 5 --output outputs/eval_video.mp4
"""

import argparse
import sys
import traceback
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate chess RL policy and record video")
parser.add_argument("--checkpoint", type=Path, required=True,
                    help="Path to policy checkpoint .pt file")
parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to record")
parser.add_argument("--output", type=Path, default=None,
                    help="Output video path (default: outputs/eval_<step>.mp4)")
parser.add_argument("--fps", type=int, default=30, help="Video FPS")
parser.add_argument("--warmup-steps", type=int, default=50,
                    help="Random-action steps for obs normalizer warmup")
parser.add_argument("--stochastic", action="store_true",
                    help="Use stochastic actions (sample from policy) instead of deterministic mean")
parser.add_argument("--random", action="store_true",
                    help="Use uniform random actions (ignores policy, for visualization)")
parser.add_argument("--num-pieces", type=int, default=32,
                    help="Pieces per env (32 for full board visualization)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch

from cosmos_chessbot.isaac.chess_env import ChessPickPlaceEnv
from cosmos_chessbot.isaac.chess_env_cfg import ChessPickPlaceEnvCfg
from cosmos_chessbot.isaac.policy_model import ActorCritic, RunningMeanStd


# --------------------------------------------------------------------------- #
# Debug marker drawing (2D overlay on captured frames)
# --------------------------------------------------------------------------- #

def _quat_wxyz_to_rotmat(q):
    """Convert (w,x,y,z) quaternion to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def _project(points_w, cam_pos, cam_R, K, img_h, flipped=True):
    """Project (N,3) world points → (N,2) pixel coords.

    cam_R is the camera-to-world rotation (3×3).
    K is the 3×3 intrinsic matrix.
    If flipped=True, accounts for the np.flipud applied to the frame.
    """
    p_cam = (cam_R.T @ (points_w - cam_pos).T).T          # (N, 3)
    z = np.clip(p_cam[:, 2:3], 1e-4, None)
    p_norm = p_cam[:, :2] / z                              # (N, 2)
    px = K[0, 0] * p_norm[:, 0] + K[0, 2]
    py = K[1, 1] * p_norm[:, 1] + K[1, 2]
    if flipped:
        py = (img_h - 1) - py
    return np.stack([px, py], axis=-1)                     # (N, 2)


def _draw_line(frame, p1, p2, color, thickness=2):
    """Bresenham-ish line on a numpy HWC uint8 frame."""
    h, w = frame.shape[:2]
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    n = max(int(max(abs(x2 - x1), abs(y2 - y1))), 1)
    for i in range(n + 1):
        t = i / n
        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))
        for dx in range(-thickness // 2, thickness // 2 + 1):
            for dy in range(-thickness // 2, thickness // 2 + 1):
                xi, yi = x + dx, y + dy
                if 0 <= xi < w and 0 <= yi < h:
                    frame[yi, xi] = color


def _draw_circle(frame, center, color, radius=5):
    """Filled circle on a numpy HWC uint8 frame."""
    cx, cy = int(center[0]), int(center[1])
    h, w = frame.shape[:2]
    r2 = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= r2:
                xi, yi = cx + dx, cy + dy
                if 0 <= xi < w and 0 <= yi < h:
                    frame[yi, xi] = color


def draw_debug_markers(frame, env, camera):
    """Draw XYZ axis arrows at piece, jaw, and target positions.

    Colours:
      - Target piece : white centre dot
      - Jaw (grasp)  : yellow centre dot
      - Target square: cyan centre dot
    Each has R/G/B axis arrows (3 cm long).
    """
    try:
        piece_pos = env._target_piece_pos[0].cpu().numpy()
        target_pos = env._target_square_pos[0].cpu().numpy()
        ee_frame = env.scene["ee_frame"]
        jaw_pos = ee_frame.data.target_pos_w[0, 1, :].cpu().numpy()

        cam_pos = camera.data.pos_w[0].cpu().numpy()
        cam_quat = camera.data.quat_w_ros[0].cpu().numpy()  # (w,x,y,z) ROS convention
        cam_R = _quat_wxyz_to_rotmat(cam_quat)
        K = camera.data.intrinsic_matrices[0].cpu().numpy()  # (3,3)
        img_h = frame.shape[0]

        axis_len = 0.03  # 3 cm arrows
        axis_colors = [(220, 60, 60), (60, 220, 60), (60, 60, 220)]
        axis_vecs = [
            np.array([axis_len, 0, 0]),
            np.array([0, axis_len, 0]),
            np.array([0, 0, axis_len]),
        ]

        markers = [
            (piece_pos,  (255, 255, 255)),   # white = piece
            (jaw_pos,    (255, 255, 0)),      # yellow = jaw
            (target_pos, (0, 255, 255)),      # cyan = target square
        ]

        for world_pos, dot_color in markers:
            pts = np.array([world_pos] + [world_pos + v for v in axis_vecs])
            px = _project(pts, cam_pos, cam_R, K, img_h, flipped=True)
            center = px[0]
            for i, ax_col in enumerate(axis_colors):
                _draw_line(frame, center, px[i + 1], ax_col, thickness=2)
            _draw_circle(frame, center, dot_color, radius=4)
    except Exception as e:
        import traceback as _tb
        print(f"    DEBUG MARKER ERROR: {e}")
        _tb.print_exc()
        sys.stdout.flush()


def main():
    print("=" * 60)
    print("Chess Policy Evaluation + Video Recording")
    print("=" * 60)
    sys.stdout.flush()

    # Load checkpoint
    ckpt_path = args.checkpoint
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        simulation_app.close()
        return False

    device = torch.device("cuda:0")
    ckpt = torch.load(ckpt_path, map_location=device)
    step = ckpt.get("step", 0)
    print(f"\nCheckpoint: {ckpt_path}")
    print(f"  Training step: {step}")
    print(f"  Episodes trained: {ckpt.get('episode_count', '?')}")
    sys.stdout.flush()

    # Create environment (single env for clear visualization)
    cfg = ChessPickPlaceEnvCfg()
    cfg.scene.num_envs = 1
    cfg.num_pieces = args.num_pieces
    print(f"\nCreating environment (1 env)...")
    sys.stdout.flush()
    env = ChessPickPlaceEnv(cfg)

    # Load policy
    obs_dim = cfg.num_rl_observations
    act_dim = cfg.action_space
    policy = ActorCritic(obs_dim, act_dim).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    print(f"  Policy loaded ({sum(p.numel() for p in policy.parameters()):,} params)")
    sys.stdout.flush()

    # Set up obs normalizer
    obs_normalizer = RunningMeanStd(obs_dim, device)
    if "obs_normalizer" in ckpt:
        norm_state = ckpt["obs_normalizer"]
        obs_normalizer.mean = norm_state["mean"].to(device)
        obs_normalizer.var = norm_state["var"].to(device)
        obs_normalizer.count = norm_state["count"]
        print(f"  Obs normalizer loaded from checkpoint")
    else:
        print(f"  Warming up obs normalizer ({args.warmup_steps} random steps)...")
        sys.stdout.flush()
        obs, info = env.reset()
        for _ in range(args.warmup_steps):
            action = torch.rand((1, act_dim), device=device) * 2 - 1
            obs, _, _, _, _ = env.step(action)
            if "rl_obs" in obs.get("policy", {}):
                obs_normalizer.update(obs["policy"]["rl_obs"])
        print(f"  Normalizer warmed up (count={obs_normalizer.count:.0f})")
    sys.stdout.flush()

    # Set up video output
    output_path = args.output
    if output_path is None:
        output_dir = PROJECT_ROOT / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"eval_step_{step}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect frames in memory, write video at the end
    frames = []

    # Get camera sensor
    camera = env.scene["front"]

    # Warm up renderer — headless mode needs ~30+ steps before
    # the RTX renderer produces real (non-black) frames.
    print(f"\nWarming up renderer (50 steps)...")
    sys.stdout.flush()
    obs, info = env.reset()
    for i in range(50):
        action = torch.zeros((1, act_dim), device=device)
        obs, _, _, _, _ = env.step(action)

    # Debug: verify camera is producing real frames
    try:
        rgb_data = camera.data.output.get("rgb")
        if rgb_data is not None and isinstance(rgb_data, torch.Tensor):
            print(f"  Camera output shape: {rgb_data.shape}, dtype: {rgb_data.dtype}")
            print(f"  Frame stats: min={rgb_data.min().item():.3f}, max={rgb_data.max().item():.3f}, mean={rgb_data.mean().item():.3f}")
            sys.stdout.flush()
        else:
            print(f"  WARNING: Camera returned {type(rgb_data)}")
            sys.stdout.flush()
    except Exception as e:
        print(f"  WARNING: Camera debug failed: {e}")
        sys.stdout.flush()

    # Run evaluation episodes
    print(f"\nRunning {args.episodes} episodes (deterministic policy)...")
    sys.stdout.flush()
    total_rewards = []

    for ep in range(args.episodes):
        obs, info = env.reset()
        # Burn one sim step so the renderer clears stale positions
        obs, _, _, _, _ = env.step(torch.zeros((1, act_dim), device=device))
        ep_reward = 0.0
        step_count = 0

        target_pos = env._target_square_pos[0].cpu().numpy()
        piece_pos = env._target_piece_pos[0].cpu().numpy()
        print(f"\n  Episode {ep + 1}/{args.episodes}")
        print(f"    Piece at: ({piece_pos[0]:.3f}, {piece_pos[1]:.3f}, {piece_pos[2]:.3f})")
        print(f"    Target:   ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")

        num_active = env._piece_active_mask[0].sum().item()
        print(f"    Active pieces: {num_active}")
        sys.stdout.flush()

        while True:
            # Get RL observation and normalize
            rl_obs = obs["policy"]["rl_obs"]
            obs_norm = obs_normalizer.normalize(rl_obs)

            # Select action (tanh-squashed)
            with torch.no_grad():
                action_mean, action_std, _ = policy(obs_norm)
                if args.random:
                    action = torch.rand((1, act_dim), device=device) * 2 - 1
                elif args.stochastic:
                    from torch.distributions import Normal
                    dist = Normal(action_mean, action_std)
                    action = torch.tanh(dist.sample())
                else:
                    action = torch.tanh(action_mean)

            if step_count == 1:
                print(f"    Action mean: [{', '.join(f'{x:.3f}' for x in action_mean[0].tolist())}]")
                print(f"    Action std:  [{', '.join(f'{x:.3f}' for x in action_std[0].tolist())}]")
                print(f"    Action used: [{', '.join(f'{x:.3f}' for x in action[0].tolist())}]")
                sys.stdout.flush()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward.item() if isinstance(reward, torch.Tensor) else reward
            step_count += 1

            # Capture camera frame
            try:
                rgb_data = camera.data.output.get("rgb")
                if rgb_data is not None:
                    if isinstance(rgb_data, torch.Tensor) and rgb_data.numel() > 0:
                        frame = rgb_data[0].cpu().numpy()
                    elif isinstance(rgb_data, np.ndarray) and rgb_data.size > 0:
                        frame = rgb_data[0] if rgb_data.ndim == 4 else rgb_data
                    else:
                        frame = None

                    if frame is not None:
                        if frame.dtype != np.uint8:
                            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                        if frame.ndim == 3 and frame.shape[-1] == 4:
                            frame = frame[:, :, :3]
                        frame = np.flipud(frame)  # headless RTX needs Y-flip
                        # Draw debug markers: piece (white), jaw (yellow), target (cyan)
                        frame = frame.copy()  # make writable
                        draw_debug_markers(frame, env, camera)
                        if step_count == 1:
                            print(f"    First frame: shape={frame.shape}, min={frame.min()}, max={frame.max()}")
                            sys.stdout.flush()
                        frames.append(frame)
            except Exception as e:
                if step_count == 1:
                    print(f"    WARNING: Frame capture failed: {e}")
                    traceback.print_exc()
                    sys.stdout.flush()

            done = False
            if isinstance(terminated, torch.Tensor) and terminated.any():
                print(f"    Terminated at step {step_count}")
                done = True
            if isinstance(truncated, torch.Tensor) and truncated.any():
                print(f"    Truncated at step {step_count}")
                done = True

            if done:
                break

        total_rewards.append(ep_reward)
        grasped = env._has_been_grasped[0].item()
        print(f"    Steps: {step_count}, Reward: {ep_reward:.2f}, Grasped: {grasped}")
        sys.stdout.flush()

    # Write video
    if frames:
        print(f"\nWriting video ({len(frames)} frames)...")
        sys.stdout.flush()
        import imageio
        writer = imageio.get_writer(str(output_path), fps=args.fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"  Video saved: {output_path}")
    else:
        print(f"\n  WARNING: No frames captured — video not created")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Evaluation Summary")
    print(f"  Episodes: {args.episodes}")
    print(f"  Mean reward: {sum(total_rewards) / len(total_rewards):.2f}")
    if frames:
        print(f"  Video: {output_path} ({len(frames)} frames)")
    print(f"{'=' * 60}")
    sys.stdout.flush()

    env.close()
    simulation_app.close()
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
