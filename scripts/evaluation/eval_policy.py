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
PROJECT_ROOT = Path(__file__).parent.parent
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
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
import torch.nn as nn

from cosmos_chessbot.isaac.chess_env import ChessPickPlaceEnv
from cosmos_chessbot.isaac.chess_env_cfg import ChessPickPlaceEnvCfg


# -- Model definitions (must match training script) -------------------------

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ELU(),
        )
        self.actor_branch = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
        )
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic_branch = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
        )
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        shared_features = self.shared(obs)
        actor_features = self.actor_branch(shared_features)
        action_mean = self.actor_mean(actor_features)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        critic_input = shared_features + 0.5 * (shared_features.detach() - shared_features)
        critic_features = self.critic_branch(critic_input)
        value = self.critic_head(critic_features).squeeze(-1)
        return action_mean, action_std, value


class RunningMeanStd:
    def __init__(self, shape, device, epsilon=1e-4):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def update(self, batch: torch.Tensor):
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        batch_count = batch.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.mean) / (self.var.sqrt() + 1e-8)


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

    # Warm up renderer (first few frames may be blank)
    obs, info = env.reset()
    for _ in range(10):
        action = torch.zeros((1, act_dim), device=device)
        obs, _, _, _, _ = env.step(action)

    # Run evaluation episodes
    print(f"\nRunning {args.episodes} episodes (deterministic policy)...")
    sys.stdout.flush()
    total_rewards = []

    for ep in range(args.episodes):
        obs, info = env.reset()
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

            # Deterministic action (use mean, no sampling)
            with torch.no_grad():
                action_mean, _, _ = policy(obs_norm)
                action = action_mean.clamp(-1.0, 1.0)

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
                        frame = np.flipud(frame)  # fix OpenGL framebuffer Y-flip
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
