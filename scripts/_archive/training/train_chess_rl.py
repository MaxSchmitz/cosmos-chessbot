#!/usr/bin/env python3
"""Training script for ChessPickPlaceEnv using PPO.

Trains an RL agent with Proximal Policy Optimization to pick and place
chess pieces using an SO-101 robot arm in Isaac Sim.

Usage (inside Isaac Sim container):
    # Short training run
    /isaac-sim/python.sh scripts/train_chess_rl.py --num-steps 10000

    # Full training run
    /isaac-sim/python.sh scripts/train_chess_rl.py --num-steps 100000

    # With wandb logging
    /isaac-sim/python.sh scripts/train_chess_rl.py --wandb --project chess-rl
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Parse args and launch Isaac Sim via AppLauncher
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train chess pick-and-place RL agent")
parser.add_argument("--num-steps", type=int, default=10000, help="Total env steps")
parser.add_argument("--num-envs", type=int, default=1024, help="Number of parallel envs")
parser.add_argument("--num-pieces", type=int, default=1,
                    help="Pieces per env (1 for fast training, 32 for full board)")
parser.add_argument("--rollout-length", type=int, default=48,
                    help="Steps per rollout before PPO update")
parser.add_argument("--ppo-epochs", type=int, default=4,
                    help="PPO optimization epochs per rollout")
parser.add_argument("--mini-batch-size", type=int, default=512,
                    help="Mini-batch size for PPO updates")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
parser.add_argument("--entropy-coeff", type=float, default=0.001,
                    help="Entropy bonus coefficient")
parser.add_argument("--value-coeff", type=float, default=0.5,
                    help="Value loss coefficient")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="Max gradient norm for clipping")
parser.add_argument("--checkpoint-dir", type=Path,
                    default=PROJECT_ROOT / "outputs" / "checkpoints",
                    help="Checkpoint save directory")
parser.add_argument("--checkpoint-interval", type=int, default=5000,
                    help="Steps between checkpoints")
parser.add_argument("--log-interval", type=int, default=1,
                    help="Rollouts between log prints (1 = every rollout)")
parser.add_argument("--early-stop-return", type=float, default=None,
                    help="Stop training when mean return exceeds this value")
parser.add_argument("--early-stop-patience", type=int, default=15,
                    help="Rollouts without improvement before early stopping")
parser.add_argument("--save-plot", action="store_true", default=True,
                    help="Save training curve plot to checkpoint dir")
parser.add_argument("--resume", type=Path, default=None,
                    help="Path to checkpoint .pt file to resume training from")
parser.add_argument("--wandb", action="store_true", help="Log to wandb")
parser.add_argument("--project", type=str, default="cosmos-chess-rl",
                    help="wandb project name")
# Reason2 critic
parser.add_argument("--reason2-critic", action="store_true",
                    help="Enable Cosmos Reason2 episode critic")
parser.add_argument("--cosmos-server", type=str, default="http://localhost:8000",
                    help="URL of the Cosmos inference server")
parser.add_argument("--critic-weight", type=float, default=10.0,
                    help="Reward scale for Reason2 critic scores")
parser.add_argument("--critic-frequency", type=float, default=0.1,
                    help="Fraction of episodes to evaluate with Reason2 (0.0-1.0)")
parser.add_argument("--critic-render-interval", type=int, default=5,
                    help="Capture camera frame every N control steps")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = args.reason2_critic  # cameras only needed for Reason2 critic

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import env/torch modules (after SimulationApp init)
import torch
import torch.nn as nn

from cosmos_chessbot.isaac.chess_env import ChessPickPlaceEnv
from cosmos_chessbot.isaac.chess_env_cfg import ChessPickPlaceEnvCfg
from cosmos_chessbot.isaac.policy_model import ActorCritic, RunningMeanStd
from cosmos_chessbot.isaac.reason2_critic import Reason2Critic


def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: (T, N) per-step rewards
        values: (T, N) per-step value estimates
        dones: (T, N) episode termination flags
        next_value: (N,) value estimate for the state after the last step
        gamma: discount factor
        gae_lambda: GAE lambda

    Returns:
        advantages: (T, N)
        returns: (T, N)
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        next_non_terminal = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def main():
    print("=" * 60)
    print("Chess Pick-and-Place PPO Training")
    print("=" * 60)

    # -- Setup ---------------------------------------------------------------
    # Clean stale HDF5 recorder file (LeIsaac writes episode data here;
    # it grows unbounded and can fill the disk on long runs).
    hdf5_path = "/tmp/isaaclab/logs/dataset.hdf5"
    if os.path.exists(hdf5_path):
        os.remove(hdf5_path)
        print(f"Cleaned stale {hdf5_path}")

    cfg = ChessPickPlaceEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.num_pieces = args.num_pieces
    cfg.enable_camera = args.reason2_critic  # disable camera to save VRAM
    N = args.num_envs
    T = args.rollout_length

    print(f"\nConfig:")
    print(f"  num_envs:        {N}")
    print(f"  num_pieces:      {cfg.num_pieces}")
    print(f"  action_space:    {cfg.action_space}")
    print(f"  rl_obs_dim:      {cfg.num_rl_observations}")
    print(f"  episode_length:  {cfg.episode_length_s}s")
    print(f"  rollout_length:  {T}")
    print(f"  ppo_epochs:      {args.ppo_epochs}")
    print(f"  mini_batch_size: {args.mini_batch_size}")
    print(f"  lr:              {args.lr}")
    print(f"  total_steps:     {args.num_steps}")

    # Create environment
    print("\nCreating environment...")
    env = ChessPickPlaceEnv(cfg)
    device = env.device

    # Create actor-critic policy
    obs_dim = cfg.num_rl_observations
    act_dim = cfg.action_space
    policy = ActorCritic(obs_dim, act_dim).to(device)
    obs_normalizer = RunningMeanStd(obs_dim, device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  Policy parameters: {total_params:,}")

    # Resume from checkpoint if specified
    resume_step = 0
    resume_episodes = 0
    resume_returns = []
    if args.resume:
        print(f"\n  Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        policy.load_state_dict(ckpt["policy_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        resume_step = ckpt.get("step", 0)
        resume_episodes = ckpt.get("episode_count", 0)
        resume_returns = ckpt.get("ep_returns_history", [])
        print(f"  Resumed at step {resume_step}, episodes {resume_episodes}")

    # Checkpoint directory
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Wandb
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.project,
                config={
                    "num_envs": N,
                    "num_steps": args.num_steps,
                    "rollout_length": T,
                    "ppo_epochs": args.ppo_epochs,
                    "mini_batch_size": args.mini_batch_size,
                    "lr": args.lr,
                    "gamma": args.gamma,
                    "gae_lambda": args.gae_lambda,
                    "clip_eps": args.clip_eps,
                    "entropy_coeff": args.entropy_coeff,
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                    "episode_length_s": cfg.episode_length_s,
                },
            )
            print(f"  wandb run: {wandb_run.url}")
        except ImportError:
            print("  WARNING: wandb not installed, skipping logging")

    # -- Reason2 critic -------------------------------------------------------
    critic = None
    if args.reason2_critic:
        print(f"\n  Reason2 critic enabled:")
        print(f"    server:          {args.cosmos_server}")
        print(f"    weight:          {args.critic_weight}")
        print(f"    frequency:       {args.critic_frequency}")
        print(f"    render_interval: {args.critic_render_interval}")
        critic = Reason2Critic(
            server_url=args.cosmos_server,
            weight=args.critic_weight,
            frequency=args.critic_frequency,
            render_interval=args.critic_render_interval,
            num_envs=N,
        )

    # -- Rollout storage -----------------------------------------------------
    # Pre-allocate tensors for rollout collection
    buf_obs = torch.zeros((T, N, obs_dim), device=device)
    buf_actions = torch.zeros((T, N, act_dim), device=device)
    buf_log_probs = torch.zeros((T, N), device=device)
    buf_rewards = torch.zeros((T, N), device=device)
    buf_dones = torch.zeros((T, N), device=device)
    buf_values = torch.zeros((T, N), device=device)

    # -- Training loop -------------------------------------------------------
    print(f"\nStarting PPO training...")
    obs, info = env.reset()

    global_step = resume_step
    rollout_count = 0
    episode_count = resume_episodes
    episode_return_sum = 0.0
    episode_return_count = 0
    ep_returns_history = list(resume_returns)
    start_time = time.time()

    # For plotting and early stopping
    plot_steps = []
    plot_returns = []
    plot_vloss = []
    plot_entropy = []
    best_mean_return = float("-inf")
    no_improve_count = 0

    # Track per-env episode returns
    env_ep_return = torch.zeros(N, device=device)

    while global_step < args.num_steps:
        rollout_count += 1

        # -- Collect rollout -------------------------------------------------
        policy.eval()
        for t in range(T):
            obs_tensor = obs["policy"].get(
                "rl_obs",
                torch.zeros((N, obs_dim), device=device),
            )
            # Normalize observations
            obs_normalizer.update(obs_tensor)
            obs_tensor = obs_normalizer.normalize(obs_tensor)

            with torch.no_grad():
                action, log_prob, value = policy.get_action_and_value(obs_tensor)

            # Store normalized obs in buffers
            buf_obs[t] = obs_tensor
            buf_actions[t] = action
            buf_log_probs[t] = log_prob
            buf_values[t] = value

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            buf_rewards[t] = reward
            done = terminated | truncated
            buf_dones[t] = done.float()

            # Capture camera frames for Reason2 critic
            if critic is not None:
                camera_rgb = env.get_camera_rgb()
                if camera_rgb is not None:
                    critic.capture_frame(camera_rgb, t)

            # Track per-env episode returns
            env_ep_return += reward
            for i in range(N):
                if done[i]:
                    episode_count += 1
                    ep_ret = env_ep_return[i].item()
                    ep_returns_history.append(ep_ret)
                    episode_return_sum += ep_ret
                    episode_return_count += 1
                    env_ep_return[i] = 0.0

                    # Notify critic of episode completion
                    if critic is not None:
                        critic.on_episode_done(
                            env_idx=i,
                            terminal_step_idx=t * N + i,
                        )

            global_step += N

        # -- Compute advantages (GAE) ---------------------------------------
        with torch.no_grad():
            next_obs_tensor = obs["policy"].get(
                "rl_obs",
                torch.zeros((N, obs_dim), device=device),
            )
            next_obs_tensor = obs_normalizer.normalize(next_obs_tensor)
            _, _, next_value = policy(next_obs_tensor)

        # -- Reason2 critic: evaluate pending episodes and inject rewards -----
        if critic is not None and critic.pending_count > 0:
            critic_results = critic.evaluate_pending()
            for step_idx, critic_reward, critique in critic_results:
                # step_idx encodes (t * N + env_i)
                t_idx = step_idx // N
                env_i = step_idx % N
                if 0 <= t_idx < T and 0 <= env_i < N:
                    buf_rewards[t_idx, env_i] += critic_reward

        advantages, returns = compute_gae(
            buf_rewards, buf_values, buf_dones,
            next_value, args.gamma, args.gae_lambda,
        )

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # -- PPO update ------------------------------------------------------
        policy.train()

        # Flatten rollout: (T, N, ...) -> (T*N, ...)
        flat_obs = buf_obs.reshape(-1, obs_dim)
        flat_actions = buf_actions.reshape(-1, act_dim)
        flat_log_probs = buf_log_probs.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_values = buf_values.reshape(-1)

        total_samples = T * N
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(args.ppo_epochs):
            # Shuffle indices
            indices = torch.randperm(total_samples, device=device)

            for start in range(0, total_samples, args.mini_batch_size):
                end = min(start + args.mini_batch_size, total_samples)
                mb_idx = indices[start:end]

                mb_obs = flat_obs[mb_idx]
                mb_actions = flat_actions[mb_idx]
                mb_old_log_probs = flat_log_probs[mb_idx]
                mb_returns = flat_returns[mb_idx]
                mb_advantages = flat_advantages[mb_idx]
                mb_old_values = flat_values[mb_idx]

                # Evaluate actions under current policy
                new_log_probs, entropy, new_values = policy.evaluate_actions(
                    mb_obs, mb_actions
                )

                # Policy loss (clipped surrogate)
                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = ratio.clamp(
                    1.0 - args.clip_eps, 1.0 + args.clip_eps
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped to prevent large critic updates)
                v_clipped = mb_old_values + (new_values - mb_old_values).clamp(
                    -args.clip_eps, args.clip_eps
                )
                vl_unclipped = (new_values - mb_returns).pow(2)
                vl_clipped = (v_clipped - mb_returns).pow(2)
                value_loss = 0.5 * torch.max(vl_unclipped, vl_clipped).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + args.value_coeff * value_loss
                    + args.entropy_coeff * entropy_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                n_updates += 1

        avg_policy_loss = total_policy_loss / max(1, n_updates)
        avg_value_loss = total_value_loss / max(1, n_updates)
        avg_entropy = total_entropy / max(1, n_updates)

        # -- Logging ---------------------------------------------------------
        if rollout_count % args.log_interval == 0:
            elapsed = time.time() - start_time
            fps = global_step / elapsed
            mean_ep_return = (
                sum(ep_returns_history[-20:]) / max(1, len(ep_returns_history[-20:]))
            )

            critic_info = ""
            if critic is not None and critic.total_critiques > 0:
                critic_info = f" | Critic score: {critic.mean_score:.1f}/10 ({critic.total_critiques} evals)"

            print(
                f"  Step {global_step:>7d}/{args.num_steps} | "
                f"FPS: {fps:.0f} | "
                f"Episodes: {episode_count} | "
                f"Mean return (last 20): {mean_ep_return:>8.2f} | "
                f"P_loss: {avg_policy_loss:.4f} | "
                f"V_loss: {avg_value_loss:.4f} | "
                f"Entropy: {avg_entropy:.4f}"
                f"{critic_info}"
            )

            log_data = {
                "step": global_step,
                "fps": fps,
                "episode_count": episode_count,
                "mean_episode_return": mean_ep_return,
                "policy_loss": avg_policy_loss,
                "value_loss": avg_value_loss,
                "entropy": avg_entropy,
            }
            if critic is not None and critic.total_critiques > 0:
                log_data["critic/mean_score"] = critic.mean_score
                log_data["critic/total_critiques"] = critic.total_critiques
                for issue, count in critic.issue_counts.items():
                    log_data[f"critic/issues/{issue}"] = count

            if wandb_run:
                wandb_run.log(log_data, step=global_step)

        # -- Track for plotting and early stopping ---------------------------
        plot_steps.append(global_step)
        plot_returns.append(mean_ep_return if ep_returns_history else 0.0)
        plot_vloss.append(avg_value_loss)
        plot_entropy.append(avg_entropy)

        # Early stopping: target return reached
        if args.early_stop_return is not None:
            if mean_ep_return >= args.early_stop_return and len(ep_returns_history) >= 20:
                print(f"\n  Early stop: mean return {mean_ep_return:.2f} >= {args.early_stop_return}")
                break

        # Early stopping: no improvement for N rollouts
        if len(ep_returns_history) >= 20:
            if mean_ep_return > best_mean_return:
                best_mean_return = mean_ep_return
                no_improve_count = 0
            else:
                no_improve_count += 1
            if args.early_stop_patience and no_improve_count >= args.early_stop_patience:
                print(f"\n  Early stop: no improvement for {no_improve_count} rollouts "
                      f"(best: {best_mean_return:.2f})")
                break

        # -- Checkpointing ---------------------------------------------------
        if global_step % args.checkpoint_interval < T * N:
            ckpt_path = args.checkpoint_dir / f"policy_step_{global_step}.pt"
            torch.save(
                {
                    "step": global_step,
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "episode_count": episode_count,
                    "ep_returns_history": ep_returns_history[-100:],
                    "obs_normalizer": {
                        "mean": obs_normalizer.mean,
                        "var": obs_normalizer.var,
                        "count": obs_normalizer.count,
                    },
                },
                ckpt_path,
            )
            print(f"  Checkpoint saved: {ckpt_path}")

    # -- Save training curve plot --------------------------------------------
    if args.save_plot and plot_steps:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

            axes[0].plot(plot_steps, plot_returns, "b-", linewidth=1.5)
            axes[0].set_ylabel("Mean Return (last 20)")
            axes[0].set_title("Chess Pick-and-Place PPO Training")
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(plot_steps, plot_vloss, "r-", linewidth=1.5)
            axes[1].set_ylabel("Value Loss")
            axes[1].grid(True, alpha=0.3)

            axes[2].plot(plot_steps, plot_entropy, "g-", linewidth=1.5)
            axes[2].set_ylabel("Entropy")
            axes[2].set_xlabel("Environment Steps")
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = args.checkpoint_dir / "training_curve.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"Training curve saved: {plot_path}")
        except ImportError:
            print("  matplotlib not installed, skipping plot")

    # -- Final summary -------------------------------------------------------
    elapsed = time.time() - start_time
    mean_ep_return = (
        sum(ep_returns_history[-20:]) / max(1, len(ep_returns_history[-20:]))
    )
    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"  Total steps:    {global_step}")
    print(f"  Total episodes: {episode_count}")
    print(f"  Mean return (last 20): {mean_ep_return:.2f}")
    print(f"  Wall time:      {elapsed:.1f}s")
    print(f"  FPS:            {global_step / elapsed:.0f}")
    print(f"{'=' * 60}")

    # Save final checkpoint
    final_path = args.checkpoint_dir / "policy_final.pt"
    torch.save(
        {
            "step": global_step,
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode_count": episode_count,
            "ep_returns_history": ep_returns_history[-100:],
            "obs_normalizer": {
                "mean": obs_normalizer.mean,
                "var": obs_normalizer.var,
                "count": obs_normalizer.count,
            },
        },
        final_path,
    )
    print(f"Final checkpoint: {final_path}")

    if wandb_run:
        wandb_run.finish()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
