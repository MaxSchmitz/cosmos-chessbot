"""Cosmos Reason2 episode critic for RL training.

Captures episode video from the Isaac Sim camera, sends it to a remote
Cosmos-Reason2 server for holistic evaluation, and converts the structured
critique into a scalar reward signal.

Usage in the training loop:
    critic = Reason2Critic(cosmos_server_url, weight=10.0, frequency=0.1)
    # During rollout:
    critic.capture_frame(env, step)
    # On episode done:
    critic.on_episode_done(env_idx, from_sq, to_sq, piece_type)
    # Between rollout and PPO update:
    rewards = critic.evaluate_and_get_rewards()
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

from ..reasoning.remote_reasoning import RemoteChessGameReasoning
from ..reasoning.game_reasoning import EpisodeCritique

logger = logging.getLogger(__name__)


@dataclass
class EpisodeRecord:
    """Buffered episode data awaiting Reason2 evaluation."""
    env_idx: int
    frames: list[Image.Image]
    from_square: str
    to_square: str
    piece_type: str
    terminal_step_idx: int  # index into rollout buffer for reward injection


class Reason2Critic:
    """Async video critic that evaluates RL episodes with Cosmos Reason2.

    Captures camera frames during rollout, queues completed episodes,
    and evaluates them via the remote Cosmos server between PPO updates.
    """

    def __init__(
        self,
        server_url: str,
        weight: float = 10.0,
        frequency: float = 0.1,
        render_interval: int = 5,
        num_envs: int = 64,
        timeout: float = 120.0,
    ):
        """
        Args:
            server_url: URL of the Cosmos inference server
            weight: Reward scale for critic scores
            frequency: Fraction of episodes to evaluate (0.0-1.0)
            render_interval: Capture a frame every N control steps
            num_envs: Number of parallel environments
            timeout: HTTP timeout for Reason2 inference
        """
        self.weight = weight
        self.frequency = frequency
        self.render_interval = render_interval
        self.num_envs = num_envs

        self.reasoning = RemoteChessGameReasoning(server_url, timeout=timeout)

        # Per-env frame buffers (cleared on episode reset)
        self._frame_buffers: list[list[Image.Image]] = [[] for _ in range(num_envs)]

        # Episodes queued for evaluation
        self._pending: list[EpisodeRecord] = []

        # Stats
        self.total_critiques = 0
        self.total_score_sum = 0.0
        self.issue_counts: dict[str, int] = {}

    def capture_frame(self, camera_data, step: int):
        """Capture camera frames at the render interval.

        Args:
            camera_data: RGB tensor from TiledCamera, shape (num_envs, H, W, 3|4),
                         float32 in [0, 1] or uint8 in [0, 255].
            step: Current step within the rollout (0-indexed).
        """
        if step % self.render_interval != 0:
            return

        # Convert tensor to numpy
        if hasattr(camera_data, "cpu"):
            data = camera_data.cpu().numpy()
        else:
            data = np.asarray(camera_data)

        # Normalize to uint8
        if data.dtype == np.float32 or data.dtype == np.float64:
            data = (data * 255).clip(0, 255).astype(np.uint8)

        # Take only RGB channels (drop alpha if present)
        if data.shape[-1] == 4:
            data = data[:, :, :, :3]

        for env_idx in range(min(self.num_envs, data.shape[0])):
            img = Image.fromarray(data[env_idx])
            self._frame_buffers[env_idx].append(img)

    def on_episode_done(
        self,
        env_idx: int,
        from_square: str = "e2",
        to_square: str = "e4",
        piece_type: str = "piece",
        terminal_step_idx: int = 0,
    ):
        """Called when an episode terminates. Decides whether to queue for evaluation.

        Args:
            env_idx: Which environment finished
            from_square: Source square for the move
            to_square: Target square for the move
            piece_type: Type of piece being moved
            terminal_step_idx: Index into the rollout reward buffer for this terminal step
        """
        frames = self._frame_buffers[env_idx]

        # Clear buffer for next episode
        self._frame_buffers[env_idx] = []

        # Skip if no frames captured
        if len(frames) < 2:
            return

        # Probabilistic selection
        if random.random() > self.frequency:
            return

        self._pending.append(EpisodeRecord(
            env_idx=env_idx,
            frames=frames,
            from_square=from_square,
            to_square=to_square,
            piece_type=piece_type,
            terminal_step_idx=terminal_step_idx,
        ))

    def clear_episode(self, env_idx: int):
        """Clear frame buffer for a specific environment (on reset)."""
        self._frame_buffers[env_idx] = []

    def evaluate_pending(self) -> list[tuple[int, float, EpisodeCritique]]:
        """Evaluate all pending episodes and return (terminal_step_idx, reward, critique).

        Call this between rollout collection and PPO update.

        Returns:
            List of (terminal_step_idx, critic_reward, critique) tuples.
        """
        results = []

        for record in self._pending:
            try:
                critique = self.reasoning.critique_episode(
                    video_frames=record.frames,
                    from_square=record.from_square,
                    to_square=record.to_square,
                    piece_type=record.piece_type,
                )
                reward = self._compute_reward(critique)
                results.append((record.terminal_step_idx, reward, critique))

                # Update stats
                self.total_critiques += 1
                self.total_score_sum += critique.overall_score
                for issue in critique.physical_issues:
                    self.issue_counts[issue] = self.issue_counts.get(issue, 0) + 1

            except Exception as e:
                logger.warning(f"Reason2 critique failed for env {record.env_idx}: {e}")

        self._pending.clear()
        return results

    def _compute_reward(self, critique: EpisodeCritique) -> float:
        """Convert a structured critique into a scalar reward.

        Reward components:
        - Overall quality score (0-10) normalized to [0, 1]
        - Penalty for each physical issue detected
        - Scaled by self.weight
        """
        quality = critique.overall_score / 10.0  # 0.0 to 1.0
        issue_penalty = -0.3 * len(critique.physical_issues)
        return (quality + issue_penalty) * self.weight

    @property
    def mean_score(self) -> float:
        """Mean overall score across all critiques."""
        if self.total_critiques == 0:
            return 0.0
        return self.total_score_sum / self.total_critiques

    @property
    def pending_count(self) -> int:
        """Number of episodes waiting for evaluation."""
        return len(self._pending)
