"""Shared ActorCritic policy and observation normalizer.

Used by both training (scripts/training/train_chess_rl.py) and evaluation
(scripts/evaluation/eval_policy.py) to prevent model definition drift.

The actor uses a tanh-squashed Gaussian: raw samples from Normal(mean, std)
are passed through tanh to bound actions to (-1, 1).  Log-probabilities
include the Jacobian correction for the tanh transform.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """Actor-Critic with shared trunk, tanh-squashed Gaussian actor.

    Architecture: shared ELU layer -> separate actor/critic branches.
    The critic gradient is scaled 0.5x through the shared trunk so it
    doesn't distort the actor's features.
    """

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
        # Start with moderate exploration; tanh compresses the range
        self.actor_log_std = nn.Parameter(torch.full((act_dim,), -0.5))

        self.critic_branch = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
        )
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        """Return raw (pre-tanh) action mean, std, and value estimate."""
        shared_features = self.shared(obs)
        actor_features = self.actor_branch(shared_features)
        action_mean = self.actor_mean(actor_features)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        # Critic: scale gradient through shared trunk by 0.5x
        critic_input = shared_features + 0.5 * (
            shared_features.detach() - shared_features
        )
        critic_features = self.critic_branch(critic_input)
        value = self.critic_head(critic_features).squeeze(-1)
        return action_mean, action_std, value

    def get_action_and_value(self, obs: torch.Tensor):
        """Sample tanh-squashed action, return (action, log_prob, value)."""
        action_mean, action_std, value = self(obs)
        dist = Normal(action_mean, action_std)
        raw_action = dist.sample()
        action = torch.tanh(raw_action)
        # Log-prob with Jacobian correction: log|d(tanh)/dx| = log(1 - tanh^2)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate stored (tanh-squashed) actions under current policy."""
        action_mean, action_std, value = self(obs)
        dist = Normal(action_mean, action_std)
        # Inverse tanh to recover raw (pre-squash) actions
        raw_actions = torch.atanh(actions.clamp(-0.999, 0.999))
        log_prob = dist.log_prob(raw_actions).sum(dim=-1)
        log_prob -= torch.log(1.0 - actions.pow(2) + 1e-6).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


class RunningMeanStd:
    """Tracks running mean and std for observation normalization."""

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
