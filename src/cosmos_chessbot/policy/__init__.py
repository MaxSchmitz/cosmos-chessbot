"""Manipulation policy interfaces for chess piece pick-and-place."""

from .base_policy import BasePolicy, PolicyAction

__all__ = [
    "BasePolicy",
    "PolicyAction",
    "PPOPolicy",
]


def __getattr__(name):
    if name == "PPOPolicy":
        from .ppo_policy import PPOPolicy
        return PPOPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
