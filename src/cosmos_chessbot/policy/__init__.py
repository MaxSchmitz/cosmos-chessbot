"""Manipulation policy interfaces for chess piece pick-and-place."""

from .base_policy import BasePolicy, PolicyAction

__all__ = [
    "BasePolicy",
    "PolicyAction",
    "PPOPolicy",
    "WaypointPolicy",
]


def __getattr__(name):
    if name == "PPOPolicy":
        from .ppo_policy import PPOPolicy
        return PPOPolicy
    if name == "WaypointPolicy":
        from .waypoint_policy import WaypointPolicy
        return WaypointPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
