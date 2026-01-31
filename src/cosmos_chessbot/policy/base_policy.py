"""Abstract base class for manipulation policies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class PolicyAction:
    """Result from policy execution.

    Attributes:
        actions: Robot joint commands (typically shape: [horizon, action_dim])
        success_probability: Confidence in action (0.0 to 1.0)
        metadata: Policy-specific information (e.g., future states, values)
    """
    actions: np.ndarray
    success_probability: float
    metadata: dict


class BasePolicy(ABC):
    """Abstract base class for manipulation policies.

    Provides unified interface for π₀.₅ and Cosmos Policy, enabling
    empirical comparison on the same task.
    """

    @abstractmethod
    def reset(self):
        """Reset policy state between episodes.

        Called at the start of each new manipulation episode to clear
        any temporal state or history.
        """
        pass

    @abstractmethod
    def select_action(
        self,
        images: dict[str, Image.Image],
        robot_state: np.ndarray,
        instruction: Optional[str] = None,
    ) -> PolicyAction:
        """Select action given current observations.

        Args:
            images: Dictionary of camera views, e.g., {"egocentric": img, "wrist": img}
            robot_state: Current robot state (joint positions, gripper state, etc.)
            instruction: Optional language instruction (used by π₀.₅)

        Returns:
            PolicyAction with predicted actions and metadata
        """
        pass

    @abstractmethod
    def plan_action(
        self,
        images: dict[str, Image.Image],
        robot_state: np.ndarray,
        instruction: Optional[str] = None,
    ) -> list[PolicyAction]:
        """Plan multiple candidate actions.

        For policies with planning capability (Cosmos Policy), generates
        and evaluates multiple action candidates. For policies without
        planning (π₀.₅), returns single action from select_action.

        Args:
            images: Dictionary of camera views
            robot_state: Current robot state
            instruction: Optional language instruction

        Returns:
            List of PolicyAction candidates, sorted by success_probability
            (highest confidence first)
        """
        pass
