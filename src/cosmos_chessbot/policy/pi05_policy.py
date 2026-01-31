"""π₀.₅ Vision-Language-Action policy implementation."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .base_policy import BasePolicy, PolicyAction


class PI05Policy(BasePolicy):
    """π₀.₅ Vision-Language-Action policy wrapper.

    Wraps the LeRobot π₀.₅ policy for chess manipulation tasks.
    Supports both pretrained and fine-tuned models.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize π₀.₅ policy.

        Args:
            checkpoint_path: Path to fine-tuned checkpoint (None uses base model)
            device: Device to run policy on
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.policy = None
        self.preprocess = None
        self.postprocess = None

        self._load_policy()

    def _load_policy(self):
        """Load π₀.₅ policy from checkpoint or pretrained weights."""
        try:
            from lerobot.policies.pi05 import PI05Policy as LeRobotPI05
            from lerobot.policies.factory import make_pre_post_processors

            if self.checkpoint_path and self.checkpoint_path.exists():
                # Load fine-tuned model
                print(f"Loading fine-tuned π₀.₅ from {self.checkpoint_path}")
                self.policy = LeRobotPI05.from_pretrained(str(self.checkpoint_path))
            else:
                # Load base pretrained model
                print("Loading base π₀.₅ model (lerobot/pi05_base)")
                self.policy = LeRobotPI05.from_pretrained("lerobot/pi05_base")

            self.policy = self.policy.to(self.device).eval()

            # Set up pre/post processors
            checkpoint = str(self.checkpoint_path) if self.checkpoint_path else "lerobot/pi05_base"
            self.preprocess, self.postprocess = make_pre_post_processors(
                self.policy.config,
                checkpoint,
                preprocessor_overrides={"device_processor": {"device": str(self.device)}},
            )

            print(f"π₀.₅ policy loaded successfully on {self.device}")

        except ImportError as e:
            raise ImportError(
                "LeRobot not found. Install with: pip install lerobot\n"
                "Note: π₀.₅ requires separate lerobot_env. See plan for setup."
            ) from e

    def reset(self):
        """Reset policy state between episodes."""
        # π₀.₅ is stateless - no reset needed
        pass

    def _prepare_batch(
        self,
        images: dict[str, Image.Image],
        robot_state: np.ndarray,
        instruction: Optional[str] = None,
    ) -> dict:
        """Convert observations to π₀.₅ input format.

        Args:
            images: Camera views {"egocentric": img, "wrist": img}
            robot_state: Robot state vector
            instruction: Language instruction

        Returns:
            Batch dictionary for π₀.₅ forward pass
        """
        # Convert images to tensors
        image_tensors = {}
        for cam_name, img in images.items():
            # Convert PIL to numpy then tensor
            img_np = np.array(img)
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            image_tensors[cam_name] = img_tensor.to(self.device)

        # Convert robot state to tensor
        state_tensor = torch.from_numpy(robot_state).unsqueeze(0).to(self.device)  # [1, state_dim]

        batch = {
            "observation.images": image_tensors,
            "observation.state": state_tensor,
        }

        # Add language instruction if provided
        if instruction is not None:
            batch["observation.instruction"] = [instruction]  # List of strings

        # Apply preprocessing
        if self.preprocess is not None:
            batch = self.preprocess(batch)

        return batch

    def select_action(
        self,
        images: dict[str, Image.Image],
        robot_state: np.ndarray,
        instruction: Optional[str] = None,
    ) -> PolicyAction:
        """Select action given current observations.

        Args:
            images: Camera views {"egocentric": img, "wrist": img}
            robot_state: Current robot state (joint positions, gripper, etc.)
            instruction: Language instruction (e.g., "Pick e2 and place at e4")

        Returns:
            PolicyAction with predicted actions
        """
        # Prepare input batch
        batch = self._prepare_batch(images, robot_state, instruction)

        # Run policy inference
        with torch.inference_mode():
            pred_action = self.policy.select_action(batch)

            # Apply postprocessing if available
            if self.postprocess is not None:
                pred_action = self.postprocess(pred_action)

        # Convert to numpy
        actions = pred_action.cpu().numpy()

        return PolicyAction(
            actions=actions,
            success_probability=1.0,  # π₀.₅ doesn't provide confidence scores
            metadata={
                "policy": "pi05",
                "used_language": instruction is not None,
                "device": str(self.device),
            }
        )

    def plan_action(
        self,
        images: dict[str, Image.Image],
        robot_state: np.ndarray,
        instruction: Optional[str] = None,
    ) -> list[PolicyAction]:
        """Plan action candidates.

        π₀.₅ doesn't support planning, so this returns a single action
        from select_action.

        Args:
            images: Camera views
            robot_state: Current robot state
            instruction: Language instruction

        Returns:
            List with single PolicyAction
        """
        action = self.select_action(images, robot_state, instruction)
        return [action]
