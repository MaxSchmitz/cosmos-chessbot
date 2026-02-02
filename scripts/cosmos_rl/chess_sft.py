#!/usr/bin/env python3
"""Chess FEN Detection Training with Cosmos-RL

Adapted from Cosmos Cookbook intelligent-transportation recipe:
https://github.com/nvidia-cosmos/cosmos-cookbook/tree/main/scripts/examples/reason2/intelligent-transportation

This script trains Cosmos-Reason2 to detect chess positions and output FEN notation
from images using the VALUE dataset (180K training samples).
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cosmos_rl.launcher.worker_entry
import cosmos_rl.policy.config
import pydantic
import torch.utils.data
from cosmos_reason2_utils.text import create_conversation
from cosmos_reason2_utils.vision import VisionConfig


class CustomDatasetConfig(pydantic.BaseModel):
    """Configuration for chess FEN dataset."""
    annotation_path: str = pydantic.Field()
    """Path to Llava format JSON annotations."""
    media_path: str = pydantic.Field(default="")
    """Media path prefix (empty if paths are absolute)."""
    system_prompt: str = pydantic.Field(default="")
    """System prompt for the assistant."""


class CustomConfig(pydantic.BaseModel):
    """Custom configuration for chess training."""
    dataset: CustomDatasetConfig = pydantic.Field()
    """Dataset configuration."""

    vision: VisionConfig = pydantic.Field(
        default=VisionConfig(nframes=1)
    )
    """Vision processor configuration (single frame for static images)."""


class ChessFENDataset(torch.utils.data.Dataset):
    """Dataset for chess FEN detection from images.

    Loads data in Llava format:
    {
        "id": "unique_id",
        "image": "/path/to/image.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\\nWhat is the FEN?"},
            {"from": "gpt", "value": "rnbqkbnr/pppppppp/... w KQkq - 0 1"}
        ]
    }
    """

    def __init__(
        self,
        config: cosmos_rl.policy.config.Config,
        custom_config: CustomConfig,
    ):
        """Initialize dataset.

        Args:
            config: Cosmos-RL policy configuration
            custom_config: Custom dataset configuration
        """
        # Load annotations
        annotation_path = custom_config.dataset.annotation_path

        # Handle relative paths
        if not os.path.isabs(annotation_path):
            # Try relative to current directory
            if not os.path.exists(annotation_path):
                # Try relative to script directory
                script_dir = os.path.dirname(os.path.abspath(__file__))
                annotation_path = os.path.join(script_dir, "..", "..", annotation_path)

        print(f"Loading annotations from: {annotation_path}")
        self.annotation = json.load(open(annotation_path))
        print(f"Loaded {len(self.annotation)} samples")

        self.media_path = custom_config.dataset.media_path
        self.system_prompt = custom_config.dataset.system_prompt
        self.config = config
        self.custom_config = custom_config
        self.vision_kwargs = custom_config.vision.model_dump(exclude_none=True)

    def __len__(self):
        """Return dataset size."""
        return len(self.annotation)

    def __getitem__(self, idx: int) -> list[dict]:
        """Get a training sample.

        Args:
            idx: Sample index

        Returns:
            Conversation data processed for Cosmos-Reason2
        """
        sample = self.annotation[idx]

        # Extract from Llava format
        user_prompt = sample["conversations"][0]["value"]
        response = sample["conversations"][1]["value"]
        image_path = sample["image"]

        # Remove <image> tag from prompt (will be added by create_conversation)
        user_prompt = re.sub(r"(\n)?</?image>(\n)?", "", user_prompt)

        # Join with media_path if specified
        if self.media_path != "":
            image_path = os.path.join(self.media_path, image_path)

        # Create conversation with Cosmos-Reason2 utilities
        conversations = create_conversation(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response=response,
            images=[image_path],
            videos=None,
            vision_kwargs=self.vision_kwargs,
        )

        return conversations


def main(args):
    """Entry point for Cosmos-RL training.

    Args:
        args: Command-line arguments
    """
    cosmos_rl.launcher.worker_entry.main(args)


if __name__ == "__main__":
    # Parse arguments and launch training
    main(sys.argv[1:])
