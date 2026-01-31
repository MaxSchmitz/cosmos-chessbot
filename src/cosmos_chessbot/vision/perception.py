"""Cosmos-Reason2 perception for chess board state extraction."""

from dataclasses import dataclass
from typing import Optional
import json
import re

import torch
import transformers
from PIL import Image


@dataclass
class BoardState:
    """Extracted chess board state."""

    fen: str
    """Board state in FEN notation, or 'NO_BOARD_DETECTED' if no board visible."""

    confidence: float
    """Confidence score (0-1) for the extracted state."""

    anomalies: list[str]
    """List of physical anomalies detected (tilted pieces, pieces between squares, etc.)."""

    raw_response: str
    """Raw text response from Cosmos."""

    @property
    def board_detected(self) -> bool:
        """Check if a chess board was detected in the image."""
        return self.fen != "NO_BOARD_DETECTED" and self.confidence > 0.0


class CosmosPerception:
    """Cosmos-Reason2 based perception for chess board analysis."""

    PIXELS_PER_TOKEN = 32**2

    SYSTEM_PROMPT = "You are a chess board vision system. Your only job is to look at chess board images and output the exact position in FEN notation."

    PERCEPTION_PROMPT = """Look at this chess board and describe what pieces you see on each rank, then output the FEN.

Rank 8 (top, black's side): [describe pieces left to right]
Rank 7: [describe]
Rank 6: [describe]
Rank 5: [describe]
Rank 4: [describe]
Rank 3: [describe]
Rank 2: [describe]
Rank 1 (bottom, white's side): [describe pieces left to right]

Then output in JSON:
{
    "fen": "position w/b KQkq - 0 1",
    "confidence": 0.0-1.0,
    "anomalies": []
}

FEN notation: K/Q/R/B/N/P = white, k/q/r/b/n/p = black, numbers = empty squares, / = rank separator
Example: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
"""

    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-8B",
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
        min_vision_tokens: int = 256,
        max_vision_tokens: int = 8192,
    ):
        """Initialize Cosmos perception model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('auto', 'cuda', 'cpu')
            dtype: Data type for model weights
            min_vision_tokens: Minimum vision tokens for image processing
            max_vision_tokens: Maximum vision tokens for image processing
        """
        self.model_name = model_name

        # Load model
        self.model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device,
            attn_implementation="sdpa"
        )

        self.processor = transformers.Qwen3VLProcessor.from_pretrained(model_name)

        # Configure vision token limits
        self.processor.image_processor.size = {
            "shortest_edge": min_vision_tokens * self.PIXELS_PER_TOKEN,
            "longest_edge": max_vision_tokens * self.PIXELS_PER_TOKEN,
        }

    def perceive(
        self,
        image: Image.Image,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> BoardState:
        """Extract board state from overhead camera image.

        Args:
            image: PIL Image from overhead camera
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)

        Returns:
            BoardState with FEN, confidence, and anomalies
        """
        # Create conversation
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.PERCEPTION_PROMPT},
                ],
            },
        ]

        # Process inputs
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Run inference
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Parse JSON response
        return self._parse_response(output_text)

    def _parse_response(self, response: str) -> BoardState:
        """Parse Cosmos response into structured BoardState.

        Args:
            response: Raw text response from model

        Returns:
            Parsed BoardState
        """
        # Try to extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return BoardState(
                    fen=data.get("fen", ""),
                    confidence=float(data.get("confidence", 0.0)),
                    anomalies=data.get("anomalies", []),
                    raw_response=response,
                )
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                # Fallback if JSON parsing fails
                pass

        # Fallback: return response as-is with low confidence
        return BoardState(
            fen="",
            confidence=0.0,
            anomalies=["Failed to parse structured response"],
            raw_response=response,
        )
