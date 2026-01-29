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

    SYSTEM_PROMPT = "You are a precise physical reasoning assistant specialized in chess board perception."

    PERCEPTION_PROMPT = """Analyze this overhead camera view and determine if there is a chess board present.

IMPORTANT: First, verify that you can see a chess board in the image. If you cannot see a chess board, set the FEN to "NO_BOARD_DETECTED" and confidence to 0.0.

Provide your response in the following JSON format:
{
    "fen": "the board position in FEN notation, or NO_BOARD_DETECTED if no board is visible",
    "confidence": a number between 0 and 1 indicating your confidence (0.0 if no board detected),
    "anomalies": [
        "list of physical anomalies you observe",
        "examples: 'no chess board visible', 'white knight on f3 is tilted', 'black pawn between d6 and d7', 'pieces occluding each other'"
    ]
}

If a chess board IS present, focus on:
1. Accurate piece identification and position
2. Physical state of pieces (upright, tilted, knocked over)
3. Pieces that are not centered on squares
4. Any occlusions or ambiguities

Be precise with the FEN notation. Remember the FEN format is: piece placement, active color, castling rights, en passant, halfmove clock, fullmove number.
If you cannot determine some aspects (like whose turn it is), use reasonable defaults (white to move, all castling available).

If NO chess board is visible in the image, respond with:
{
    "fen": "NO_BOARD_DETECTED",
    "confidence": 0.0,
    "anomalies": ["no chess board visible in image"]
}
"""

    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-2B",
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
