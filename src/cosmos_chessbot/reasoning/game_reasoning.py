"""Chess game reasoning using Cosmos Reason2 for embodied AI.

This module uses Cosmos Reason2's embodied reasoning capabilities to understand
the chess game flow: whose turn it is, when moves are complete, and what actions
to take. This is the key differentiator from traditional chess AI.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import json
import re

from PIL import Image
import torch
import transformers


class Turn(Enum):
    """Whose turn it is."""
    ROBOT = "robot"
    OPPONENT = "opponent"
    UNKNOWN = "unknown"


@dataclass
class MoveDetection:
    """Result of move detection from video."""
    move_occurred: bool
    """Whether a move was detected."""

    from_square: Optional[str] = None
    """Starting square in algebraic notation (e.g., 'e2')."""

    to_square: Optional[str] = None
    """Ending square in algebraic notation (e.g., 'e4')."""

    piece_type: Optional[str] = None
    """Type of piece that moved (e.g., 'pawn', 'knight')."""

    confidence: float = 0.0
    """Confidence in the detection."""

    reasoning: str = ""
    """Step-by-step reasoning from Cosmos."""


@dataclass
class GameState:
    """Current state of the chess game."""
    whose_turn: Turn
    """Whose turn it is."""

    opponent_moving: bool
    """Whether opponent is currently making a move."""

    should_robot_act: bool
    """Whether the robot should make its move now."""

    reasoning: str
    """Step-by-step reasoning from Cosmos."""

    confidence: float
    """Confidence in the game state analysis."""


class ChessGameReasoning:
    """Chess game reasoning using Cosmos Reason2.

    Uses Cosmos Reason2's embodied reasoning to understand game flow:
    - Whose turn is it?
    - Is the opponent currently moving?
    - What piece did they move?
    - Should the robot act now?

    This demonstrates Cosmos Reason2's key strength: embodied AI reasoning
    about multi-agent interactions and temporal dynamics.
    """

    PIXELS_PER_TOKEN = 32**2

    SYSTEM_PROMPT = "You are an embodied chess-playing robot with an egocentric camera view. The camera view is YOUR view of the chess board and your opponent."

    TURN_DETECTION_PROMPT = """Watch this video from my egocentric camera and reason about the chess game.

The camera view is MY view as the robot. I am playing chess against a human opponent.

Analyze the video and answer:
1. Whose turn is it? (mine or my opponent's)
2. Is my opponent currently making a move? (reaching for pieces, moving a piece)
3. Should I make my move now, or should I wait?

Reason step-by-step about what you observe, then provide your conclusion in JSON:
{
    "whose_turn": "robot" or "opponent",
    "opponent_moving": true/false,
    "should_robot_act": true/false,
    "reasoning": "your step-by-step reasoning",
    "confidence": 0.0-1.0
}
"""

    MOVE_DETECTION_PROMPT = """Watch this video from my egocentric camera. My opponent just made a chess move.

The camera view is MY view as the robot.

Identify what move they made:
1. Which piece did they move?
2. Where did it start? (square in algebraic notation like 'e2')
3. Where did it end? (square in algebraic notation like 'e4')

Reason step-by-step about what you observe, then provide your conclusion in JSON:
{
    "move_occurred": true/false,
    "from_square": "starting square or null",
    "to_square": "ending square or null",
    "piece_type": "pawn/knight/bishop/rook/queen/king or null",
    "reasoning": "your step-by-step reasoning",
    "confidence": 0.0-1.0
}
"""

    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-8B",
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize Cosmos Reason2 for chess game reasoning.

        Args:
            model_name: Cosmos model identifier
            device: Device to run on ('auto', 'cuda', 'cpu')
            dtype: Data type for model weights
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

    def analyze_game_state(
        self,
        video_frames: list[Image.Image],
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> GameState:
        """Analyze game state from video frames.

        Uses embodied reasoning to determine whose turn it is and whether
        the robot should act.

        Args:
            video_frames: List of PIL Images (video frames)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            GameState with turn information and reasoning
        """
        # Create conversation with robot-centric framing
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_frames},
                    {"type": "text", "text": self.TURN_DETECTION_PROMPT},
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
        return self._parse_game_state(output_text)

    def detect_move(
        self,
        video_frames: list[Image.Image],
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> MoveDetection:
        """Detect what move the opponent made from video.

        Uses embodied reasoning to identify which piece moved and where.

        Args:
            video_frames: List of PIL Images showing the move
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            MoveDetection with move information and reasoning
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
                    {"type": "video", "video": video_frames},
                    {"type": "text", "text": self.MOVE_DETECTION_PROMPT},
                ],
            },
        ]

        # Process and generate
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

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
        return self._parse_move_detection(output_text)

    def _parse_game_state(self, response: str) -> GameState:
        """Parse Cosmos response into GameState.

        Args:
            response: Raw text response from model

        Returns:
            Parsed GameState
        """
        # Try to extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group(0))

                # Parse whose turn
                turn_str = data.get("whose_turn", "unknown").lower()
                whose_turn = Turn.ROBOT if turn_str == "robot" else (
                    Turn.OPPONENT if turn_str == "opponent" else Turn.UNKNOWN
                )

                return GameState(
                    whose_turn=whose_turn,
                    opponent_moving=data.get("opponent_moving", False),
                    should_robot_act=data.get("should_robot_act", False),
                    reasoning=data.get("reasoning", response),
                    confidence=float(data.get("confidence", 0.0)),
                )
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        # Fallback if JSON parsing fails
        return GameState(
            whose_turn=Turn.UNKNOWN,
            opponent_moving=False,
            should_robot_act=False,
            reasoning=response,
            confidence=0.0,
        )

    def _parse_move_detection(self, response: str) -> MoveDetection:
        """Parse Cosmos response into MoveDetection.

        Args:
            response: Raw text response from model

        Returns:
            Parsed MoveDetection
        """
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return MoveDetection(
                    move_occurred=data.get("move_occurred", False),
                    from_square=data.get("from_square"),
                    to_square=data.get("to_square"),
                    piece_type=data.get("piece_type"),
                    confidence=float(data.get("confidence", 0.0)),
                    reasoning=data.get("reasoning", response),
                )
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        # Fallback
        return MoveDetection(
            move_occurred=False,
            confidence=0.0,
            reasoning=response,
        )
