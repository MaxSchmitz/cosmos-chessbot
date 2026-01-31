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

    PERCEPTION_PROMPT = """Analyze this image and extract the chess board position in FEN (Forsyth-Edwards Notation).

IMPORTANT: First, verify that you can see a chess board in the image. If you cannot see a chess board, set the FEN to "NO_BOARD_DETECTED" and confidence to 0.0.

=== FEN NOTATION EXPLAINED ===

FEN has 6 components separated by spaces:
1. Piece placement (rank 8 to rank 1, separated by /)
2. Active color (w or b)
3. Castling availability (KQkq or - if none)
4. En passant target square (e.g., e3 or - if none)
5. Halfmove clock (number, use 0 if unknown)
6. Fullmove number (number, use 1 if unknown)

Example: "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

=== PIECE NOTATION ===
- Uppercase = White pieces: K (King), Q (Queen), R (Rook), B (Bishop), N (Knight), P (Pawn)
- Lowercase = Black pieces: k (king), q (queen), r (rook), b (bishop), n (knight), p (pawn)
- Numbers 1-8 = consecutive empty squares
- / = separates ranks (rows)

=== COORDINATE SYSTEM ===
Files (columns): a-h from left to right (from white's perspective)
Ranks (rows): 8 at the top (black's side) to 1 at the bottom (white's side)

=== HOW TO SCAN THE BOARD ===

Step 1: Identify the board orientation
- Rank 8 (black's starting rank) should be at the TOP
- Rank 1 (white's starting rank) should be at the BOTTOM
- If the board is viewed from black's perspective, mentally rotate it

Step 2: Scan each rank from rank 8 to rank 1
For EACH rank, scan from file a to file h (left to right):
  - If square is empty, count it (combine consecutive empty squares into a number)
  - If square has a piece, identify it (K/Q/R/B/N/P for white, k/q/r/b/n/p for black)

Step 3: Build the FEN string
- Rank 8: e.g., "rnbqkbnr" (black's back rank)
- Add "/" separator
- Rank 7: e.g., "pppppppp" (black's pawns)
- Continue for all 8 ranks
- Add space and other components: " w KQkq - 0 1"

Example scanning:
- Starting position rank 8: r n b q k b n r → "rnbqkbnr"
- Starting position rank 7: p p p p p p p p → "pppppppp"
- Starting position rank 6: empty row → "8"
- Starting position rank 5: empty row → "8"
- Starting position rank 4: empty row → "8"
- Starting position rank 3: empty row → "8"
- Starting position rank 2: P P P P P P P P → "PPPPPPPP"
- Starting position rank 1: R N B Q K B N R → "RNBQKBNR"
Full FEN: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

=== YOUR TASK ===

1. First, confirm you see a chess board
2. Identify board orientation (which side is rank 8, which is rank 1)
3. Scan systematically rank by rank (8→1), file by file (a→h)
4. Build the piece placement string
5. Determine active color (w or b)
6. Determine castling rights (KQkq, or - if unknown)
7. Determine en passant square (or - if none)
8. Use 0 for halfmove clock, 1 for fullmove number (if unknown)

Provide your response in JSON format:
{
    "fen": "complete FEN notation with all 6 components",
    "confidence": a number between 0 and 1,
    "anomalies": ["list any physical issues: tilted pieces, pieces between squares, occlusions, etc."]
}

If NO chess board is visible:
{
    "fen": "NO_BOARD_DETECTED",
    "confidence": 0.0,
    "anomalies": ["no chess board visible in image"]
}

CRITICAL: The FEN MUST have all 6 components separated by spaces. Example format:
"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
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
