"""FEN detection using vision LLMs (Claude, GPT-4V, Gemini).

This approach uses state-of-the-art vision language models for chess board
position detection. Simpler and more accurate than specialized CV models.
"""

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Optional, Literal

import httpx
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


ProviderType = Literal["anthropic", "openai", "google"]


class LLMFenDetector:
    """FEN detection using vision LLMs.

    Supports multiple providers:
    - anthropic: Claude 3.5 Sonnet (best vision quality)
    - openai: GPT-4V (excellent chess understanding)
    - google: Gemini 2.0 Flash (fastest, cheapest)
    """

    FEN_PROMPT = """Analyze this chess board and output the FEN position.

Steps:
1. Scan rank 8 (top, Black's back rank) from left to right (a8 to h8)
2. Continue with rank 7, then 6, 5, 4, 3, 2, and finally rank 1
3. For each rank, note pieces (K/Q/R/B/N/P for White, k/q/r/b/n/p for Black) and empty squares (use numbers)
4. Determine whose turn it is (w or b)
5. Check castling rights (KQkq or -)

Output format: piece_placement active_color castling en_passant halfmove fullmove

Output only the FEN string, nothing else."""

    def __init__(
        self,
        provider: ProviderType = "anthropic",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        detail: str = "high",  # For OpenAI: "low", "high", or "auto"
    ):
        """Initialize LLM FEN detector.

        Args:
            provider: Which LLM provider to use
            api_key: API key (or set via environment variables)
            model: Specific model name (uses defaults if not provided)
        """
        self.provider = provider

        # Get API key from parameter or environment
        if api_key is None:
            if provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
            elif provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
            elif provider == "google":
                api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                f"API key not provided. Set {provider.upper()}_API_KEY environment variable "
                f"or pass api_key parameter"
            )

        self.api_key = api_key

        # Set default models
        if model is None:
            if provider == "anthropic":
                model = "claude-sonnet-4-5"
            elif provider == "openai":
                model = "gpt-4o"  # Can be overridden with model parameter
            elif provider == "google":
                model = "gemini-2.0-flash-exp"

        self.model = model
        self.detail = detail
        self.client = httpx.Client(timeout=30.0)

        print(f"Initialized LLM FEN detector: {provider} ({model}, detail={detail})")

    def _image_to_base64(self, image: Image.Image, max_size: int = 1200) -> str:
        """Convert PIL Image to base64 string.

        Args:
            image: PIL Image
            max_size: Maximum dimension (width or height) in pixels

        Returns:
            Base64-encoded image string
        """
        # Resize if image is too large
        if max(image.size) > max_size:
            # Calculate new size maintaining aspect ratio
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"Resized image from {image.size} to {new_size}")

        # Convert to JPEG for smaller size (chess boards compress well)
        buffer = io.BytesIO()
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            image = rgb_image
        elif image.mode != 'RGB':
            image = image.convert('RGB')

        image.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)

        encoded = base64.b64encode(buffer.read()).decode("utf-8")
        print(f"Encoded image size: {len(encoded)} bytes")

        return encoded

    def _call_anthropic(self, image_b64: str) -> str:
        """Call Anthropic Claude API."""
        response = self.client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 200,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": self.FEN_PROMPT,
                            },
                        ],
                    }
                ],
            },
        )

        # Better error handling
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            error_detail = response.text
            raise ValueError(
                f"Anthropic API error ({response.status_code}): {error_detail}"
            ) from e

        result = response.json()
        return result["content"][0]["text"]

    def _call_openai(self, image_b64: str) -> str:
        """Call OpenAI GPT API (GPT-4o, GPT-5.2, etc.)."""
        # GPT-5.2 and o1 models use max_completion_tokens instead of max_tokens
        if "gpt-5" in self.model.lower() or "o1" in self.model.lower():
            max_tokens_key = "max_completion_tokens"
        else:
            max_tokens_key = "max_tokens"

        request_body = {
            "model": self.model,
            max_tokens_key: 200,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": self.detail,
                            },
                        },
                        {
                            "type": "text",
                            "text": self.FEN_PROMPT,
                        },
                    ],
                }
            ],
        }

        response = self.client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=request_body,
        )

        # Better error handling
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            error_detail = response.text
            raise ValueError(
                f"OpenAI API error ({response.status_code}): {error_detail}"
            ) from e

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def _call_google(self, image_b64: str) -> str:
        """Call Google Gemini API."""
        response = self.client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            headers={
                "Content-Type": "application/json",
            },
            params={
                "key": self.api_key,
            },
            json={
                "contents": [
                    {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": image_b64,
                                }
                            },
                            {
                                "text": self.FEN_PROMPT,
                            },
                        ]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": 200,
                },
            },
        )
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]

    def _extract_fen(self, response_text: str) -> str:
        """Extract FEN string from LLM response.

        LLMs might add extra text, so we extract just the FEN string.
        """
        # Remove any markdown code blocks
        response_text = re.sub(r"```[\w]*\n?", "", response_text)
        response_text = response_text.strip()

        # FEN pattern: piece placement, active color, castling, en passant, halfmove, fullmove
        # Example: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e3 0 1
        fen_pattern = r"([rnbqkpRNBQKP1-8/]+)\s+([wb])\s+((?:K?Q?k?q?|-))\s+((?:[a-h][36]|-))\s+(\d+)\s+(\d+)"

        match = re.search(fen_pattern, response_text)
        if match:
            return match.group(0)

        # If no match, try to find just the piece placement part and assume defaults
        piece_pattern = r"[rnbqkpRNBQKP1-8/]+"
        match = re.search(piece_pattern, response_text)
        if match:
            # Found piece placement, but missing metadata - return what we found
            # and let the caller decide if it's acceptable
            return response_text.strip()

        raise ValueError(
            f"Could not extract valid FEN from response: {response_text}"
        )

    def detect_fen(
        self,
        image: Image.Image,
        return_confidence: bool = False,
    ) -> str:
        """Detect FEN from chess board image.

        Args:
            image: PIL Image of chess board
            return_confidence: If True, return (fen, confidence_dict)
                Note: LLMs don't provide per-square confidence, so this
                will return a dummy confidence of 1.0

        Returns:
            FEN string, or (FEN string, confidence dict) if return_confidence=True
        """
        # Convert image to base64
        image_b64 = self._image_to_base64(image)

        # Call appropriate provider
        if self.provider == "anthropic":
            response_text = self._call_anthropic(image_b64)
        elif self.provider == "openai":
            response_text = self._call_openai(image_b64)
        elif self.provider == "google":
            response_text = self._call_google(image_b64)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        # Extract FEN from response
        fen = self._extract_fen(response_text)

        if return_confidence:
            # LLMs don't provide per-square confidence
            # Return dummy confidence of 1.0
            confidence = {"overall": 1.0}
            return fen, confidence

        return fen

    def detect_fen_from_path(
        self,
        image_path: str,
        return_confidence: bool = False,
    ) -> str:
        """Detect FEN from image file path.

        Args:
            image_path: Path to image file
            return_confidence: If True, return (fen, confidence_dict)

        Returns:
            FEN string, or (FEN string, confidence dict) if return_confidence=True
        """
        image = Image.open(image_path)
        return self.detect_fen(image, return_confidence=return_confidence)

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()
