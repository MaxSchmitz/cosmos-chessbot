"""Remote Cosmos perception client for connecting to GPU server."""

import base64
import io
from typing import Optional

import httpx
from PIL import Image

from .perception import BoardState


class RemoteCosmosPerception:
    """Client for remote Cosmos inference server.

    Use this instead of CosmosPerception when running Cosmos on a remote GPU.
    """

    def __init__(self, server_url: str, timeout: float = 60.0):
        """Initialize remote perception client.

        Args:
            server_url: URL of the Cosmos inference server (e.g., "http://gpu-server:8000")
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def __del__(self):
        """Close HTTP client on cleanup."""
        self.client.close()

    def health_check(self) -> bool:
        """Check if server is healthy and ready.

        Returns:
            True if server is ready
        """
        try:
            response = self.client.get(f"{self.server_url}/health")
            response.raise_for_status()
            data = response.json()
            return data.get("status") == "healthy" and data.get("model_loaded", False)
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

    def perceive(
        self,
        image: Image.Image,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> BoardState:
        """Extract board state from egocentric camera image via remote server.

        Args:
            image: PIL Image from egocentric camera
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            BoardState with FEN, confidence, and anomalies
        """
        # Encode image as base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Send request
        try:
            response = self.client.post(
                f"{self.server_url}/perceive",
                json={
                    "image_base64": image_base64,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                },
            )
            response.raise_for_status()
            data = response.json()

            return BoardState(
                fen=data["fen"],
                confidence=data["confidence"],
                anomalies=data["anomalies"],
                raw_response=data["raw_response"],
            )

        except httpx.HTTPError as e:
            raise RuntimeError(f"Remote perception request failed: {e}")
