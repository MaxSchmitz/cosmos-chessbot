"""Camera capture interface for overhead and wrist cameras."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class CameraConfig:
    """Configuration for camera capture."""

    device_id: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    name: str = "camera"


class Camera:
    """Interface for capturing images from cameras."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self._cap = None

    def __enter__(self):
        """Initialize camera on context entry."""
        import cv2
        self._cap = cv2.VideoCapture(self.config.device_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release camera on context exit."""
        if self._cap is not None:
            self._cap.release()

    def reconnect(self) -> None:
        """Release and re-open the camera to recover from USB failures."""
        import cv2

        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self.config.device_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)

    def capture(self) -> Image.Image:
        """Capture a single frame and return as PIL Image.

        Returns:
            PIL Image in RGB format
        """
        import cv2

        if self._cap is None:
            raise RuntimeError("Camera not initialized. Use as context manager.")

        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError(f"Failed to capture frame from camera {self.config.name}")

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def save_frame(self, output_path: Path) -> None:
        """Capture and save a frame to disk."""
        img = self.capture()
        img.save(output_path)
