"""Chess board segmentation using Ultimate V2 ONNX model.

This module uses the Ultimate V2 Breakthrough model for real-time chess board
detection and segmentation. The segmented board can then be used for FEN detection.

Model: https://huggingface.co/yamero999/ultimate-v2-chess-onnx
- Speed: ~15ms on CPU
- Accuracy: Perfect (Dice score: 1.0)
- Size: 2.09MB
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort


class BoardSegmentation:
    """Chess board segmentation using Ultimate V2 ONNX model."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        input_size: Tuple[int, int] = (256, 256),
        threshold: float = 0.5,
    ):
        """Initialize board segmentation.

        Args:
            model_path: Path to ONNX model file
            input_size: Model input size (width, height)
            threshold: Threshold for binary mask (0.0-1.0)
        """
        self.input_size = input_size
        self.threshold = threshold

        # Default model path
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent.parent / "models" / "ultimate_v2_breakthrough_accurate.onnx"

        # Load ONNX model
        self.session = ort.InferenceSession(str(model_path))

        print(f"Loaded Ultimate V2 Chess Board Segmentation model from {model_path}")

    def segment_board(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """Segment chess board from image.

        Args:
            image: PIL Image

        Returns:
            Tuple of (segmentation_mask, cropped_board)
            - segmentation_mask: Binary mask of board location (H, W)
            - cropped_board: Cropped and aligned board image
        """
        # Convert PIL to numpy
        image_np = np.array(image)
        if image_np.shape[2] == 4:  # RGBA
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        # Preprocess
        input_tensor = self._preprocess(image_np)

        # Run inference
        mask = self._inference(input_tensor)

        # Postprocess
        binary_mask = (mask > self.threshold).astype(np.uint8)

        # Crop board region
        cropped_board = self._crop_board(image_np, binary_mask)

        return binary_mask, cropped_board

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input.

        Args:
            image: RGB image (H, W, C)

        Returns:
            Preprocessed tensor (1, C, H, W)
        """
        # Resize to model input size
        image_resized = cv2.resize(image, self.input_size)

        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0

        # Convert to NCHW format
        input_tensor = np.transpose(image_normalized, (2, 0, 1))[np.newaxis, ...]

        return input_tensor

    def _inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run inference on the model.

        Args:
            input_tensor: Preprocessed input (1, C, H, W)

        Returns:
            Segmentation mask (H, W)
        """
        # Get input name
        input_name = self.session.get_inputs()[0].name

        # Run inference
        outputs = self.session.run(None, {input_name: input_tensor})

        # Apply sigmoid to get probabilities
        mask = 1.0 / (1.0 + np.exp(-outputs[0]))

        return mask.squeeze()

    def _crop_board(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Crop board region from image using segmentation mask.

        Args:
            image: Original RGB image
            mask: Binary segmentation mask

        Returns:
            Cropped board image
        """
        # Resize mask to image size
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # Find contours
        contours, _ = cv2.findContours(
            mask_resized.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            # No board detected, return original image
            return image

        # Get largest contour (assumed to be the board)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop image
        cropped = image[y:y+h, x:x+w]

        return cropped

    def visualize_segmentation(
        self,
        image: Image.Image,
        save_path: Optional[Path] = None,
    ) -> Image.Image:
        """Visualize segmentation results.

        Args:
            image: Input PIL Image
            save_path: Optional path to save visualization

        Returns:
            PIL Image with visualization
        """
        import matplotlib.pyplot as plt

        # Run segmentation
        mask, cropped = self.segment_board(image)

        # Convert to numpy for plotting
        image_np = np.array(image)

        # Create overlay
        overlay = image_np.copy()
        mask_resized = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
        overlay[mask_resized > 0] = [255, 0, 0]  # Red overlay

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='hot')
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')

        axes[2].imshow(cropped)
        axes[2].set_title('Cropped Board')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Saved visualization to {save_path}")

        # Convert matplotlib figure to PIL Image
        fig.canvas.draw()
        vis_image = Image.frombytes(
            'RGB',
            fig.canvas.get_width_height(),
            fig.canvas.tostring_rgb()
        )
        plt.close(fig)

        return vis_image
