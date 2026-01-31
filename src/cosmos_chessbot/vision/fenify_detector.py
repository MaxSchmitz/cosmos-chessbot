"""FEN detection using Fenify-3D.

Fenify-3D transforms real-world chess board images into FEN notation.
Works with various viewing angles, not just top-down views.

Repository: https://github.com/notnil/fenify-3D
Accuracy: 95% per-square (validation), 85.2% (real-world footage)
"""

from pathlib import Path
from typing import Optional
import sys

from PIL import Image
import torch

# Add fenify-3D to path
FENIFY_DIR = Path(__file__).parent.parent.parent.parent / "external" / "fenify-3D"


class FenifyDetector:
    """FEN detection using Fenify-3D.

    This detector works with various viewing angles, making it ideal for
    egocentric camera views.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        """Initialize Fenify detector.

        Args:
            model_path: Path to Fenify model file (.pt)
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        # Add fenify-3D to Python path
        if str(FENIFY_DIR) not in sys.path:
            sys.path.insert(0, str(FENIFY_DIR))

        try:
            from prediction import BoardPredictor
        except ImportError:
            raise ImportError(
                f"Fenify-3D not found at {FENIFY_DIR}. "
                "Please clone it: git clone https://github.com/notnil/fenify-3D.git external/fenify-3D"
            )

        # Default model path
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent.parent / "models" / "fenify_model.pt"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Fenify model not found at {model_path}. "
                "Download from: https://github.com/notnil/fenify-3D/releases"
            )

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        # Load model
        self.predictor = BoardPredictor(
            model_file_path=str(model_path),
            device=device
        )

        print(f"Loaded Fenify-3D model from {model_path} on {device}")

    def detect_fen(
        self,
        image: Image.Image,
        return_confidence: bool = False,
    ) -> str:
        """Detect FEN from chess board image.

        Args:
            image: PIL Image of chess board
            return_confidence: If True, return (fen, confidence_dict)

        Returns:
            FEN string, or (FEN string, confidence dict) if return_confidence=True
        """
        # Save image temporarily (Fenify expects file path)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        try:
            if return_confidence:
                # Get FEN with per-square confidence
                result = self.predictor.predict_with_confidence(tmp_path)
                fen = result['fen']
                confidence = result['confidence']
                return fen, confidence
            else:
                # Get FEN only
                board = self.predictor.predict(tmp_path)
                fen = board.fen()
                return fen
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

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
        if return_confidence:
            result = self.predictor.predict_with_confidence(image_path)
            return result['fen'], result['confidence']
        else:
            board = self.predictor.predict(image_path)
            return board.fen()
