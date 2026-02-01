"""Vision module: camera capture, FEN detection, and Cosmos reasoning."""

from .camera import Camera, CameraConfig
from .perception import CosmosPerception, BoardState
from .remote_perception import RemoteCosmosPerception
from .board_segmentation import BoardSegmentation
from .fen_detection import FENDetector, ChessPiece
from .fenify_detector import FenifyDetector
from .llm_fen_detector import LLMFenDetector

__all__ = [
    "Camera",
    "CameraConfig",
    "CosmosPerception",
    "BoardState",
    "RemoteCosmosPerception",
    "BoardSegmentation",
    "FENDetector",
    "ChessPiece",
    "FenifyDetector",
    "LLMFenDetector",
]
