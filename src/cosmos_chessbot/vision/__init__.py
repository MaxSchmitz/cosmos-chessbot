"""Vision module: camera capture and FEN detection."""

from .camera import Camera, CameraConfig
from .perception import CosmosPerception, BoardState
from .remote_perception import RemoteCosmosPerception
from .yolo_dino_detector import YOLODINOFenDetector

__all__ = [
    "Camera",
    "CameraConfig",
    "CosmosPerception",
    "BoardState",
    "RemoteCosmosPerception",
    "YOLODINOFenDetector",
]
