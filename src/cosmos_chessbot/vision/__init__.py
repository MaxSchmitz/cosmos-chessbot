"""Vision module: camera capture and Cosmos perception."""

from .camera import Camera, CameraConfig
from .perception import CosmosPerception, BoardState
from .remote_perception import RemoteCosmosPerception

__all__ = [
    "Camera",
    "CameraConfig",
    "CosmosPerception",
    "BoardState",
    "RemoteCosmosPerception",
]
