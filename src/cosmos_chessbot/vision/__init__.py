"""Vision module: camera capture and FEN detection."""

from .camera import Camera, CameraConfig
from .hud_overlay import (
    apply_hud,
    compute_drop_zone,
    detect_corners,
    draw_hud,
    drop_zone,
    load_corners,
    resolve_location,
)
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
    "apply_hud",
    "compute_drop_zone",
    "detect_corners",
    "draw_hud",
    "drop_zone",
    "load_corners",
    "resolve_location",
]
