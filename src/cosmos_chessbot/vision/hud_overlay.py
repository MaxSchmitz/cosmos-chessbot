"""HUD overlay for visual move encoding.

Draws source (pick) and target (place) markers on camera images so a VLA
policy can learn a single visual task: "pick highlighted, place at target."

The chess engine / perception pipeline decides *which* piece to move where;
this module just encodes that decision visually.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

# Marker style
_SOURCE_COLOR = (0, 255, 0)    # green -- same in RGB and BGR
_TARGET_COLOR = (255, 0, 255)  # magenta -- same in RGB and BGR
_OUTLINE_COLOR = (255, 255, 255)
_MARKER_RADIUS = 14
_OUTLINE_RADIUS = 17

_DEFAULT_CORNER_WEIGHTS = "runs/pose/board_corners/weights/best.pt"

_SQUARE_RE = re.compile(r"^[a-h][1-8]$")
_PIXEL_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)$")


def compute_homography(corners: np.ndarray) -> np.ndarray:
    """Compute pixel-to-board homography from 4 board corners.

    Args:
        corners: (4, 2) float array in [TL, TR, BR, BL] order (pixel coords).

    Returns:
        3x3 homography mapping pixels to canonical 8x8 board space where
        (0,0) = TL (a8) and (8,8) = BR (h1).
    """
    src = corners.reshape(4, 1, 2).astype(np.float32)
    dst = np.array([
        [0, 0],  # TL
        [8, 0],  # TR
        [8, 8],  # BR
        [0, 8],  # BL
    ], dtype=np.float32).reshape(4, 1, 2)
    H, _ = cv2.findHomography(src, dst)
    return H


def detect_corners(
    image: np.ndarray,
    corner_weights: str | Path = _DEFAULT_CORNER_WEIGHTS,
) -> np.ndarray | None:
    """Detect board corners from an image using the YOLO pose model.

    Args:
        image: HWC uint8 image (RGB or BGR).
        corner_weights: Path to YOLO pose model weights.

    Returns:
        (4, 2) float32 array in [TL, TR, BR, BL] order, or None if not detected.
    """
    from ultralytics import YOLO

    model = YOLO(corner_weights)
    results = model.predict(image, verbose=False)
    if len(results) == 0 or results[0].keypoints is None:
        return None
    kpts = results[0].keypoints
    if len(kpts.xy) == 0:
        return None
    corners = kpts.xy[0].cpu().numpy()
    return corners.astype(np.float32)


def resolve_location(
    location: str,
    corners: np.ndarray | None = None,
    homography: np.ndarray | None = None,
) -> tuple[int, int]:
    """Convert a location string to pixel coordinates.

    Args:
        location: Chess square (e.g. "e4") or pixel coords (e.g. "320,240").
        corners: (4, 2) corners in [TL, TR, BR, BL] order.  Required for
            square names (used to compute homography if not provided).
        homography: Pre-computed pixel-to-board homography.  If None and
            corners are given, computed on the fly.

    Returns:
        (x, y) integer pixel coordinates.
    """
    location = location.strip().lower()

    # Direct pixel coordinates
    m = _PIXEL_RE.match(location)
    if m:
        return int(float(m.group(1))), int(float(m.group(2)))

    # Chess square
    if not _SQUARE_RE.match(location):
        raise ValueError(f"Invalid location: {location!r}")

    if homography is None:
        if corners is None:
            raise ValueError("corners or homography required for square locations")
        homography = compute_homography(corners)

    file_idx = ord(location[0]) - ord("a")  # 0-7
    rank_idx = int(location[1]) - 1          # 0-7

    # Board-space coords (center of square)
    board_x = file_idx + 0.5
    board_y = (7 - rank_idx) + 0.5

    # Inverse homography: board -> pixel
    H_inv = np.linalg.inv(homography)
    pt = np.array([[[board_x, board_y]]], dtype=np.float64)
    px = cv2.perspectiveTransform(pt, H_inv)[0, 0]
    return int(round(px[0])), int(round(px[1]))


def load_corners(path: str | Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Load calibration corners from JSON.

    Expected keys: ``corners`` (4x2 list), ``resolution`` ([w, h]).

    Args:
        path: Path to corners JSON file.

    Returns:
        (corners, (width, height)) where corners is (4, 2) float32 array
        in [TL, TR, BR, BL] order.
    """
    with open(path) as f:
        cal = json.load(f)
    corners = np.array(cal["corners"], dtype=np.float32)
    resolution = tuple(cal["resolution"])  # (width, height)
    return corners, resolution


def draw_hud(
    image: np.ndarray,
    source_px: tuple[int, int],
    target_px: tuple[int, int],
) -> np.ndarray:
    """Draw source and target markers on an image (in-place).

    Args:
        image: HWC uint8 image (modified in place).
        source_px: (x, y) pixel location for pick marker (green).
        target_px: (x, y) pixel location for place marker (magenta).

    Returns:
        The same image array (for convenience).
    """
    # Source marker (green with white outline)
    cv2.circle(image, source_px, _OUTLINE_RADIUS, _OUTLINE_COLOR, -1, cv2.LINE_AA)
    cv2.circle(image, source_px, _MARKER_RADIUS, _SOURCE_COLOR, -1, cv2.LINE_AA)

    # Target marker (magenta with white outline)
    cv2.circle(image, target_px, _OUTLINE_RADIUS, _OUTLINE_COLOR, -1, cv2.LINE_AA)
    cv2.circle(image, target_px, _MARKER_RADIUS, _TARGET_COLOR, -1, cv2.LINE_AA)

    return image


def apply_hud(
    image: np.ndarray,
    source: str,
    target: str,
    corners: np.ndarray | None = None,
    homography: np.ndarray | None = None,
    corner_weights: str | Path | None = None,
) -> np.ndarray:
    """Resolve location strings and draw HUD markers on an image.

    Convenience wrapper around :func:`resolve_location` + :func:`draw_hud`.
    If source or target are chess square names and no corners/homography are
    provided, auto-detects corners from the image using the YOLO pose model.

    Args:
        image: HWC uint8 image (modified in place).
        source: Source location (square name or "x,y" pixels).
        target: Target location (square name or "x,y" pixels).
        corners: (4, 2) corners in [TL, TR, BR, BL] order.
        homography: Pre-computed pixel-to-board homography.
        corner_weights: Path to YOLO pose weights for auto-detection.

    Returns:
        The same image array.
    """
    # Auto-detect corners if needed for square-name locations
    needs_corners = (
        corners is None
        and homography is None
        and (_SQUARE_RE.match(source.strip().lower())
             or _SQUARE_RE.match(target.strip().lower()))
    )
    if needs_corners:
        corners = detect_corners(
            image,
            corner_weights=corner_weights or _DEFAULT_CORNER_WEIGHTS,
        )
        if corners is None:
            raise RuntimeError("Could not detect board corners from image")
        homography = compute_homography(corners)

    source_px = resolve_location(source, corners, homography)
    target_px = resolve_location(target, corners, homography)
    return draw_hud(image, source_px, target_px)
