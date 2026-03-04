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

_DEFAULT_CORNER_WEIGHTS = "models/yolo_corners.pt"

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
    conf: float = 0.25,
) -> np.ndarray | None:
    """Detect board corners from an image using the YOLO pose model.

    If multiple detections are returned, selects the one whose keypoints
    span the largest area (most likely to be the actual board rather than
    a spurious small pattern).

    Args:
        image: HWC uint8 image (RGB or BGR).
        corner_weights: Path to YOLO pose model weights.
        conf: Minimum detection confidence threshold.

    Returns:
        (4, 2) float32 array in [TL, TR, BR, BL] order, or None if not detected.
    """
    from ultralytics import YOLO

    model = YOLO(corner_weights)

    def _detect(img):
        results = model.predict(img, verbose=False, conf=conf)
        if len(results) == 0 or results[0].keypoints is None:
            return None
        kpts = results[0].keypoints
        if len(kpts.xy) == 0:
            return None
        # Pick detection with largest keypoint bounding-box area
        best_corners = None
        best_area = 0.0
        confs = results[0].boxes.conf if results[0].boxes is not None else []
        for i in range(len(kpts.xy)):
            pts = kpts.xy[i].cpu().numpy().astype(np.float32)
            c = float(confs[i]) if i < len(confs) else 0.0
            x_range = pts[:, 0].max() - pts[:, 0].min()
            y_range = pts[:, 1].max() - pts[:, 1].min()
            area = x_range * y_range
            if area > best_area:
                best_area = area
                best_corners = pts
        return best_corners

    # Try on raw image first
    corners = _detect(image)
    if corners is not None:
        return corners

    # Retry with contrast enhancement (handles low-light conditions)
    inv_gamma = 1.0 / 0.5
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
    )
    enhanced = cv2.LUT(image, table)
    corners = _detect(enhanced)
    if corners is not None:
        print("detect_corners: detected after contrast enhancement")
    return corners


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


# ---------------------------------------------------------------------------
# Drop zone computation for captured pieces
# ---------------------------------------------------------------------------

def compute_drop_zone(
    corners: np.ndarray,
    side: str = "white",
    offset_squares: float = 1.5,
) -> tuple[int, int]:
    """Compute a drop zone pixel location outside the board edge.

    White captures go to the right side (near H-file), black captures go
    to the left side (near A-file), matching tournament convention.

    Args:
        corners: (4, 2) float array in [TL, TR, BR, BL] order.
        side: "white" (right side) or "black" (left side) -- the color
            of the captured piece being removed.
        offset_squares: How far outside the board edge, in square-widths.

    Returns:
        (x, y) integer pixel coordinates for the drop zone.
    """
    H = compute_homography(corners)
    H_inv = np.linalg.inv(H)

    if side == "white":
        # Right side: midpoint of TR-BR edge, offset further right
        board_x = 8.0 + offset_squares
        board_y = 4.0  # vertical center
    else:
        # Left side: midpoint of TL-BL edge, offset further left
        board_x = -offset_squares
        board_y = 4.0

    pt = np.array([[[board_x, board_y]]], dtype=np.float64)
    px = cv2.perspectiveTransform(pt, H_inv)[0, 0]
    return int(round(px[0])), int(round(px[1]))


def cosmos_drop_zone(
    image: np.ndarray,
    captured_piece: str,
    cosmos_url: str = "http://localhost:8000",
    timeout: float = 30.0,
) -> tuple[int, int] | None:
    """Ask Cosmos-Reason2 where to place a captured piece.

    Sends the overhead image to the Cosmos server and asks it to reason
    about the best location to place a captured piece off the board.

    Args:
        image: HWC uint8 overhead camera image.
        captured_piece: Description of the piece (e.g. "black pawn", "white knight").
        cosmos_url: URL of the Cosmos reasoning server.
        timeout: Request timeout in seconds.

    Returns:
        (x, y) pixel coordinates, or None if reasoning fails.
    """
    import base64
    import io

    import httpx
    from PIL import Image as PILImage

    # Encode image to base64 PNG
    if image.shape[2] == 3:
        pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        pil_img = PILImage.fromarray(image)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    h, w = image.shape[:2]
    prompt = f"""Looking at this overhead view of a chess board, I need to place a captured {captured_piece} off the board.

Find a good location on the table next to the board where captured pieces should go. The location should be:
- Outside the board but still on the table surface
- On the right side of the board for captured white pieces, left side for captured black pieces
- Clear of any obstacles (robot arm, other objects)
- A reasonable distance from the board edge (not too far, not too close)

The image is {w}x{h} pixels. Respond in JSON:
{{
    "x": <pixel x coordinate>,
    "y": <pixel y coordinate>,
    "reasoning": "brief explanation of why this location"
}}
"""

    try:
        # Use the action reasoning endpoint with a custom prompt
        # by calling the trajectory endpoint (it returns point_2d coords)
        resp = httpx.post(
            f"{cosmos_url}/reason/action",
            json={
                "image_base64": image_b64,
                "move_uci": "capture",
                "from_square": "board",
                "to_square": "off-board",
                "max_new_tokens": 256,
                "temperature": 0.1,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        # Parse coordinates from the reasoning text
        reasoning = data.get("reasoning", "")
        # Try to extract JSON with x,y from the response
        coord_match = re.search(r'"x"\s*:\s*(\d+).*?"y"\s*:\s*(\d+)', reasoning, re.DOTALL)
        if coord_match:
            x = int(coord_match.group(1))
            y = int(coord_match.group(2))
            # Clamp to image bounds
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            return x, y

    except Exception:
        pass

    return None


def drop_zone(
    image: np.ndarray,
    captured_piece: str = "piece",
    side: str | None = None,
    corners: np.ndarray | None = None,
    cosmos_url: str | None = None,
    corner_weights: str | Path | None = None,
) -> str:
    """Compute a drop zone for a captured piece.

    Tries Cosmos-Reason2 first (if cosmos_url is provided), falls back to
    geometric computation based on board corners.

    Args:
        image: HWC uint8 overhead camera image.
        captured_piece: Description like "black pawn" or "white knight".
        side: "white" or "black" (color of captured piece). If None,
            inferred from captured_piece string.
        corners: Pre-detected (4, 2) corners in [TL, TR, BR, BL] order.
        cosmos_url: Cosmos-Reason2 server URL. If provided, tries Cosmos first.
        corner_weights: YOLO pose weights for corner auto-detection.

    Returns:
        Location string "x,y" suitable for passing to apply_hud().
    """
    # Infer side from piece description if not given
    if side is None:
        lower = captured_piece.lower()
        if "white" in lower:
            side = "white"
        elif "black" in lower:
            side = "black"
        else:
            side = "white"  # default

    # Try Cosmos-Reason2
    if cosmos_url:
        px = cosmos_drop_zone(image, captured_piece, cosmos_url=cosmos_url)
        if px is not None:
            return f"{px[0]},{px[1]}"

    # Fall back to geometric
    if corners is None:
        corners = detect_corners(
            image,
            corner_weights=corner_weights or _DEFAULT_CORNER_WEIGHTS,
        )
    if corners is None:
        raise RuntimeError("Could not detect board corners for drop zone")

    px = compute_drop_zone(corners, side=side)
    return f"{px[0]},{px[1]}"
