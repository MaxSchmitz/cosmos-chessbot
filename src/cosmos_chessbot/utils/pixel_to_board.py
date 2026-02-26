"""Pixel-to-board-plane coordinate conversion via homography.

Converts Cosmos-Reason2 normalized pixel coordinates (0-1000) to 3D
world coordinates on the board plane using four board-corner
correspondences.

Usage:
    calibration = BoardCalibration(
        pixel_corners=[(120, 150), (880, 150), (880, 850), (120, 850)],
        image_size=(1920, 1080),
    )
    world_xyz = calibration.normalized_to_board(450, 620)
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


# Board geometry constants
BOARD_SQUARE_SIZE: float = 0.05  # meters per square (default for real robot)
TABLE_HEIGHT: float = 0.0  # board surface height relative to robot base
BOARD_CENTER_OFFSET_Y: float = 0.20  # meters forward of robot


def _board_corner_world_coords(
    square_size: float = BOARD_SQUARE_SIZE,
    center_y: float = BOARD_CENTER_OFFSET_Y,
    table_z: float = TABLE_HEIGHT,
) -> np.ndarray:
    """Compute the 4 board corner world positions.

    Board center at (0, center_y, table_z).
    X = A-H direction, Y = 1-8 direction.
    Corners ordered: a1, h1, h8, a8 (bottom-left CW from white's view).

    Returns:
        (4, 3) array of [x, y, z] world coordinates.
    """
    half = 4.0 * square_size  # half-board width (4 squares)
    return np.array(
        [
            [-half, center_y - half, table_z],  # a1
            [half, center_y - half, table_z],  # h1
            [half, center_y + half, table_z],  # h8
            [-half, center_y + half, table_z],  # a8
        ],
        dtype=np.float64,
    )


def square_to_world(
    square: str,
    square_size: float = BOARD_SQUARE_SIZE,
    center_y: float = BOARD_CENTER_OFFSET_Y,
    table_z: float = TABLE_HEIGHT,
) -> tuple[float, float, float]:
    """Convert a chess square name to world coordinates.

    Args:
        square: Algebraic notation (e.g. 'e2', 'd1').
        square_size: Board square size in meters.
        center_y: Board center Y offset from robot.
        table_z: Board surface height.

    Returns:
        (x, y, z) world coordinates of the square center.
    """
    file_idx = ord(square[0].lower()) - ord('a')  # 0-7 for a-h
    rank_idx = int(square[1]) - 1                   # 0-7 for 1-8
    x = (file_idx - 3.5) * square_size
    y = (rank_idx + 0.5) * square_size
    return (x, y, table_z)


@dataclass
class BoardCalibration:
    """Homography-based pixel-to-board-plane mapping.

    Requires 4 pixel coordinates corresponding to the 4 board corners
    (a1, h1, h8, a8 in order). These can come from board segmentation
    or a manual calibration step.

    Attributes:
        pixel_corners: 4 pixel coords [(x,y), ...] for a1, h1, h8, a8
            in actual image pixels.
        image_size: (width, height) of the source image.
        square_size: Board square size in meters.
    """

    pixel_corners: list[tuple[int, int]]
    image_size: tuple[int, int] = (1920, 1080)
    square_size: float = BOARD_SQUARE_SIZE
    table_z: float = TABLE_HEIGHT
    center_y: float = BOARD_CENTER_OFFSET_Y

    # Computed in __post_init__
    _H: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if len(self.pixel_corners) != 4:
            raise ValueError("Exactly 4 board corner pixel coordinates required")

        src_pts = np.array(self.pixel_corners, dtype=np.float64)
        dst_pts = _board_corner_world_coords(
            self.square_size, center_y=self.center_y, table_z=self.table_z,
        )[:, :2]  # XY only

        self._H, _ = cv2.findHomography(src_pts, dst_pts)

    def pixel_to_board(self, px: float, py: float) -> tuple[float, float, float]:
        """Convert actual image pixel to 3D board-plane coordinates.

        Args:
            px: Pixel X in image coordinates.
            py: Pixel Y in image coordinates.

        Returns:
            (x, y, z) world coordinates on the board plane.
        """
        src = np.array([[[px, py]]], dtype=np.float64)
        dst = cv2.perspectiveTransform(src, self._H)
        wx, wy = dst[0, 0]
        return (float(wx), float(wy), self.table_z)

    def normalized_to_board(self, nx: int, ny: int) -> tuple[float, float, float]:
        """Convert Cosmos normalized coords (0-1000) to 3D board-plane.

        Cosmos-Reason2 outputs pixel coordinates normalized to 0-1000
        regardless of actual image resolution.

        Args:
            nx: Normalized X (0-1000, left to right).
            ny: Normalized Y (0-1000, top to bottom).

        Returns:
            (x, y, z) world coordinates on the board plane.
        """
        w, h = self.image_size
        px = nx * w / 1000.0
        py = ny * h / 1000.0
        return self.pixel_to_board(px, py)

    def waypoints_to_3d(
        self,
        waypoints: list,
        lift_height: float = 0.05,
    ) -> list[tuple[float, float, float]]:
        """Convert TrajectoryWaypoint objects to 3D coordinates.

        Waypoints with 'lift' or 'above' in the label get Z elevated
        by ``lift_height`` above the table surface.

        Args:
            waypoints: List of TrajectoryWaypoint (from plan_trajectory()).
            lift_height: Height above table for lift/above waypoints (meters).

        Returns:
            List of (x, y, z) world coordinates.
        """
        coords_3d = []
        for wp in waypoints:
            x, y, z = self.normalized_to_board(wp.point_2d[0], wp.point_2d[1])
            label_lower = wp.label.lower()
            if "lift" in label_lower or "above" in label_lower:
                z += lift_height
            coords_3d.append((x, y, z))
        return coords_3d
