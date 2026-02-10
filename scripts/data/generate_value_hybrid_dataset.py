#!/usr/bin/env python3
"""
VALUE-based Hybrid Chess Dataset Generator

Uses VALUE's clean board.blend with:
- VALUE's simple solid-color materials (no texture issues)
- ChessR-style camera randomization
- ChessR-style HDRI lighting
- Real Lichess FEN positions

This avoids ChessR's purple texture problems while maintaining visual diversity.
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False
    # Can't use logger here as it's not configured yet
    print("Warning: pyyaml not available, using defaults")

# Add user site-packages so Blender can find installed packages (chess, etc.)
import site
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.insert(0, user_site)

# Try to import Blender
try:
    import bpy
    import bpy_extras
    import bpy_extras.object_utils
    import numpy as np
    from mathutils import Vector
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    # Can't use logger here as it's not configured yet
    print("Warning: Not running in Blender")

# Import chess for FEN parsing
try:
    import chess
    HAVE_CHESS = True
except ImportError:
    HAVE_CHESS = False
    # Can't use logger here as it's not configured yet
    print("Warning: chess library not available")


# FEN to VALUE piece mapping
FEN_TO_PIECE = {
    # White pieces (uppercase FEN -> Uppercase VALUE names)
    'P': 'P',  # Pawn
    'N': 'N',  # Knight
    'B': 'B',  # Bishop
    'R': 'R',  # Rook
    'Q': 'Q',  # Queen
    'K': 'K',  # King
    # Black pieces (lowercase FEN -> lowercase VALUE names)
    'p': 'p',
    'n': 'n',
    'b': 'b',
    'r': 'r',
    'q': 'q',
    'k': 'k',
}

# STL filename to FEN piece type mapping
STL_TO_FEN = {
    "pawn": "P", "rook": "R", "knight": "N",
    "bishop": "B", "queen": "Q", "king": "K",
}

# Piece material schemes with realistic material properties
PIECE_COLOR_SCHEMES = [
    # -- Plastic sets (tournament / club standard) --
    {
        "name": "plastic_cream_black",
        "type": "plastic",
        "white": (0.95, 0.92, 0.85, 1.0),  # Cream/ivory
        "black": (0.05, 0.05, 0.05, 1.0),  # Black
        "roughness": (0.45, 0.6),
        "metallic": 0.0,
        "specular": 0.3,
    },
    {
        "name": "plastic_cream_brown",
        "type": "plastic",
        "white": (0.95, 0.9, 0.8, 1.0),    # Cream
        "black": (0.45, 0.3, 0.2, 1.0),    # Brown
        "roughness": (0.4, 0.6),
        "metallic": 0.0,
        "specular": 0.3,
    },
    # -- Wood sets (the most common real-world material) --
    # white_texture/black_texture reference dirs in data/textures/pieces/
    {
        "name": "wood_boxwood_sheesham",
        "type": "wood",
        "white": (0.82, 0.72, 0.52, 1.0),  # Boxwood (pale golden)
        "black": (0.42, 0.25, 0.15, 1.0),  # Sheesham/rosewood (warm brown)
        "white_texture": "boxwood",
        "black_texture": "walnut",
        "roughness": (0.5, 0.7),
        "metallic": 0.0,
        "specular": 0.4,
    },
    {
        "name": "wood_boxwood_ebonized",
        "type": "wood",
        "white": (0.82, 0.72, 0.52, 1.0),  # Boxwood
        "black": (0.08, 0.07, 0.06, 1.0),  # Ebonized boxwood (near-black)
        "white_texture": "boxwood",
        "black_texture": "rosewood",
        "roughness": (0.5, 0.7),
        "metallic": 0.0,
        "specular": 0.4,
    },
    {
        "name": "wood_maple_walnut",
        "type": "wood",
        "white": (0.85, 0.75, 0.6, 1.0),   # Maple (light)
        "black": (0.32, 0.22, 0.15, 1.0),  # Walnut (dark brown)
        "white_texture": "maple",
        "black_texture": "walnut",
        "roughness": (0.5, 0.7),
        "metallic": 0.0,
        "specular": 0.4,
    },
    {
        "name": "wood_boxwood_rosewood",
        "type": "wood",
        "white": (0.80, 0.70, 0.50, 1.0),  # Boxwood (slightly darker)
        "black": (0.35, 0.18, 0.12, 1.0),  # Rosewood (deep reddish-brown)
        "white_texture": "boxwood",
        "black_texture": "rosewood",
        "roughness": (0.45, 0.65),
        "metallic": 0.0,
        "specular": 0.45,
    },
    {
        "name": "wood_ebony_boxwood",
        "type": "wood",
        "white": (0.88, 0.78, 0.58, 1.0),  # Light boxwood
        "black": (0.12, 0.10, 0.08, 1.0),  # Ebony (very dark)
        "white_texture": "maple",
        "black_texture": "rosewood",
        "roughness": (0.45, 0.65),
        "metallic": 0.0,
        "specular": 0.45,
    },
]

BOARD_MATERIALS = [
    # Classic black and white - pure monochrome
    {
        "name": "classic_white_black",
        "type": "classic",
        "light": (0.98, 0.98, 0.98, 1.0),  # Very white
        "dark": (0.02, 0.02, 0.02, 1.0),   # Very black
        "roughness": (0.3, 0.5),
        "metallic": 0.0,
        "specular": 0.5,
    },
    {
        "name": "classic_cream_black",
        "type": "classic",
        "light": (0.9, 0.88, 0.82, 1.0),   # Cream
        "dark": (0.05, 0.05, 0.05, 1.0),   # Very dark
        "roughness": (0.3, 0.5),
        "metallic": 0.0,
        "specular": 0.5,
    },
    # Classic wood boards - traditional chess boards
    {
        "name": "maple_walnut",
        "type": "wood",
        "light": (0.9, 0.85, 0.7, 1.0),  # Light maple
        "dark": (0.4, 0.3, 0.2, 1.0),    # Dark walnut
        "roughness": (0.4, 0.6),
        "metallic": 0.0,
        "specular": 0.5,
    },
    {
        "name": "oak_rosewood",
        "type": "wood",
        "light": (0.8, 0.7, 0.5, 1.0),   # Light oak
        "dark": (0.3, 0.2, 0.15, 1.0),   # Dark rosewood
        "roughness": (0.3, 0.5),
        "metallic": 0.0,
        "specular": 0.5,
    },
    {
        "name": "birch_mahogany",
        "type": "wood",
        "light": (0.85, 0.75, 0.6, 1.0), # Light birch
        "dark": (0.5, 0.25, 0.2, 1.0),   # Dark mahogany
        "roughness": (0.4, 0.6),
        "metallic": 0.0,
        "specular": 0.5,
    },
    # Marble/Stone boards - luxury boards
    {
        "name": "white_marble_black_granite",
        "type": "stone",
        "light": (0.95, 0.95, 0.93, 1.0), # White marble
        "dark": (0.02, 0.02, 0.02, 1.0),  # Black granite
        "roughness": (0.1, 0.3),
        "metallic": 0.0,
        "specular": 0.7,
    },
    {
        "name": "cream_marble_gray_stone",
        "type": "stone",
        "light": (0.9, 0.88, 0.85, 1.0),  # Cream marble
        "dark": (0.4, 0.4, 0.4, 1.0),     # Gray stone
        "roughness": (0.15, 0.35),
        "metallic": 0.0,
        "specular": 0.65,
    },
    # Plastic/Vinyl boards - tournament and casual
    {
        "name": "tournament_green",
        "type": "plastic",
        "light": (0.9, 0.9, 0.85, 1.0),   # Cream
        "dark": (0.25, 0.45, 0.25, 1.0),  # Forest green
        "roughness": (0.5, 0.7),
        "metallic": 0.0,
        "specular": 0.4,
    },
    {
        "name": "tournament_blue",
        "type": "plastic",
        "light": (0.9, 0.9, 0.9, 1.0),    # White
        "dark": (0.2, 0.3, 0.55, 1.0),    # Navy blue
        "roughness": (0.5, 0.7),
        "metallic": 0.0,
        "specular": 0.4,
    },
    {
        "name": "tournament_brown",
        "type": "plastic",
        "light": (0.85, 0.8, 0.7, 1.0),   # Tan
        "dark": (0.35, 0.25, 0.2, 1.0),   # Brown
        "roughness": (0.5, 0.7),
        "metallic": 0.0,
        "specular": 0.4,
    },
    # Painted/Lacquered boards - colorful modern boards
    {
        "name": "red_black_glossy",
        "type": "painted",
        "light": (0.85, 0.2, 0.2, 1.0),   # Red
        "dark": (0.02, 0.02, 0.02, 1.0),  # Black
        "roughness": (0.2, 0.4),
        "metallic": 0.0,
        "specular": 0.6,
    },
    {
        "name": "blue_white_glossy",
        "type": "painted",
        "light": (0.9, 0.9, 0.95, 1.0),   # Light blue-white
        "dark": (0.2, 0.35, 0.65, 1.0),   # Royal blue
        "roughness": (0.2, 0.4),
        "metallic": 0.0,
        "specular": 0.6,
    },
    {
        "name": "green_cream",
        "type": "painted",
        "light": (0.9, 0.88, 0.8, 1.0),   # Cream
        "dark": (0.3, 0.5, 0.3, 1.0),     # Green
        "roughness": (0.3, 0.5),
        "metallic": 0.0,
        "specular": 0.5,
    },
    # Gray/Neutral boards - modern minimalist
    {
        "name": "light_gray_dark_gray",
        "type": "modern",
        "light": (0.75, 0.75, 0.75, 1.0), # Light gray
        "dark": (0.25, 0.25, 0.25, 1.0),  # Dark gray
        "roughness": (0.3, 0.5),
        "metallic": 0.0,
        "specular": 0.5,
    },
    {
        "name": "beige_charcoal",
        "type": "modern",
        "light": (0.8, 0.75, 0.7, 1.0),   # Beige
        "dark": (0.2, 0.2, 0.2, 1.0),     # Charcoal
        "roughness": (0.4, 0.6),
        "metallic": 0.0,
        "specular": 0.5,
    },
]

TABLE_MATERIALS = [
    # Wood materials - natural finish
    {
        "name": "oak",
        "type": "wood",
        "color": (0.6, 0.45, 0.3, 1.0),
        "roughness": (0.4, 0.6),
        "metallic": 0.0,
        "specular": 0.5,
    },
    {
        "name": "walnut",
        "type": "wood",
        "color": (0.4, 0.3, 0.25, 1.0),
        "roughness": (0.4, 0.6),
        "metallic": 0.0,
        "specular": 0.5,
    },
    {
        "name": "mahogany",
        "type": "wood",
        "color": (0.5, 0.25, 0.2, 1.0),
        "roughness": (0.3, 0.5),
        "metallic": 0.0,
        "specular": 0.5,
    },
    {
        "name": "pine",
        "type": "wood",
        "color": (0.7, 0.55, 0.35, 1.0),
        "roughness": (0.5, 0.7),
        "metallic": 0.0,
        "specular": 0.4,
    },
    # Metal materials - modern/industrial
    {
        "name": "brushed_steel",
        "type": "metal",
        "color": (0.6, 0.6, 0.6, 1.0),
        "roughness": (0.2, 0.3),
        "metallic": 1.0,
        "specular": 0.5,
    },
    {
        "name": "polished_aluminum",
        "type": "metal",
        "color": (0.75, 0.75, 0.75, 1.0),
        "roughness": (0.05, 0.15),
        "metallic": 1.0,
        "specular": 0.5,
    },
    {
        "name": "brushed_brass",
        "type": "metal",
        "color": (0.7, 0.6, 0.3, 1.0),
        "roughness": (0.2, 0.3),
        "metallic": 1.0,
        "specular": 0.5,
    },
    # Stone materials - elegant/classic
    {
        "name": "white_marble",
        "type": "stone",
        "color": (0.9, 0.9, 0.88, 1.0),
        "roughness": (0.2, 0.4),
        "metallic": 0.0,
        "specular": 0.6,
    },
    {
        "name": "black_granite",
        "type": "stone",
        "color": (0.15, 0.15, 0.15, 1.0),
        "roughness": (0.1, 0.3),
        "metallic": 0.0,
        "specular": 0.7,
    },
    {
        "name": "gray_stone",
        "type": "stone",
        "color": (0.5, 0.5, 0.5, 1.0),
        "roughness": (0.3, 0.5),
        "metallic": 0.0,
        "specular": 0.5,
    },
    # Plastic/Painted - modern/colorful
    {
        "name": "white_plastic",
        "type": "plastic",
        "color": (0.95, 0.95, 0.95, 1.0),
        "roughness": (0.3, 0.4),
        "metallic": 0.0,
        "specular": 0.5,
    },
    {
        "name": "black_plastic",
        "type": "plastic",
        "color": (0.1, 0.1, 0.1, 1.0),
        "roughness": (0.3, 0.4),
        "metallic": 0.0,
        "specular": 0.5,
    },
    {
        "name": "red_painted",
        "type": "painted",
        "color": (0.7, 0.15, 0.15, 1.0),
        "roughness": (0.4, 0.6),
        "metallic": 0.0,
        "specular": 0.4,
    },
    {
        "name": "blue_painted",
        "type": "painted",
        "color": (0.2, 0.3, 0.6, 1.0),
        "roughness": (0.4, 0.6),
        "metallic": 0.0,
        "specular": 0.4,
    },
    {
        "name": "green_painted",
        "type": "painted",
        "color": (0.3, 0.5, 0.3, 1.0),
        "roughness": (0.4, 0.6),
        "metallic": 0.0,
        "specular": 0.4,
    },
]

# Floor/Ground materials for background variety
# Piece-specific collision radii (in meters)
PIECE_COLLISION_RADII = {
    'P': 0.020, 'p': 0.020,  # Pawns
    'N': 0.025, 'n': 0.025,  # Knights
    'B': 0.027, 'b': 0.027,  # Bishops
    'R': 0.028, 'r': 0.028,  # Rooks
    'Q': 0.035, 'q': 0.035,  # Queens
    'K': 0.035, 'k': 0.035,  # Kings
}

# Camera variation profiles for diverse training data
# Default uses 'standard' profile (100% weight) matching original settings
CAMERA_PROFILES = {
    'standard': {
        'distance': (1.5, 2.0),        # SO-101 head camera height
        'theta': (0, 45),              # 0-45° from vertical
        'focal_length': (30, 50),      # Phone camera simulation
        'weight': 1.0                  # 100% - default profile
    },
    'close_detail': {
        'distance': (0.8, 1.2),
        'theta': (15, 35),
        'focal_length': (45, 65),
        'weight': 0.0                  # Disabled by default
    },
    'wide_overview': {
        'distance': (2.2, 3.0),
        'theta': (20, 50),
        'focal_length': (25, 35),
        'weight': 0.0                  # Disabled by default
    },
    'low_angle': {
        'distance': (1.2, 1.8),
        'theta': (55, 75),
        'focal_length': (40, 55),
        'weight': 0.0                  # Disabled by default
    },
}

FLOOR_MATERIALS = [
    # Wood floors
    {
        "name": "oak_floor",
        "type": "wood",
        "color": (0.6, 0.5, 0.35, 1.0),
        "roughness": (0.3, 0.5),
        "metallic": 0.0,
        "specular": 0.4,
    },
    {
        "name": "dark_wood_floor",
        "type": "wood",
        "color": (0.3, 0.2, 0.15, 1.0),
        "roughness": (0.2, 0.4),
        "metallic": 0.0,
        "specular": 0.5,
    },
    # Carpet/Fabric
    {
        "name": "beige_carpet",
        "type": "carpet",
        "color": (0.7, 0.65, 0.55, 1.0),
        "roughness": (0.8, 0.95),
        "metallic": 0.0,
        "specular": 0.1,
    },
    {
        "name": "gray_carpet",
        "type": "carpet",
        "color": (0.4, 0.4, 0.4, 1.0),
        "roughness": (0.85, 0.95),
        "metallic": 0.0,
        "specular": 0.1,
    },
    {
        "name": "blue_carpet",
        "type": "carpet",
        "color": (0.2, 0.3, 0.5, 1.0),
        "roughness": (0.85, 0.95),
        "metallic": 0.0,
        "specular": 0.1,
    },
    # Tile/Stone
    {
        "name": "white_tile",
        "type": "tile",
        "color": (0.95, 0.95, 0.95, 1.0),
        "roughness": (0.1, 0.2),
        "metallic": 0.0,
        "specular": 0.6,
    },
    {
        "name": "gray_tile",
        "type": "tile",
        "color": (0.5, 0.5, 0.5, 1.0),
        "roughness": (0.1, 0.25),
        "metallic": 0.0,
        "specular": 0.5,
    },
    {
        "name": "black_tile",
        "type": "tile",
        "color": (0.1, 0.1, 0.1, 1.0),
        "roughness": (0.05, 0.15),
        "metallic": 0.0,
        "specular": 0.7,
    },
    # Concrete
    {
        "name": "light_concrete",
        "type": "concrete",
        "color": (0.6, 0.6, 0.6, 1.0),
        "roughness": (0.6, 0.8),
        "metallic": 0.0,
        "specular": 0.2,
    },
    {
        "name": "dark_concrete",
        "type": "concrete",
        "color": (0.3, 0.3, 0.3, 1.0),
        "roughness": (0.65, 0.85),
        "metallic": 0.0,
        "specular": 0.2,
    },
]


class DatasetConfig:
    """Configuration for dataset generation."""

    def __init__(self, config_path: Path = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "dataset_generation.yaml"

        if HAVE_YAML and config_path.exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration if YAML not available or file missing
            self.config = {
                'render': {
                    'resolution': 640,
                    'samples': 256,
                    'device': 'CPU',
                    'engine': 'CYCLES'
                },
                'camera': {
                    'distance_range': [1.5, 2.0],
                    'theta_range': [0, 45],
                    'phi_range': [0, 360],
                    'focal_length_range': [30, 50]
                },
                'placement': {
                    'neat_grid': 0.25,
                    'scattered': 0.35,
                    'piled': 0.30,
                    'hidden': 0.10
                },
                'collision': {
                    'pawn': 0.020,
                    'knight': 0.025,
                    'bishop': 0.027,
                    'rook': 0.028,
                    'queen': 0.035,
                    'king': 0.035
                },
                'paths': {
                    'textures_dir': 'data/textures/floors',
                    'fen_source': 'data/value_llava/chess_fen_train.json',
                    'output_dir': 'data/chess',
                    'hdri_dir': None
                }
            }

    @property
    def resolution(self) -> int:
        return self.config['render']['resolution']

    @property
    def samples(self) -> int:
        return self.config['render']['samples']

    @property
    def camera_distance_range(self) -> tuple[float, float]:
        return tuple(self.config['camera']['distance_range'])

    @property
    def camera_theta_range(self) -> tuple[float, float]:
        return tuple(self.config['camera']['theta_range'])

    @property
    def camera_phi_range(self) -> tuple[float, float]:
        return tuple(self.config['camera']['phi_range'])

    @property
    def camera_focal_length_range(self) -> tuple[float, float]:
        return tuple(self.config['camera']['focal_length_range'])

    @property
    def placement_weights(self) -> dict:
        return self.config['placement']

    @property
    def collision_radii(self) -> dict:
        return self.config['collision']


# Chess square positions (A1 = bottom-left from white's perspective)
# VALUE board squares are positioned in a grid
def get_square_position(square_name: str, board_center: Vector, square_size: float) -> Vector:
    """
    Get 3D position for a chess square on VALUE board.

    Args:
        square_name: e.g., "A1", "E4"
        board_center: Center of the board
        square_size: Size of each square

    Returns:
        3D position vector
    """
    file_letter = square_name[0]
    rank_number = square_name[1]

    # Convert to 0-7 indices
    file_idx = ord(file_letter) - ord('A')  # 0=A, 7=H
    rank_idx = int(rank_number) - 1  # 0=1, 7=8

    # Calculate position (centered on board)
    # X increases from A to H (left to right)
    # Y increases from 1 to 8 (bottom to top from white's view)
    x_offset = (file_idx - 3.5) * square_size
    y_offset = (rank_idx - 3.5) * square_size

    return Vector((
        board_center.x + x_offset,
        board_center.y + y_offset,
        board_center.z
    ))


class VALUEHybridGenerator:
    """
    Generate chess position dataset using VALUE board with visual randomization.

    This generator:
    - Loads FEN positions from JSON
    - Randomizes piece, board, table, and floor materials
    - Randomizes camera position and focal length
    - Places captured pieces realistically on the table
    - Renders to high-quality JPEG images
    - Outputs Llava-format annotations for training

    The generator supports automatic resume from previous runs and saves
    annotations incrementally to prevent data loss.

    Attributes:
        fen_source: Path to input FEN JSON file
        output_dir: Directory for rendered images
        max_positions: Maximum number of positions to render
        hdri_dir: Optional directory containing HDRI environment maps
        config: Configuration object controlling render settings

    Example:
        >>> generator = VALUEHybridGenerator(
        ...     fen_source=Path("data/chess_fen.json"),
        ...     output_dir=Path("data/renders"),
        ...     max_positions=1000
        ... )
        >>> generator.generate()
    """

    def _validate_scene_objects(self) -> None:
        """
        Validate required objects exist in scene.

        Raises:
            ValueError: If any required scene objects or materials are missing
        """
        required_objects = ['Camera', 'chessBoard']
        required_materials = ['WhitePieces', 'BlackPieces', 'BoardWhite', 'BoardBlack', 'TableTop', 'Floor2']

        missing = []
        for obj_name in required_objects:
            if obj_name not in bpy.data.objects:
                missing.append(f"Object: {obj_name}")

        for mat_name in required_materials:
            if mat_name not in bpy.data.materials:
                missing.append(f"Material: {mat_name}")

        if missing:
            raise ValueError(f"Required scene elements missing: {', '.join(missing)}")

    def __init__(
        self,
        fen_source: Path,
        output_dir: Path,
        max_positions: int = 10000,
        hdri_dir: Path = None,
        config: DatasetConfig = None,
    ):
        """
        Initialize generator.

        Args:
            fen_source: Path to Llava JSON with FEN positions
            output_dir: Output directory
            max_positions: Number of positions to render
            hdri_dir: Directory containing HDRI files (optional)
            config: Configuration object (optional, creates default if None)

        Raises:
            RuntimeError: If not running inside Blender
            FileNotFoundError: If fen_source doesn't exist
            ValueError: If max_positions is not a positive integer
            NotADirectoryError: If hdri_dir is provided but doesn't exist
        """
        if not IN_BLENDER:
            raise RuntimeError("Must run inside Blender")

        # Validate inputs
        if not fen_source.exists():
            raise FileNotFoundError(f"FEN source not found: {fen_source}")
        if not isinstance(max_positions, int) or max_positions <= 0:
            raise ValueError(f"max_positions must be positive int, got {max_positions}")
        if hdri_dir and not hdri_dir.is_dir():
            raise NotADirectoryError(f"HDRI directory not found: {hdri_dir}")

        # Validate scene objects and materials exist
        self._validate_scene_objects()

        self.fen_source = fen_source
        self.output_dir = output_dir
        self.max_positions = max_positions
        self.hdri_dir = hdri_dir

        # Load or use provided config
        self.config = config if config is not None else DatasetConfig()

        # Load placement weights from config
        self.placement_weights = {
            'neat_grid': self.config.placement_weights.get('neat_grid', 0.25),
            'scattered': self.config.placement_weights.get('scattered', 0.35),
            'piled': self.config.placement_weights.get('piled', 0.30),
            'hidden': self.config.placement_weights.get('hidden', 0.10),
        }

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set render settings from config
        scene = bpy.context.scene
        resolution = self.config.resolution
        samples = self.config.samples
        scene.render.resolution_x = resolution
        scene.render.resolution_y = resolution
        scene.render.resolution_percentage = 100
        scene.cycles.samples = samples

        # Use CPU rendering for best performance on M1 Macs
        # (Metal GPU is 30x slower for Cycles path tracing on Apple Silicon)
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'CPU'
        logger.info(f"Render settings: {resolution}x{resolution}, {samples} samples, device: CPU")

        # Get scene objects
        self.camera = bpy.data.objects['Camera']
        self.board = bpy.data.objects['chessBoard']

        # Remove/disable camera constraints that would override our positioning
        logger.debug(f"Camera constraints before: {len(self.camera.constraints)}")
        for constraint in self.camera.constraints:
            logger.debug(f"  Removing constraint: {constraint.name} ({constraint.type})")
            self.camera.constraints.remove(constraint)
        logger.debug(f"Camera constraints after: {len(self.camera.constraints)}")

        # Hide chairs (if present)
        self.hide_chairs()

        # Get board dimensions
        self.board_center = self.board.location
        self.square_size = 0.106768  # From VALUE config

        # Get table surface height (for placing captured pieces)
        # Board thickness is Y dimension (0.012m = 1.2cm)
        # Pieces on board sit at board_center.z (0) and appear to float by board thickness
        # So table surface is below board center by the board thickness

        board_thickness_y = self.board.dimensions.y
        logger.debug(f"Board location: ({self.board.location.x:.4f}, {self.board.location.y:.4f}, {self.board.location.z:.4f})m")
        logger.debug(f"Board thickness (Y): {board_thickness_y:.4f}m")

        # Table is below the board by the board thickness
        self.table_surface_z = self.board_center.z - board_thickness_y
        logger.debug(f"Board surface Z: {self.board_center.z:.4f}m")
        logger.debug(f"Table surface Z: {self.table_surface_z:.4f}m (board - thickness)")

        # Get all piece objects (2-char names)
        self.all_pieces = {
            obj.name: obj
            for obj in bpy.data.objects
            if len(obj.name) == 2 and obj.type == 'MESH'
        }

        # Organize pieces by type and color
        self.available_pieces = {
            'P': [], 'N': [], 'B': [], 'R': [], 'Q': [], 'K': [],
            'p': [], 'n': [], 'b': [], 'r': [], 'q': [], 'k': [],
        }

        for name, obj in self.all_pieces.items():
            piece_type = name[0]
            if piece_type in self.available_pieces:
                self.available_pieces[piece_type].append(obj)

        logger.info(f"Found {len(self.all_pieces)} piece objects")
        for piece_type, pieces in self.available_pieces.items():
            logger.debug(f"  {piece_type}: {len(pieces)} pieces")

        # Load piece mesh sets from data/piece_meshes/
        self.piece_mesh_sets = self._load_piece_mesh_sets()

        # Load HDRIs if available
        self.hdris = []
        if hdri_dir and hdri_dir.exists():
            for hdr_file in hdri_dir.glob("*.hdr"):
                img = bpy.data.images.load(str(hdr_file), check_existing=True)
                self.hdris.append(img)
            logger.info(f"Loaded {len(self.hdris)} HDRI files")

        # Discover and cache floor textures
        self.floor_textures = self._discover_floor_textures()

    def _discover_floor_textures(self) -> list[tuple[str, str]]:
        """Discover available floor textures at startup."""
        textures_dir = Path(__file__).parent.parent.parent / "data" / "textures" / "floors"

        textures = []
        if textures_dir.exists():
            for img_file in sorted(textures_dir.glob("*.jpg")):
                name = img_file.stem.replace('_', ' ').title()
                textures.append((img_file.name, name))

        if not textures:
            logger.warning(f"No floor textures found in {textures_dir}")
            # Use fallback solid colors
            textures = [("solid_gray", "Solid Gray")]

        logger.info(f"Loaded {len(textures)} floor textures")
        return textures

    def _load_piece_mesh_sets(self) -> dict:
        """Load piece mesh sets from data/piece_meshes/ and the existing VALUE pieces.

        Returns:
            Dict mapping set name to dict of piece type -> mesh data, e.g.:
            {"value": {"P": Mesh, "N": Mesh, ...}, "staunton": {...}, ...}
        """
        meshes_dir = Path(__file__).parent.parent.parent / "data" / "piece_meshes"

        # Record reference heights from VALUE pieces (already in the blend file)
        # VALUE pieces are Y-up (Y is the height/vertical axis)
        ref_objects = {'P': 'P0', 'N': 'N0', 'B': 'B0', 'R': 'R0', 'Q': 'Q0', 'K': 'K0'}
        reference_heights = {}
        for piece_char, obj_name in ref_objects.items():
            if obj_name in bpy.data.objects:
                obj = bpy.data.objects[obj_name]
                bbox = [Vector(c) for c in obj.bound_box]
                height = max(c.y for c in bbox) - min(c.y for c in bbox)
                reference_heights[piece_char] = height
                logger.debug(f"  Reference {piece_char} height (Y-up): {height:.4f}m")

        if not reference_heights:
            logger.warning("No reference piece objects found, skipping mesh set loading")
            return {}

        # Store VALUE set from existing blend file objects
        sets = {}
        value_meshes = {}
        for piece_char, obj_name in ref_objects.items():
            if obj_name in bpy.data.objects:
                value_meshes[piece_char] = bpy.data.objects[obj_name].data
        if value_meshes:
            sets["value"] = value_meshes
            logger.info(f"Registered 'value' piece set (from blend file)")

        # Import STL sets
        if not meshes_dir.exists():
            logger.warning(f"Piece meshes directory not found: {meshes_dir}")
            return sets

        for set_dir in sorted(meshes_dir.iterdir()):
            if not set_dir.is_dir():
                continue
            set_name = set_dir.name
            if set_name == "value":
                continue  # Already registered from blend file

            set_meshes = {}
            for stl_file in sorted(set_dir.glob("*.stl")):
                piece_name = stl_file.stem.lower()
                piece_char = STL_TO_FEN.get(piece_name)
                if not piece_char:
                    logger.warning(f"  Unknown piece file: {stl_file.name}")
                    continue
                if piece_char not in reference_heights:
                    continue

                # Import STL -- creates a temporary object
                bpy.ops.wm.stl_import(filepath=str(stl_file))
                tmp_obj = bpy.context.selected_objects[0]
                mesh = tmp_obj.data

                # Fix orientation: VALUE pieces are Y-up, so rotate imported
                # meshes so that the tallest axis becomes Y
                bbox = [Vector(c) for c in tmp_obj.bound_box]
                extents = {
                    'x': max(c.x for c in bbox) - min(c.x for c in bbox),
                    'y': max(c.y for c in bbox) - min(c.y for c in bbox),
                    'z': max(c.z for c in bbox) - min(c.z for c in bbox),
                }
                tallest = max(extents, key=extents.get)

                if tallest == 'z':
                    # Z-up -> Y-up: rotate 90 degrees around X
                    for v in mesh.vertices:
                        old_y, old_z = v.co.y, v.co.z
                        v.co.y = old_z
                        v.co.z = -old_y
                    logger.debug(f"  {piece_name}: rotated Z-up -> Y-up")
                elif tallest == 'x':
                    # X-up -> Y-up: rotate -90 degrees around Z
                    for v in mesh.vertices:
                        old_x, old_y = v.co.x, v.co.y
                        v.co.x = old_y
                        v.co.y = old_x
                    logger.debug(f"  {piece_name}: rotated X-up -> Y-up")

                # Recompute bbox after rotation (Y is now height)
                xs = [v.co.x for v in mesh.vertices]
                ys = [v.co.y for v in mesh.vertices]
                zs = [v.co.z for v in mesh.vertices]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                min_z, max_z = min(zs), max(zs)
                imported_height = max_y - min_y

                # Auto-scale to match VALUE piece dimensions
                if imported_height > 0:
                    scale = reference_heights[piece_char] / imported_height
                    cx = (min_x + max_x) / 2
                    cz = (min_z + max_z) / 2

                    for v in mesh.vertices:
                        v.co.x = (v.co.x - cx) * scale
                        v.co.y = (v.co.y - min_y) * scale  # bottom at y=0
                        v.co.z = (v.co.z - cz) * scale

                    logger.debug(f"  {piece_name}: {imported_height:.4f}m -> "
                                 f"{reference_heights[piece_char]:.4f}m (scale {scale:.3f}x)")

                # Rename mesh for clarity
                mesh.name = f"mesh_{set_name}_{piece_name}"

                # Delete the temporary object (keep the mesh data)
                bpy.data.objects.remove(tmp_obj, do_unlink=True)

                set_meshes[piece_char] = mesh

            if len(set_meshes) == 6:
                sets[set_name] = set_meshes
                logger.info(f"Loaded '{set_name}' piece set ({len(set_meshes)} pieces)")
            else:
                missing = [p for p in "PNBRQK" if p not in set_meshes]
                logger.warning(f"Incomplete set '{set_name}': missing {missing}, skipping")

        logger.info(f"Total piece mesh sets: {len(sets)} ({', '.join(sets.keys())})")
        return sets

    def _select_piece_set(self, set_name: str):
        """Swap all piece objects to use meshes from the given set.

        Uses object-level material linking so white and black pieces can
        share the same mesh data with different materials.
        """
        if set_name not in self.piece_mesh_sets:
            return

        white_mat = bpy.data.materials.get('WhitePieces')
        black_mat = bpy.data.materials.get('BlackPieces')

        meshes = self.piece_mesh_sets[set_name]
        for piece_char, mesh in meshes.items():
            # Ensure mesh has at least one material slot
            if not mesh.materials:
                mesh.materials.append(None)

            # Apply to white pieces (uppercase)
            for obj in self.available_pieces.get(piece_char, []):
                obj.data = mesh
                if white_mat:
                    if not obj.material_slots:
                        obj.data.materials.append(None)
                    obj.material_slots[0].link = 'OBJECT'
                    obj.material_slots[0].material = white_mat

            # Apply to black pieces (lowercase)
            for obj in self.available_pieces.get(piece_char.lower(), []):
                obj.data = mesh
                if black_mat:
                    if not obj.material_slots:
                        obj.data.materials.append(None)
                    obj.material_slots[0].link = 'OBJECT'
                    obj.material_slots[0].material = black_mat

    def disconnect_socket(self, node_tree, target_node, socket_name) -> int:
        """Disconnect all links to a specific socket.

        Returns:
            Number of links disconnected
        """
        links = [link for link in node_tree.links
                 if link.to_node == target_node and link.to_socket.name == socket_name]
        for link in links:
            node_tree.links.remove(link)
        return len(links)

    def apply_material_properties(self, bsdf, scheme, roughness_value):
        """Apply material properties to BSDF node."""
        if not bsdf:
            return
        bsdf.inputs['Roughness'].default_value = roughness_value
        bsdf.inputs['Metallic'].default_value = scheme['metallic']
        bsdf.inputs['Specular IOR Level'].default_value = scheme['specular']

    def _apply_wood_texture(self, material, texture_name):
        """Apply an image-based wood texture to a piece material's Base Color.

        Creates/reuses 'Wood Texture' and 'Wood Mapping' nodes.
        Uses Box projection with Object coordinates for proper 3D mapping
        without UV maps.
        """
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        bsdf = nodes.get('Principled BSDF')
        if not bsdf:
            return False

        tex_dir = Path(__file__).parent.parent.parent / "data" / "textures" / "pieces" / texture_name
        diff_path = None
        for name in ('diff.jpg', 'diff.png'):
            p = tex_dir / name
            if p.exists():
                diff_path = p
                break
        if not diff_path:
            logger.warning(f"Wood texture not found: {texture_name}")
            return False

        # Get or create Texture Coordinate node
        tex_coord = nodes.get('Texture Coordinate')
        if not tex_coord:
            tex_coord = nodes.new('ShaderNodeTexCoord')
            tex_coord.name = 'Texture Coordinate'

        # Dedicated Mapping node for wood texture (don't disturb noise texture mapping)
        wood_mapping = nodes.get('Wood Mapping')
        if not wood_mapping:
            wood_mapping = nodes.new('ShaderNodeMapping')
            wood_mapping.name = 'Wood Mapping'
        wood_mapping.inputs['Scale'].default_value = (15.0, 15.0, 15.0)
        # Randomize offset so each render gets different grain
        wood_mapping.inputs['Location'].default_value = (
            np.random.uniform(0, 10),
            np.random.uniform(0, 10),
            np.random.uniform(0, 10),
        )

        # Image Texture node with Box projection (avoids stretching without UVs)
        img_tex = nodes.get('Wood Texture')
        if not img_tex:
            img_tex = nodes.new('ShaderNodeTexImage')
            img_tex.name = 'Wood Texture'
        img_tex.projection = 'BOX'
        img_tex.projection_blend = 0.2
        img_tex.image = bpy.data.images.load(str(diff_path), check_existing=True)

        # Connect: TexCoord.Object -> Wood Mapping -> Wood Texture -> BSDF.Base Color
        links.new(tex_coord.outputs['Object'], wood_mapping.inputs['Vector'])
        links.new(wood_mapping.outputs['Vector'], img_tex.inputs['Vector'])
        self.disconnect_socket(material.node_tree, bsdf, 'Base Color')
        links.new(img_tex.outputs['Color'], bsdf.inputs['Base Color'])

        logger.debug(f"  Applied wood texture: {texture_name}")
        return True

    def _reset_piece_material(self, material, color, is_white_mat):
        """Reset piece material to solid color (for plastic schemes).

        Disconnects any wood texture from Base Color and restores the original
        connection (ColorRamp for white, direct value for black).
        """
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        bsdf = nodes.get('Principled BSDF')
        if not bsdf:
            return

        self.disconnect_socket(material.node_tree, bsdf, 'Base Color')

        if is_white_mat:
            # Reconnect existing ColorRamp -> Base Color
            colorramp = nodes.get('ColorRamp')
            if colorramp:
                for elem in colorramp.color_ramp.elements:
                    elem.color = color
                links.new(colorramp.outputs['Color'], bsdf.inputs['Base Color'])
        else:
            bsdf.inputs['Base Color'].default_value = color

    def randomize_materials(self):
        """Randomize piece, board, and table materials."""
        # Randomize piece materials with realistic properties
        piece_scheme = np.random.choice(PIECE_COLOR_SCHEMES)
        logger.debug(f"Selected piece material: {piece_scheme['name']} ({piece_scheme['type']})")

        white_pieces_mat = bpy.data.materials.get('WhitePieces')
        black_pieces_mat = bpy.data.materials.get('BlackPieces')

        # Randomize material properties within the scheme's range
        roughness_min, roughness_max = piece_scheme['roughness']
        piece_roughness = np.random.uniform(roughness_min, roughness_max)

        is_wood = piece_scheme['type'] == 'wood'

        if white_pieces_mat and white_pieces_mat.node_tree:
            bsdf = white_pieces_mat.node_tree.nodes.get('Principled BSDF')

            # Disconnect Mix node from Specular Tint (causes color artifacts)
            count = self.disconnect_socket(white_pieces_mat.node_tree, bsdf, 'Specular Tint')
            if count > 0:
                logger.debug(f"Disconnected {count} link(s) from Specular Tint")

            if is_wood and piece_scheme.get('white_texture'):
                self._apply_wood_texture(white_pieces_mat, piece_scheme['white_texture'])
            else:
                self._reset_piece_material(white_pieces_mat, piece_scheme['white'], is_white_mat=True)

            if bsdf:
                self.apply_material_properties(bsdf, piece_scheme, piece_roughness)

        if black_pieces_mat and black_pieces_mat.node_tree:
            bsdf = black_pieces_mat.node_tree.nodes.get('Principled BSDF')

            if is_wood and piece_scheme.get('black_texture'):
                self._apply_wood_texture(black_pieces_mat, piece_scheme['black_texture'])
            else:
                self._reset_piece_material(black_pieces_mat, piece_scheme['black'], is_white_mat=False)

            if bsdf:
                self.apply_material_properties(bsdf, piece_scheme, piece_roughness)

        # Randomize board material (wood, stone, plastic, painted, modern)
        board_material = np.random.choice(BOARD_MATERIALS)
        logger.debug(f"Selected board material: {board_material['name']} ({board_material['type']})")

        board_white_mat = bpy.data.materials.get('BoardWhite')
        board_black_mat = bpy.data.materials.get('BoardBlack')

        logger.debug(f"  BoardWhite material found: {board_white_mat is not None}")
        logger.debug(f"  BoardBlack material found: {board_black_mat is not None}")

        # Randomize roughness within material's range
        roughness_min, roughness_max = board_material['roughness']
        board_roughness = np.random.uniform(roughness_min, roughness_max)

        if board_white_mat:
            # The base color is controlled by Mix.001 node, not BSDF directly
            mix_node = board_white_mat.node_tree.nodes.get('Mix.001')
            bsdf = board_white_mat.node_tree.nodes.get('Principled BSDF')

            if mix_node:
                # Set both A and B inputs of the mix node to our color
                # (they blend with procedural texture for subtle variation)
                mix_node.inputs['A'].default_value = board_material['light']
                mix_node.inputs['B'].default_value = board_material['light']
                logger.debug(f"  Applied light squares to Mix.001: RGB{board_material['light'][:3]}")

            if bsdf:
                # Still set roughness, metallic, specular on BSDF
                bsdf.inputs['Roughness'].default_value = board_roughness
                bsdf.inputs['Metallic'].default_value = board_material['metallic']
                bsdf.inputs['Specular IOR Level'].default_value = board_material['specular']

                # Disconnect bump/normal to remove wood grain texture
                if bsdf.inputs['Normal'].is_linked:
                    self.disconnect_socket(board_white_mat.node_tree, bsdf, 'Normal')

                logger.debug(f"  Applied material properties: roughness={board_roughness:.2f}, metallic={board_material['metallic']}")

        if board_black_mat:
            # The base color is controlled by Mix.001 node, not BSDF directly
            mix_node = board_black_mat.node_tree.nodes.get('Mix.001')
            bsdf = board_black_mat.node_tree.nodes.get('Principled BSDF')

            if mix_node:
                # Set both A and B inputs of the mix node to our color
                mix_node.inputs['A'].default_value = board_material['dark']
                mix_node.inputs['B'].default_value = board_material['dark']
                logger.debug(f"  Applied dark squares to Mix.001: RGB{board_material['dark'][:3]}")

            if bsdf:
                # Still set roughness, metallic, specular on BSDF
                bsdf.inputs['Roughness'].default_value = board_roughness
                bsdf.inputs['Metallic'].default_value = board_material['metallic']
                bsdf.inputs['Specular IOR Level'].default_value = board_material['specular']

                # Disconnect bump/normal to remove wood grain texture
                if bsdf.inputs['Normal'].is_linked:
                    self.disconnect_socket(board_black_mat.node_tree, bsdf, 'Normal')

        # Randomize table material (wood, metal, stone, plastic, painted)
        table_material = np.random.choice(TABLE_MATERIALS)
        logger.debug(f"Selected table material: {table_material['name']} ({table_material['type']})")

        table_mat = bpy.data.materials.get('TableTop')

        if table_mat and table_mat.node_tree:
            # Base color is controlled by ColorRamp node, not BSDF directly
            colorramp = table_mat.node_tree.nodes.get('ColorRamp')
            bsdf = table_mat.node_tree.nodes.get('Principled BSDF')

            # Disconnect procedural texture from Roughness (causes visible artifacts)
            count = self.disconnect_socket(table_mat.node_tree, bsdf, 'Roughness')
            if count > 0:
                logger.debug(f"  Disconnected {count} procedural roughness texture link(s) from table")

            if colorramp:
                # Set all color stops to our base color (with slight variation for texture)
                base_color = table_material['color']
                for i, elem in enumerate(colorramp.color_ramp.elements):
                    # Add slight variation (darker to lighter) for wood grain effect
                    variation = i / (len(colorramp.color_ramp.elements) - 1)  # 0 to 1
                    # Darken the first stops, keep later ones at base or lighter
                    factor = 0.6 + (variation * 0.4)  # 0.6 to 1.0
                    varied_color = tuple(c * factor for c in base_color[:3]) + (1.0,)
                    elem.color = varied_color
                logger.debug(f"  Applied table color to ColorRamp: RGB{base_color[:3]}")

            if bsdf:
                # Still set roughness, metallic, specular on BSDF
                roughness_min, roughness_max = table_material['roughness']
                bsdf.inputs['Roughness'].default_value = np.random.uniform(roughness_min, roughness_max)
                bsdf.inputs['Metallic'].default_value = table_material['metallic']
                bsdf.inputs['Specular IOR Level'].default_value = table_material['specular']

        # Randomize floor/ground material using cached textures
        if self.floor_textures:
            texture_file, texture_name = self.floor_textures[
                np.random.randint(len(self.floor_textures))
            ]
            texture_path = Path(__file__).parent.parent.parent / "data" / "textures" / "floors" / texture_file
            logger.debug(f"Selected floor texture: {texture_name}")
        else:
            texture_file, texture_name = None, None
            texture_path = None
            logger.debug("Using solid color floor (no textures available)")

        floor_mat = bpy.data.materials.get('Floor2')
        if floor_mat and floor_mat.node_tree:
            bsdf = floor_mat.node_tree.nodes.get('Principled BSDF')
            material_output = floor_mat.node_tree.nodes.get('Material Output')

            # Collect links to remove first (to avoid modifying while iterating)
            links_to_remove = []
            for link in floor_mat.node_tree.links:
                # Disconnect roughness texture
                if link.to_node == bsdf and link.to_socket.name == 'Roughness':
                    links_to_remove.append(link)
                # Disconnect normal map
                elif link.to_node == bsdf and link.to_socket.name == 'Normal':
                    links_to_remove.append(link)
                # Disconnect displacement (this creates the stone tile lines!)
                elif link.to_node == material_output and link.to_socket.name == 'Displacement':
                    links_to_remove.append(link)

            # Now remove the collected links
            for link in links_to_remove:
                floor_mat.node_tree.links.remove(link)

            if links_to_remove:
                logger.debug(f"  Disconnected {len(links_to_remove)} old floor texture connections")

            # Find and update the Image Texture node for Base Color
            img_texture = floor_mat.node_tree.nodes.get('Image Texture.001')
            if img_texture and img_texture.type == 'TEX_IMAGE' and texture_path:
                if texture_path.exists():
                    img = bpy.data.images.load(str(texture_path), check_existing=True)
                    img_texture.image = img
                    logger.debug(f"  Applied texture: {texture_path.name}")
                else:
                    logger.warning(f"  Texture not found: {texture_path}")

            # Set material properties directly
            if bsdf:
                bsdf.inputs['Roughness'].default_value = np.random.uniform(0.3, 0.7)
                bsdf.inputs['Metallic'].default_value = np.random.uniform(0.0, 0.1)
                bsdf.inputs['Specular IOR Level'].default_value = np.random.uniform(0.3, 0.6)

    def hide_chairs(self):
        """Hide chair objects from the scene."""
        for obj in bpy.data.objects:
            # Check if object name contains "chair" (case insensitive)
            if 'chair' in obj.name.lower():
                obj.hide_render = True
                obj.hide_viewport = True
                logger.debug(f"Hiding chair: {obj.name}")

    def hide_all_pieces(self):
        """Hide all piece objects."""
        for obj in self.all_pieces.values():
            obj.location.z = -10  # Move far below board

    def validate_fen(self, fen: str) -> bool:
        """Validate FEN is a legal chess position."""
        if not HAVE_CHESS:
            # If chess library not available, do basic string validation
            return isinstance(fen, str) and len(fen) > 10 and '/' in fen

        try:
            board = chess.Board(fen)

            # Basic sanity checks
            piece_map = board.piece_map()
            if len(piece_map) == 0:
                logger.warning(f"FEN has no pieces: {fen}")
                return False

            # Check for at least one king per side
            white_kings = sum(1 for p in piece_map.values() if p.piece_type == chess.KING and p.color == chess.WHITE)
            black_kings = sum(1 for p in piece_map.values() if p.piece_type == chess.KING and p.color == chess.BLACK)

            if white_kings != 1 or black_kings != 1:
                logger.warning(f"FEN has invalid king count: {fen}")
                return False

            return True
        except ValueError as e:
            logger.warning(f"Invalid FEN: {fen} - {e}")
            return False

    def fen_to_board_state(self, fen: str) -> dict:
        """
        Convert FEN to board state (square -> piece type).

        Args:
            fen: FEN string

        Returns:
            Dict mapping square names to piece types (P, p, N, etc.)
        """
        board_part = fen.split()[0]
        ranks = board_part.split('/')

        state = {}

        for rank_idx, rank in enumerate(ranks):
            file_idx = 0

            for char in rank:
                if char.isdigit():
                    file_idx += int(char)
                else:
                    file_letter = chr(ord('A') + file_idx)
                    rank_number = str(8 - rank_idx)
                    square = f"{file_letter}{rank_number}"

                    state[square] = char
                    file_idx += 1

        return state

    def place_captured_pieces(self, piece_usage: dict):
        """
        Place captured pieces on the table beside the board with realistic variety.
        Only places pieces that were actually captured (starting position - current position).

        Randomizes placement style for training data diversity:
        - Neat grid (organized player)
        - Scattered random (casual game)
        - Piled up (hurried game)
        - Mixed orientations (pieces tipped over)
        - Hidden (in box/off camera)

        Args:
            piece_usage: Dict tracking how many pieces of each type are on the board
        """
        # Standard starting position piece counts
        STARTING_COUNTS = {
            'P': 8, 'N': 2, 'B': 2, 'R': 2, 'Q': 1, 'K': 1,
            'p': 8, 'n': 2, 'b': 2, 'r': 2, 'q': 1, 'k': 1,
        }

        # Randomly choose placement style (for training diversity)
        placement_style = np.random.choice(
            list(self.placement_weights.keys()),
            p=list(self.placement_weights.values())
        )

        # If hidden, just hide all captured pieces (as if in a box)
        if placement_style == 'hidden':
            # Pieces are already hidden by hide_all_pieces(), just don't place them
            return

        # Define areas on the table for captured pieces (left and right of board)
        board_width = 8 * self.square_size  # ~0.85m
        table_edge_offset = board_width / 2 + 0.15  # 15cm beyond board edge

        # Separate white and black captured pieces
        # Use table surface Z, not board center Z (which is on top of board)
        white_pieces_area = self.board_center + Vector((-table_edge_offset, 0, self.table_surface_z - self.board_center.z))
        black_pieces_area = self.board_center + Vector((table_edge_offset, 0, self.table_surface_z - self.board_center.z))

        # Track position for stacking/arranging captured pieces
        white_offset = 0
        black_offset = 0

        # Track placed piece positions and types for collision detection
        placed_positions = []  # List of (Vector, piece_type) tuples

        def is_position_valid(pos, placed_positions, piece_type, min_dist_override=None):
            """Check if position is valid with piece-specific collision."""
            piece_radius = PIECE_COLLISION_RADII.get(piece_type, 0.03)

            for placed_pos, placed_type in placed_positions:
                placed_radius = PIECE_COLLISION_RADII.get(placed_type, 0.03)
                required_distance = piece_radius + placed_radius + 0.005  # 5mm safety margin

                if min_dist_override:
                    required_distance = max(required_distance, min_dist_override)

                actual_distance = (pos - placed_pos).length
                if actual_distance < required_distance:
                    return False

            return True

        for piece_type, count_on_board in piece_usage.items():
            # How many pieces of this type exist in a starting position?
            starting_count = STARTING_COUNTS.get(piece_type, 0)

            # How many were captured?
            captured_count = max(0, starting_count - count_on_board)

            if captured_count == 0:
                continue  # No captures of this piece type

            available = self.available_pieces[piece_type]

            # Place captured pieces (starting from unused pieces)
            for i in range(captured_count):
                piece_idx = count_on_board + i
                if piece_idx >= len(available):
                    break  # Not enough piece objects available

                piece_obj = available[piece_idx]

                # Determine if white or black piece
                is_white = piece_type.isupper()

                if is_white:
                    # Place on left side of board
                    base_pos = white_pieces_area
                    offset = white_offset
                    white_offset += 1
                else:
                    # Place on right side of board
                    base_pos = black_pieces_area
                    offset = black_offset
                    black_offset += 1

                # Position based on placement style (with collision avoidance)
                max_retries = 20
                position = None

                for attempt in range(max_retries):
                    if placement_style == 'neat_grid':
                        # Organized grid layout (deterministic, no collision issues)
                        pieces_per_row = 3
                        row = offset // pieces_per_row
                        col = offset % pieces_per_row
                        spacing_y = 0.06  # 6cm spacing along board side
                        spacing_x = 0.07  # 7cm spacing for rows
                        random_x = np.random.uniform(-0.008, 0.008)
                        random_y = np.random.uniform(-0.008, 0.008)

                        # White pieces extend left (negative X), black pieces extend right (positive X)
                        x_direction = -1 if is_white else 1

                        candidate_pos = base_pos + Vector((
                            x_direction * row * spacing_x + random_x,  # Extend away from board
                            (col * spacing_y) - spacing_y + random_y,
                            0  # base_pos already at table surface
                        ))
                        # Upright with slight random rotation
                        piece_obj.rotation_euler = (0, 0, np.random.uniform(0, 2 * np.pi))

                    elif placement_style == 'scattered':
                        # Random scattered placement with collision check
                        # White pieces scatter left, black pieces scatter right
                        x_range = (-0.05, -0.20) if is_white else (0.05, 0.20)
                        candidate_pos = base_pos + Vector((
                            np.random.uniform(*x_range),  # Scattered area (away from board)
                            np.random.uniform(-0.15, 0.15),
                            0  # base_pos already at table surface
                        ))
                        # Some pieces upright, some tipped
                        if np.random.random() < 0.3:  # 30% tipped over
                            piece_obj.rotation_euler = (
                                np.random.uniform(-np.pi/2, np.pi/2),
                                np.random.uniform(-np.pi/4, np.pi/4),
                                np.random.uniform(0, 2 * np.pi)
                            )
                        else:
                            piece_obj.rotation_euler = (0, 0, np.random.uniform(0, 2 * np.pi))

                    elif placement_style == 'piled':
                        # Rough pile with collision check
                        pile_radius = 0.15  # 15cm radius pile (larger to fit more pieces)
                        # Limit angle range so pile extends away from board, not toward it
                        # White (left): 90° to 270° (left semicircle)
                        # Black (right): -90° to 90° (right semicircle)
                        if is_white:
                            angle = np.random.uniform(np.pi/2, 3*np.pi/2)  # Left side
                        else:
                            angle = np.random.uniform(-np.pi/2, np.pi/2)  # Right side
                        radius = np.random.uniform(0, pile_radius)

                        candidate_pos = base_pos + Vector((
                            radius * np.cos(angle),
                            radius * np.sin(angle),
                            np.random.uniform(0, 0.02)  # Small Z variation for pile effect (base_pos already at table surface)
                        ))
                        # Random orientations (reduced tilt to minimize footprint)
                        piece_obj.rotation_euler = (
                            np.random.uniform(-np.pi/6, np.pi/6),  # Reduced tilt (30° max)
                            np.random.uniform(-np.pi/6, np.pi/6),
                            np.random.uniform(0, 2 * np.pi)
                        )

                    # Check if position is valid (no collisions)
                    if is_position_valid(candidate_pos, placed_positions, piece_type):
                        position = candidate_pos
                        break

                # If we couldn't find a valid position after retries, hide the piece
                # (Don't force invalid placement that would cause intersections)
                if position is None:
                    # Hide piece below table (too crowded to place safely)
                    piece_obj.location = self.board_center + Vector((0, 0, -10))
                    logger.debug(f"  Hidden {piece_type} (no valid position)")
                else:
                    # Place piece at valid position
                    piece_obj.location = position
                    placed_positions.append((position, piece_type))
                    logger.debug(f"  Placed {piece_type} at Z={position.z:.4f}m")

    def get_piece_bounding_boxes(self, board_state: dict) -> list:
        """
        Calculate 2D bounding boxes for all visible pieces on the board.

        Args:
            board_state: Dict mapping square names to FEN piece characters

        Returns:
            List of dicts with piece info and normalized bounding boxes
        """
        scene = bpy.context.scene
        render = scene.render
        camera = self.camera

        bboxes = []

        # Track which piece objects we've already processed
        piece_usage = {key: 0 for key in self.available_pieces.keys()}

        for square, fen_piece in board_state.items():
            piece_type = FEN_TO_PIECE[fen_piece]

            # Get the piece object
            available = self.available_pieces[piece_type]
            if piece_usage[piece_type] >= len(available):
                continue

            obj = available[piece_usage[piece_type]]
            piece_usage[piece_type] += 1

            # Skip if piece is hidden or has no geometry
            if obj.hide_render or not obj.data:
                continue

            try:
                # Get 3D bounding box corners in world space
                bbox_corners_local = [Vector(corner) for corner in obj.bound_box]
                bbox_corners_world = [obj.matrix_world @ corner for corner in bbox_corners_local]

                # Project to camera space
                render_scale = render.resolution_percentage / 100.0
                render_width = int(render.resolution_x * render_scale)
                render_height = int(render.resolution_y * render_scale)

                # Project each corner to 2D screen space
                coords_2d = []
                for corner in bbox_corners_world:
                    # Convert world space to camera view
                    co_camera = bpy_extras.object_utils.world_to_camera_view(
                        scene, camera, corner
                    )
                    # Convert to pixel coordinates
                    x_pixel = co_camera.x * render_width
                    y_pixel = (1.0 - co_camera.y) * render_height  # Flip y-axis
                    coords_2d.append((x_pixel, y_pixel))

                # Calculate 2D bounding box from projected corners
                x_coords = [c[0] for c in coords_2d]
                y_coords = [c[1] for c in coords_2d]

                x_min = max(0, min(x_coords))
                x_max = min(render_width, max(x_coords))
                y_min = max(0, min(y_coords))
                y_max = min(render_height, max(y_coords))

                # Skip invalid bboxes (behind camera or collapsed)
                if x_max <= x_min or y_max <= y_min:
                    continue

                # Calculate YOLO format (normalized x_center, y_center, width, height)
                x_center = (x_min + x_max) / 2 / render_width
                y_center = (y_min + y_max) / 2 / render_height
                width = (x_max - x_min) / render_width
                height = (y_max - y_min) / render_height

                # Determine piece type for class mapping
                is_white = fen_piece.isupper()
                piece_name = fen_piece.upper()  # P, N, B, R, Q, K

                # Map to full piece type name
                piece_type_name_map = {
                    'P': 'pawn', 'N': 'knight', 'B': 'bishop',
                    'R': 'rook', 'Q': 'queen', 'K': 'king'
                }
                base_piece_name = piece_type_name_map[piece_name]
                color = 'w' if is_white else 'b'
                full_piece_type = f"{base_piece_name}_{color}"

                bboxes.append({
                    "square": square,
                    "piece_type": full_piece_type,
                    "bbox": {
                        "x_center": float(x_center),
                        "y_center": float(y_center),
                        "width": float(width),
                        "height": float(height),
                        "x_min": float(x_min / render_width),
                        "y_min": float(y_min / render_height),
                        "x_max": float(x_max / render_width),
                        "y_max": float(y_max / render_height),
                    }
                })

            except Exception as e:
                logger.warning(f"Failed to calculate bbox for {piece_type} at {square}: {e}")
                continue

        return bboxes

    def get_board_corners_2d(self) -> dict:
        """
        Calculate 2D positions of the 4 outer board corners.

        Projects the 3D corners of the 8x8 playing area to 2D camera space
        and sorts them into visual TL/TR/BR/BL order (matching ChessReD2k
        annotation convention).

        Returns:
            Dict with keys:
                - corners: dict with top_left, top_right, bottom_right,
                  bottom_left (each [x, y] normalized to [0,1])
                - bbox: dict with x_center, y_center, width, height (normalized)
        """
        scene = bpy.context.scene
        render = scene.render
        camera = self.camera

        render_scale = render.resolution_percentage / 100.0
        render_width = int(render.resolution_x * render_scale)
        render_height = int(render.resolution_y * render_scale)

        sq = self.square_size
        bc = self.board_center

        # 4 outer corners of the 8x8 grid in 3D world space
        # These are at the edges of the outermost squares (half a square
        # beyond the outermost square centers at +-3.5 * sq)
        corners_3d = [
            Vector((bc.x - 4 * sq, bc.y + 4 * sq, bc.z)),  # A8-side
            Vector((bc.x + 4 * sq, bc.y + 4 * sq, bc.z)),  # H8-side
            Vector((bc.x + 4 * sq, bc.y - 4 * sq, bc.z)),  # H1-side
            Vector((bc.x - 4 * sq, bc.y - 4 * sq, bc.z)),  # A1-side
        ]

        # Project to 2D pixel coordinates
        corners_2d = []
        for pt3d in corners_3d:
            co_cam = bpy_extras.object_utils.world_to_camera_view(
                scene, camera, pt3d
            )
            x_px = co_cam.x * render_width
            y_px = (1.0 - co_cam.y) * render_height  # Flip y-axis
            corners_2d.append((x_px, y_px))

        # Sort into visual TL/TR/BR/BL based on position in image
        # 1. Sort by y ascending (top of image first)
        sorted_by_y = sorted(corners_2d, key=lambda p: p[1])
        # 2. Top pair sorted by x
        top_pair = sorted(sorted_by_y[:2], key=lambda p: p[0])
        # 3. Bottom pair sorted by x
        bottom_pair = sorted(sorted_by_y[2:], key=lambda p: p[0])

        tl = top_pair[0]
        tr = top_pair[1]
        bl = bottom_pair[0]
        br = bottom_pair[1]

        # Normalize to [0, 1]
        def norm(pt):
            return [pt[0] / render_width, pt[1] / render_height]

        tl_n, tr_n, br_n, bl_n = norm(tl), norm(tr), norm(br), norm(bl)

        # Bounding box around all 4 corners (normalized)
        all_x = [tl_n[0], tr_n[0], br_n[0], bl_n[0]]
        all_y = [tl_n[1], tr_n[1], br_n[1], bl_n[1]]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

        return {
            "corners": {
                "top_left": tl_n,
                "top_right": tr_n,
                "bottom_right": br_n,
                "bottom_left": bl_n,
            },
            "bbox": {
                "x_center": (x_min + x_max) / 2,
                "y_center": (y_min + y_max) / 2,
                "width": x_max - x_min,
                "height": y_max - y_min,
            },
        }

    def setup_board(self, fen: str):
        """
        Set up board according to FEN position.

        Args:
            fen: FEN string
        """
        # Hide all pieces first
        self.hide_all_pieces()

        # Get board state from FEN
        board_state = self.fen_to_board_state(fen)

        # Track which pieces we've used
        piece_usage = {key: 0 for key in self.available_pieces.keys()}

        # Place pieces according to FEN
        for square, fen_piece in board_state.items():
            piece_type = FEN_TO_PIECE[fen_piece]

            # Get next available piece of this type
            available = self.available_pieces[piece_type]
            if piece_usage[piece_type] >= len(available):
                logger.warning(f"Not enough {piece_type} pieces available")
                continue

            piece_obj = available[piece_usage[piece_type]]
            piece_usage[piece_type] += 1

            # Position piece on square
            position = get_square_position(square, self.board_center, self.square_size)
            piece_obj.location = position

        # Place captured pieces on the table beside the board
        self.place_captured_pieces(piece_usage)

    def randomize_camera(self):
        """Randomize camera with profile-based variation."""
        # Select profile
        profiles = list(CAMERA_PROFILES.keys())
        weights = [CAMERA_PROFILES[p]['weight'] for p in profiles]
        profile_name = np.random.choice(profiles, p=weights)
        profile = CAMERA_PROFILES[profile_name]

        # Random spherical coordinates from profile
        theta = np.random.uniform(*[np.radians(x) for x in profile['theta']])
        phi = np.random.uniform(0, 2 * np.pi)  # 360° rotation

        # Camera distance from profile
        distance = np.random.uniform(*profile['distance'])

        # Convert to Cartesian
        x = distance * np.sin(theta) * np.cos(phi)
        y = distance * np.sin(theta) * np.sin(phi)
        z = distance * np.cos(theta)

        new_location = self.board_center + Vector((x, y, z))

        logger.debug(f"Camera randomization:")
        logger.debug(f"  Board center: {self.board_center}")
        logger.debug(f"  Distance: {distance:.3f}m")
        logger.debug(f"  Theta: {np.degrees(theta):.1f}°, Phi: {np.degrees(phi):.1f}°")
        logger.debug(f"  Offset (x,y,z): ({x:.3f}, {y:.3f}, {z:.3f})")
        logger.debug(f"  New location: {new_location}")

        self.camera.location = new_location

        # Point camera at board center
        direction = self.board_center - self.camera.location
        # Use Z as up vector (Blender world up)
        rot_quat = direction.to_track_quat('-Z', 'Z')

        # Add random roll rotation around the camera's view axis
        # This simulates viewing the board from different orientations
        from mathutils import Quaternion
        roll_angle = np.random.uniform(0, 2 * np.pi)
        # Create roll rotation around the view direction (normalized direction vector)
        view_axis = direction.normalized()
        roll_quat = Quaternion(view_axis, roll_angle)

        # Combine rotations: apply roll around view axis
        final_quat = roll_quat @ rot_quat
        self.camera.rotation_euler = final_quat.to_euler()

        # Random focal length from profile
        self.camera.data.lens = np.random.uniform(*profile['focal_length'])

        logger.debug(f"Camera profile: {profile_name}")

        # Force scene update to ensure camera changes are applied before rendering
        bpy.context.view_layer.update()

        logger.debug(f"  Camera location after update: {self.camera.location}")

    def set_random_hdri(self):
        """Set random HDRI for lighting and randomize world settings."""
        world = bpy.context.scene.world
        if not world or not world.node_tree:
            return

        # If HDRIs are available, randomly select one
        if len(self.hdris) > 0:
            hdri = np.random.choice(self.hdris)
            env_node = world.node_tree.nodes.get('Environment Texture')
            if env_node:
                env_node.image = hdri

        # Always randomize environment rotation for lighting variation (spherical)
        mapping_node = world.node_tree.nodes.get('Mapping')
        if mapping_node:
            # Random spherical rotation for varied lighting angles
            rotation_x = np.random.uniform(0, 2 * np.pi)  # Tilt
            rotation_y = np.random.uniform(0, 2 * np.pi)  # Roll
            rotation_z = np.random.uniform(0, 2 * np.pi)  # Pan
            mapping_node.inputs['Rotation'].default_value = (rotation_x, rotation_y, rotation_z)
            logger.debug(f"Environment rotation: X={np.degrees(rotation_x):.1f}°, Y={np.degrees(rotation_y):.1f}°, Z={np.degrees(rotation_z):.1f}°")

        # Randomize environment strength (brightness)
        background_node = world.node_tree.nodes.get('Background')
        if background_node:
            # Vary strength between 0.9 and 1.05 to avoid harsh over/underexposure
            strength = np.random.uniform(0.9, 1.05)
            background_node.inputs['Strength'].default_value = strength
            logger.debug(f"Environment strength: {strength:.2f}")

    def render_position(self, fen: str, image_index: int) -> dict:
        """
        Render a single chess position with randomized materials and camera.

        Steps:
        1. Randomizes all materials (pieces, board, table, floor)
        2. Sets up board pieces according to FEN
        3. Places captured pieces on table beside board
        4. Randomizes camera position and focal length
        5. Sets random HDRI lighting
        6. Renders to JPEG

        Args:
            fen: FEN notation string (e.g., "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            image_index: Zero-based index for file naming (chess_0000042.jpg)

        Returns:
            Metadata dict with keys:
                - id: Unique identifier (value_hybrid_0000042)
                - image: Absolute path to rendered JPEG
                - fen: Input FEN string

        Raises:
            RuntimeError: If rendering fails or produces invalid output

        Example:
            >>> metadata = gen.render_position(
            ...     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            ...     42
            ... )
            >>> print(metadata['image'])
            /path/to/output/chess_0000042.jpg
        """
        # Select random piece mesh set
        if self.piece_mesh_sets:
            piece_set = np.random.choice(list(self.piece_mesh_sets.keys()))
            self._select_piece_set(piece_set)
            logger.debug(f"Selected piece set: {piece_set}")
        else:
            piece_set = "value"

        # Randomize materials (pieces, board, table colors)
        self.randomize_materials()

        # Set up board
        self.setup_board(fen)

        # Get board state for bounding box calculation
        board_state = self.fen_to_board_state(fen)

        # Randomize camera
        self.randomize_camera()

        # Set random HDRI lighting
        self.set_random_hdri()

        # Calculate bounding boxes and board corners BEFORE rendering
        # (after camera is positioned so projections are correct)
        bounding_boxes = self.get_piece_bounding_boxes(board_state)
        board_corners = self.get_board_corners_2d()

        # Render
        output_path = self.output_dir / f"chess_{image_index:07d}.jpg"
        bpy.context.scene.render.filepath = str(output_path)

        try:
            bpy.ops.render.render(write_still=True)
        except Exception as e:
            logger.error(f"Render failed for image {image_index}: {e}")
            raise RuntimeError(f"Failed to render image {image_index}") from e

        # Validate output file
        if not output_path.exists():
            raise RuntimeError(f"Render produced no output file: {output_path}")
        if output_path.stat().st_size < 10000:  # Minimum reasonable JPEG size
            raise RuntimeError(f"Render produced tiny file (likely corrupted): {output_path}")

        return {
            "id": f"value_hybrid_{image_index:07d}",
            "image": str(output_path.absolute()),
            "fen": fen,
            "piece_set": piece_set,
            "bounding_boxes": bounding_boxes,
            "board_corners": board_corners,
        }

    def load_fen_positions(self) -> list[str]:
        """Load and validate FEN positions from JSON."""
        logger.info(f"Loading FEN positions from {self.fen_source}...")

        try:
            with open(self.fen_source, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.fen_source}: {e}") from e

        fen_positions = []
        for i, item in enumerate(data[:self.max_positions]):
            try:
                fen = item['conversations'][1]['value']
                if self.validate_fen(fen):
                    fen_positions.append(fen)
                else:
                    logger.warning(f"Skipping invalid FEN at index {i}")
            except (KeyError, IndexError):
                logger.warning(f"Skipping item {i} with missing FEN")
                continue

        if not fen_positions:
            raise ValueError(f"No valid FEN positions found in {self.fen_source}")

        logger.info(f"Loaded {len(fen_positions)} FEN positions")
        return fen_positions

    def generate(self) -> Path:
        """Generate full dataset."""
        # Load FEN positions
        fen_positions = self.load_fen_positions()

        annotation_file = self.output_dir / "annotations.json"

        # Check for existing progress
        start_idx = 0
        annotations = []

        if annotation_file.exists():
            logger.info(f"Found existing annotations at {annotation_file}")
            with open(annotation_file) as f:
                annotations = json.load(f)
            start_idx = len(annotations)
            logger.info(f"Resuming from image {start_idx}/{len(fen_positions)}")
        else:
            logger.info(f"Generating {len(fen_positions)} images from scratch...")

        if start_idx >= len(fen_positions):
            logger.info("All images already generated!")
            return annotation_file

        logger.info("=" * 60)

        # Render each position (starting from where we left off)
        for idx in range(start_idx, len(fen_positions)):
            fen = fen_positions[idx]
            metadata = self.render_position(fen, idx)

            # Convert to Llava format (with bounding boxes)
            llava_annotation = {
                "id": metadata["id"],
                "image": metadata["image"],
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\nWhat is the FEN position of this chess board?"
                    },
                    {
                        "from": "gpt",
                        "value": metadata["fen"]
                    }
                ],
                "bounding_boxes": metadata.get("bounding_boxes", []),
                "board_corners": metadata.get("board_corners"),
                "piece_set": metadata.get("piece_set"),
            }

            annotations.append(llava_annotation)

            # Save annotations after every image (negligible overhead vs render time)
            with open(annotation_file, 'w') as f:
                json.dump(annotations, f, indent=2)

            if len(annotations) % 10 == 0:
                logger.info(f"Rendered {len(annotations)}/{len(fen_positions)} images")

        logger.info("=" * 60)
        logger.info(f"Dataset generation complete!")
        logger.info(f"Images: {self.output_dir}")
        logger.info(f"Annotations: {annotation_file}")
        logger.info(f"Total: {len(annotations)} samples")

        return annotation_file


def main():
    """Main entry point."""

    # Project root for relative paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent

    # Parse arguments (Blender passes args after "--")
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Generate hybrid chess dataset using VALUE board"
    )
    parser.add_argument(
        "--fen-source",
        type=Path,
        default=PROJECT_ROOT / "data" / "value_llava" / "chess_fen_train.json",
        help="Path to Llava JSON with FEN positions"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "value_hybrid",
        help="Output directory"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--hdri-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "hdri",
        help="Directory with HDRI files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug output"
    )

    args = parser.parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(PROJECT_ROOT / 'dataset_generation.log'),
            logging.StreamHandler()
        ]
    )

    logger.info("=" * 60)
    logger.info("VALUE-based Hybrid Chess Dataset Generator")
    logger.info("=" * 60)
    logger.info(f"FEN source: {args.fen_source}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Images: {args.num_images}")
    logger.info(f"HDRI dir: {args.hdri_dir}")
    logger.info("=" * 60)

    # Generate dataset
    generator = VALUEHybridGenerator(
        fen_source=args.fen_source,
        output_dir=args.output_dir,
        max_positions=args.num_images,
        hdri_dir=args.hdri_dir if args.hdri_dir and args.hdri_dir.exists() else None,
    )

    annotation_file = generator.generate()

    logger.info("\nNext steps:")
    logger.info(f"1. View images: open {args.output_dir}")
    logger.info(f"2. Train Cosmos-RL with: {annotation_file}")


if __name__ == "__main__":
    if IN_BLENDER:
        main()
    else:
        print("ERROR: Must run inside Blender")
        print("\nUsage:")
        print("  /Applications/Blender.app/Contents/MacOS/Blender \\")
        print("    /Users/max/Code/VALUE-Dataset/rendering/board.blend -b \\")
        print("    -P /Users/max/Code/cosmos-chessbot/scripts/generate_value_hybrid_dataset.py \\")
        print("    -- --num-images 10")
        sys.exit(1)
