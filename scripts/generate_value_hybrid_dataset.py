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

# Piece material schemes with realistic material properties
PIECE_COLOR_SCHEMES = [
    # Plastic pieces - matte finish
    {
        "name": "plastic_white_black",
        "type": "plastic",
        "white": (0.95, 0.95, 0.95, 1.0),
        "black": (0.02, 0.02, 0.02, 1.0),
        "roughness": (0.4, 0.6),
        "metallic": 0.0,
        "specular": 0.3,
    },
    {
        "name": "plastic_ivory",
        "type": "plastic",
        "white": (0.9, 0.85, 0.7, 1.0),
        "black": (0.2, 0.15, 0.1, 1.0),
        "roughness": (0.4, 0.6),
        "metallic": 0.0,
        "specular": 0.3,
    },
    # Wood pieces - medium roughness
    {
        "name": "wood_light",
        "type": "wood",
        "white": (0.7, 0.5, 0.3, 1.0),
        "black": (0.3, 0.2, 0.1, 1.0),
        "roughness": (0.5, 0.7),
        "metallic": 0.0,
        "specular": 0.4,
    },
    {
        "name": "wood_dark",
        "type": "wood",
        "white": (0.85, 0.8, 0.7, 1.0),
        "black": (0.4, 0.3, 0.2, 1.0),
        "roughness": (0.5, 0.7),
        "metallic": 0.0,
        "specular": 0.4,
    },
    # Stone/Marble - polished finish
    {
        "name": "marble_white",
        "type": "stone",
        "white": (0.95, 0.95, 0.95, 1.0),
        "black": (0.15, 0.15, 0.15, 1.0),
        "roughness": (0.1, 0.3),
        "metallic": 0.0,
        "specular": 0.6,
    },
    {
        "name": "marble_onyx",
        "type": "stone",
        "white": (0.9, 0.88, 0.85, 1.0),
        "black": (0.05, 0.05, 0.05, 1.0),
        "roughness": (0.1, 0.3),
        "metallic": 0.0,
        "specular": 0.6,
    },
    # Metal pieces - shiny
    {
        "name": "metal_silver_bronze",
        "type": "metal",
        "white": (0.75, 0.75, 0.75, 1.0),
        "black": (0.5, 0.35, 0.2, 1.0),
        "roughness": (0.15, 0.3),
        "metallic": 0.8,
        "specular": 0.7,
    },
    {
        "name": "metal_gold_black",
        "type": "metal",
        "white": (0.8, 0.65, 0.3, 1.0),
        "black": (0.15, 0.15, 0.15, 1.0),
        "roughness": (0.15, 0.3),
        "metallic": 0.7,
        "specular": 0.7,
    },
    # Painted pieces - semi-glossy
    {
        "name": "painted_red_black",
        "type": "painted",
        "white": (0.7, 0.2, 0.2, 1.0),
        "black": (0.1, 0.1, 0.1, 1.0),
        "roughness": (0.3, 0.5),
        "metallic": 0.0,
        "specular": 0.4,
    },
    {
        "name": "painted_blue_black",
        "type": "painted",
        "white": (0.3, 0.4, 0.7, 1.0),
        "black": (0.1, 0.1, 0.1, 1.0),
        "roughness": (0.3, 0.5),
        "metallic": 0.0,
        "specular": 0.4,
    },
    {
        "name": "painted_green_black",
        "type": "painted",
        "white": (0.4, 0.6, 0.4, 1.0),
        "black": (0.1, 0.1, 0.1, 1.0),
        "roughness": (0.3, 0.5),
        "metallic": 0.0,
        "specular": 0.4,
    },
    # More wood varieties
    {
        "name": "wood_maple_walnut",
        "type": "wood",
        "white": (0.85, 0.75, 0.6, 1.0),  # Light maple
        "black": (0.35, 0.25, 0.2, 1.0),  # Dark walnut
        "roughness": (0.5, 0.7),
        "metallic": 0.0,
        "specular": 0.4,
    },
    {
        "name": "wood_cherry_ebony",
        "type": "wood",
        "white": (0.65, 0.35, 0.25, 1.0),  # Cherry red
        "black": (0.15, 0.12, 0.1, 1.0),   # Ebony
        "roughness": (0.5, 0.7),
        "metallic": 0.0,
        "specular": 0.4,
    },
    # Colored stone/marble
    {
        "name": "marble_green",
        "type": "stone",
        "white": (0.85, 0.9, 0.85, 1.0),   # White marble
        "black": (0.2, 0.4, 0.3, 1.0),     # Green marble
        "roughness": (0.1, 0.3),
        "metallic": 0.0,
        "specular": 0.6,
    },
    {
        "name": "marble_red",
        "type": "stone",
        "white": (0.9, 0.88, 0.85, 1.0),   # Cream marble
        "black": (0.45, 0.2, 0.2, 1.0),    # Red marble
        "roughness": (0.1, 0.3),
        "metallic": 0.0,
        "specular": 0.6,
    },
    # More metal combinations
    {
        "name": "metal_brass_steel",
        "type": "metal",
        "white": (0.7, 0.6, 0.3, 1.0),     # Brass
        "black": (0.6, 0.6, 0.65, 1.0),    # Steel
        "roughness": (0.15, 0.3),
        "metallic": 0.75,
        "specular": 0.7,
    },
    {
        "name": "metal_copper_iron",
        "type": "metal",
        "white": (0.72, 0.45, 0.2, 1.0),   # Copper
        "black": (0.35, 0.35, 0.4, 1.0),   # Dark iron
        "roughness": (0.2, 0.35),
        "metallic": 0.7,
        "specular": 0.6,
    },
    # Modern colored plastic
    {
        "name": "plastic_cream_brown",
        "type": "plastic",
        "white": (0.95, 0.9, 0.8, 1.0),    # Cream
        "black": (0.5, 0.35, 0.25, 1.0),   # Brown
        "roughness": (0.4, 0.6),
        "metallic": 0.0,
        "specular": 0.3,
    },
    {
        "name": "plastic_gray_charcoal",
        "type": "plastic",
        "white": (0.7, 0.7, 0.7, 1.0),     # Light gray
        "black": (0.25, 0.25, 0.25, 1.0),  # Charcoal
        "roughness": (0.4, 0.6),
        "metallic": 0.0,
        "specular": 0.3,
    },
    # Colorful modern sets
    {
        "name": "painted_orange_purple",
        "type": "painted",
        "white": (0.85, 0.45, 0.2, 1.0),   # Orange
        "black": (0.4, 0.2, 0.5, 1.0),     # Purple
        "roughness": (0.3, 0.5),
        "metallic": 0.0,
        "specular": 0.4,
    },
    {
        "name": "painted_yellow_navy",
        "type": "painted",
        "white": (0.9, 0.8, 0.3, 1.0),     # Yellow
        "black": (0.1, 0.15, 0.3, 1.0),    # Navy blue
        "roughness": (0.3, 0.5),
        "metallic": 0.0,
        "specular": 0.4,
    },
    {
        "name": "painted_pink_teal",
        "type": "painted",
        "white": (0.9, 0.5, 0.6, 1.0),     # Pink
        "black": (0.2, 0.5, 0.5, 1.0),     # Teal
        "roughness": (0.3, 0.5),
        "metallic": 0.0,
        "specular": 0.4,
    },
    # Tournament style
    {
        "name": "plastic_tournament",
        "type": "plastic",
        "white": (0.98, 0.98, 0.95, 1.0),  # Off-white
        "black": (0.08, 0.08, 0.08, 1.0),  # Near black
        "roughness": (0.5, 0.65),
        "metallic": 0.0,
        "specular": 0.25,
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
            config_path = Path(__file__).parent.parent / "config" / "dataset_generation.yaml"

        if HAVE_YAML and config_path.exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration if YAML not available or file missing
            self.config = {
                'render': {
                    'resolution': 600,
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
        textures_dir = Path(__file__).parent.parent / "data" / "textures" / "floors"

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

        if white_pieces_mat and white_pieces_mat.node_tree:
            bsdf = white_pieces_mat.node_tree.nodes.get('Principled BSDF')
            colorramp = white_pieces_mat.node_tree.nodes.get('ColorRamp')

            # Disconnect Mix node from Specular Tint (causes color artifacts)
            count = self.disconnect_socket(white_pieces_mat.node_tree, bsdf, 'Specular Tint')
            if count > 0:
                logger.debug(f"Disconnected {count} link(s) from Specular Tint")

            # Update ColorRamp (connected to Base Color)
            if colorramp:
                white_color = piece_scheme['white']
                for elem in colorramp.color_ramp.elements:
                    elem.color = white_color

            if bsdf:
                self.apply_material_properties(bsdf, piece_scheme, piece_roughness)

        if black_pieces_mat:
            bsdf = black_pieces_mat.node_tree.nodes.get('Principled BSDF')
            if bsdf:
                bsdf.inputs['Base Color'].default_value = piece_scheme['black']
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
            texture_path = Path(__file__).parent.parent / "data" / "textures" / "floors" / texture_file
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
        self.camera.rotation_euler = rot_quat.to_euler()

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
            # Vary strength between 0.75 and 1.25 (75% to 125% of default)
            # Narrower range to avoid under/overexposure
            strength = np.random.uniform(0.75, 1.25)
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
        # Randomize materials (pieces, board, table colors)
        self.randomize_materials()

        # Set up board
        self.setup_board(fen)

        # Randomize camera
        self.randomize_camera()

        # Set random HDRI lighting
        self.set_random_hdri()

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

            # Convert to Llava format
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
                ]
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
    PROJECT_ROOT = Path(__file__).parent.parent

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
