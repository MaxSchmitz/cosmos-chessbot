#!/usr/bin/env python3
"""
Hybrid Chess Dataset Generator

Combines:
- ChessR's visual diversity (boards, pieces, lighting, angles)
- VALUE's realistic Lichess FEN positions

Usage:
    cd /Users/max/Code/ChessR
    blender ChessR_datagen.blend -b -P /Users/max/Code/cosmos-chessbot/scripts/generate_hybrid_dataset.py -- --num-images 10000
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add ChessR and VALUE paths to import their utilities
CHESSR_PATH = Path("/Users/max/Code/ChessR/src")
VALUE_PATH = Path("/Users/max/Code/VALUE-Dataset/rendering/utils")

sys.path.insert(0, str(CHESSR_PATH))
sys.path.insert(0, str(VALUE_PATH))

# ChessR imports (when running in Blender)
try:
    import bpy
    import numpy as np
    from mathutils import Vector
    from BoardImagesGenerator import BoardConfigurationGenerator
    from Board import Board, PiecesSet
    import globals as chessrGlobals
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    print("Warning: Not running in Blender, imports will fail")

# VALUE imports
try:
    from ChessReader import BoardData
    import chess
    HAVE_CHESS = True
except ImportError:
    HAVE_CHESS = False
    print("Warning: chess library not available. Install with: pip install chess")


# FEN to ChessR piece type mapping
FEN_TO_PIECE_TYPE = {
    'P': 'pawn',   'p': 'pawn',
    'N': 'knight', 'n': 'knight',
    'B': 'bishop', 'b': 'bishop',
    'R': 'rook',   'r': 'rook',
    'Q': 'queen',  'q': 'queen',
    'K': 'king',   'k': 'king',
}

FEN_TO_COLOR = {
    'P': 'white', 'N': 'white', 'B': 'white', 'R': 'white', 'Q': 'white', 'K': 'white',
    'p': 'black', 'n': 'black', 'b': 'black', 'r': 'black', 'q': 'black', 'k': 'black',
}


def fen_to_configuration(fen_string: str) -> dict:
    """
    Convert FEN string to ChessR board configuration.

    Args:
        fen_string: FEN notation (e.g., "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    Returns:
        Dict mapping cell names (e.g., "A1") to piece info {"type": "pawn", "color": "white"}
    """
    # Parse only the board position part (before first space)
    board_part = fen_string.split()[0]
    ranks = board_part.split('/')  # Split into 8 ranks (rows)

    configuration = {}

    # FEN starts from rank 8 (top of board from white's perspective)
    for rank_idx, rank in enumerate(ranks):
        file_idx = 0  # Column index (A-H)

        for char in rank:
            if char.isdigit():
                # Empty squares
                file_idx += int(char)
            else:
                # Piece
                cell_letter = chessrGlobals.CELLS_LETTERS[file_idx]
                # FEN rank 8 = chess row 8, rank 1 = chess row 1
                cell_number = chessrGlobals.CELLS_NUMBERS[7 - rank_idx]
                cell_name = f"{cell_letter}{cell_number}"

                configuration[cell_name] = {
                    "type": FEN_TO_PIECE_TYPE[char],
                    "color": FEN_TO_COLOR[char]
                }

                file_idx += 1

    return configuration


class HybridDatasetGenerator:
    """Generates chess dataset combining ChessR visuals with VALUE FEN positions."""

    def __init__(
        self,
        pgn_file: Path,
        output_dir: Path,
        max_positions: int = 10000,
    ):
        """
        Initialize hybrid generator.

        Args:
            pgn_file: Path to Lichess PGN file
            output_dir: Where to save rendered images and annotations
            max_positions: How many positions to render
        """
        self.pgn_file = pgn_file
        self.output_dir = output_dir
        self.max_positions = max_positions

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChessR components (must be in Blender)
        if not IN_BLENDER:
            raise RuntimeError("This script must be run inside Blender")

        # Get all boards and piece sets from Blender scene
        self.all_boards = [
            Board(plateau)
            for plateau in bpy.data.collections[chessrGlobals.PLATEAUX_COLLECTION].objects
        ]

        self.all_piece_sets = [
            PiecesSet(collection)
            for collection in bpy.data.collections[chessrGlobals.GAME_SETS_COLLECTION].children
        ]

        print(f"Found {len(self.all_boards)} board styles")
        print(f"Found {len(self.all_piece_sets)} piece sets")

        # Initialize board configuration generator
        self.config_generator = BoardConfigurationGenerator()

    def load_fen_positions(self) -> list[str]:
        """
        Load FEN positions from Lichess PGN file.

        Returns:
            List of FEN strings
        """
        if not HAVE_CHESS:
            raise RuntimeError("chess library required. Install with: pip install chess")

        print(f"Loading FEN positions from {self.pgn_file}...")

        # Use VALUE's BoardData iterator
        board_data = BoardData(str(self.pgn_file), max_boards=self.max_positions)

        fen_positions = []
        for board in board_data:
            fen_positions.append(board.fen())

            if len(fen_positions) % 1000 == 0:
                print(f"  Loaded {len(fen_positions)} positions...")

        print(f"Loaded {len(fen_positions)} FEN positions")
        return fen_positions

    def render_position(
        self,
        fen: str,
        image_index: int,
    ) -> dict:
        """
        Render one chess position with ChessR-style variations.

        Args:
            fen: FEN string to render
            image_index: Unique image ID

        Returns:
            Metadata dict with FEN, configuration, etc.
        """
        # Random board and piece set selection (ChessR-style)
        board = np.random.choice(self.all_boards)
        piece_set = np.random.choice(self.all_piece_sets)

        # Duplicate board to avoid polluting original
        plateau_copy = board.mesh.copy()
        plateau_copy.data = plateau_copy.data.copy()
        bpy.context.collection.objects.link(plateau_copy)

        # Position board
        plateau_pos = Vector((-10, 10, 0))
        plateau_copy.location = plateau_pos

        # Random board rotation (ChessR-style)
        plateau_copy.rotation_euler[2] = np.random.uniform(2.0 * np.pi)

        # Convert FEN to configuration
        configuration = fen_to_configuration(fen)

        # Create Board object from copy
        board_instance = Board(plateau_copy)

        # Apply FEN configuration to board
        self.config_generator.applyConfigurationToBoard(
            board=board_instance,
            chessPiecesSet=piece_set,
            configuration=configuration
        )

        # Set random HDRI lighting (ChessR-style)
        self.config_generator.setRandomHDRI()

        # Position camera with random angle (ChessR-style)
        self.config_generator.positionCameraAroundBoardCenter(board_instance)

        # Render
        output_path = self.output_dir / f"chess_{image_index:07d}.jpg"
        bpy.context.scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)

        # Clean up scene (important to avoid memory buildup)
        # Delete all pieces that were created
        for cell_name in configuration.keys():
            if cell_name in board_instance.cellsDict:
                cell = board_instance.cellsDict[cell_name]
                if cell.piece is not None and cell.piece.mesh is not None:
                    bpy.data.objects.remove(cell.piece.mesh, do_unlink=True)

        # Delete board copy
        bpy.data.objects.remove(plateau_copy, do_unlink=True)

        # Return metadata
        return {
            "id": f"hybrid_{image_index:07d}",
            "image": str(output_path),
            "fen": fen,
            "configuration": configuration,
            "board_style": board.mesh.name,
            "piece_set": piece_set.name,
        }

    def generate(self) -> Path:
        """
        Generate full dataset.

        Returns:
            Path to Llava-format JSON annotation file
        """
        # Load FEN positions from Lichess
        fen_positions = self.load_fen_positions()

        # Limit to max_positions
        fen_positions = fen_positions[:self.max_positions]

        print(f"\nGenerating {len(fen_positions)} images...")
        print("=" * 60)

        # Render each position
        annotations = []
        for idx, fen in enumerate(fen_positions):
            metadata = self.render_position(fen, idx)

            # Convert to Llava format for Cosmos training
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

            # Progress
            if (idx + 1) % 10 == 0:
                print(f"Rendered {idx + 1}/{len(fen_positions)} images")

        # Save annotations in Llava format
        annotation_file = self.output_dir / "annotations.json"
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f, indent=2)

        print("=" * 60)
        print(f"\nDataset generation complete!")
        print(f"Images: {self.output_dir}")
        print(f"Annotations: {annotation_file}")
        print(f"Total samples: {len(annotations)}")

        return annotation_file


def main():
    """Main entry point when running in Blender."""

    # Parse command-line arguments
    # Blender passes args after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Generate hybrid chess dataset combining ChessR visuals with VALUE FEN positions"
    )
    parser.add_argument(
        "--pgn-file",
        type=Path,
        default=Path("/Users/max/Code/VALUE-Dataset/rendering/data/Dec18.pgn"),
        help="Path to Lichess PGN file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/max/Code/cosmos-chessbot/data/hybrid_dataset"),
        help="Output directory for images and annotations"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=10000,
        help="Number of images to generate"
    )

    args = parser.parse_args(argv)

    print("=" * 60)
    print("Hybrid Chess Dataset Generator")
    print("=" * 60)
    print(f"PGN file: {args.pgn_file}")
    print(f"Output: {args.output_dir}")
    print(f"Images: {args.num_images}")
    print("=" * 60)
    print()

    # Generate dataset
    generator = HybridDatasetGenerator(
        pgn_file=args.pgn_file,
        output_dir=args.output_dir,
        max_positions=args.num_images,
    )

    annotation_file = generator.generate()

    print("\nNext steps:")
    print("1. Verify sample quality:")
    print(f"   open {args.output_dir}")
    print("2. Train Cosmos-RL with this dataset:")
    print(f"   Update chess_sft_config.toml annotation_path to: {annotation_file}")


if __name__ == "__main__":
    if IN_BLENDER:
        main()
    else:
        print("ERROR: This script must be run inside Blender")
        print("\nUsage:")
        print("  cd /Users/max/Code/ChessR")
        print("  blender ChessR_datagen.blend -b -P /Users/max/Code/cosmos-chessbot/scripts/generate_hybrid_dataset.py -- --num-images 10000")
        sys.exit(1)
