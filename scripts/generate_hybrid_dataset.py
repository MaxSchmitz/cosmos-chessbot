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

# Add user site-packages so Blender can find installed packages (chess, tqdm, etc.)
import site
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.insert(0, user_site)

# Add ChessR and VALUE paths to import their utilities
CHESSR_PATH = Path("/Users/max/Code/ChessR/src")
VALUE_PATH = Path("/Users/max/Code/VALUE-Dataset/rendering/utils")

sys.path.insert(0, str(CHESSR_PATH))
sys.path.insert(0, str(VALUE_PATH))

# ChessR imports (when running in Blender)
try:
    import bpy
    import bpy_extras
    import bpy_extras.object_utils
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
    import chess
    HAVE_CHESS = True
except ImportError:
    HAVE_CHESS = False
    print("Warning: chess library not available. Install with: pip install chess")

try:
    from ChessReader import BoardData
    HAVE_CHESS_READER = True
except ImportError as e:
    HAVE_CHESS_READER = False
    print(f"Warning: ChessReader not available: {e}")


# FEN to ChessR piece type mapping (includes color suffix _w/_b)
FEN_TO_PIECE_TYPE = {
    # White pieces (uppercase)
    'P': 'pawn_w',
    'N': 'knight_w',
    'B': 'bishop_w',
    'R': 'rook_w',
    'Q': 'queen_w',
    'K': 'king_w',
    # Black pieces (lowercase)
    'p': 'pawn_b',
    'n': 'knight_b',
    'b': 'bishop_b',
    'r': 'rook_b',
    'q': 'queen_b',
    'k': 'king_b',
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

                # Map FEN piece to ChessR piece type (with color suffix)
                configuration[cell_name] = FEN_TO_PIECE_TYPE[char]

                file_idx += 1

    return configuration


class HybridDatasetGenerator:
    """Generates chess dataset combining ChessR visuals with VALUE FEN positions."""

    def __init__(
        self,
        fen_source: Path,
        output_dir: Path,
        max_positions: int = 10000,
    ):
        """
        Initialize hybrid generator.

        Args:
            fen_source: Path to PGN file OR Llava JSON with FENs
            output_dir: Where to save rendered images and annotations
            max_positions: How many positions to render
        """
        self.fen_source = fen_source
        self.output_dir = output_dir
        self.max_positions = max_positions

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChessR components (must be in Blender)
        if not IN_BLENDER:
            raise RuntimeError("This script must be run inside Blender")

        # Get all boards and piece sets from Blender scene
        self.all_boards = [
            Board(plateau, chessrGlobals.CELLS_NAMES)
            for plateau in chessrGlobals.PLATEAUX_COLLECTION.objects
        ]

        self.all_piece_sets = [
            PiecesSet(collection, chessrGlobals.PIECES_TYPES_COLLECTION)
            for collection in chessrGlobals.GAME_SETS_COLLECTION.children
        ]

        print(f"Found {len(self.all_boards)} board styles")
        print(f"Found {len(self.all_piece_sets)} piece sets")

        # Get scene camera
        self.camera = bpy.data.objects['Camera']
        print(f"Using camera: {self.camera.name}")

        # Initialize board configuration generator
        self.config_generator = BoardConfigurationGenerator()

    def load_fen_positions(self) -> list[str]:
        """
        Load FEN positions from PGN file or Llava JSON.

        Returns:
            List of FEN strings
        """
        print(f"Loading FEN positions from {self.fen_source}...")

        # Check if source is JSON (Llava format) or PGN
        if str(self.fen_source).endswith('.json'):
            # Load from Llava JSON format
            with open(self.fen_source, 'r') as f:
                data = json.load(f)

            fen_positions = []
            for item in data[:self.max_positions]:
                # Extract FEN from Llava conversation format
                # conversations[1] is the "gpt" response which contains the FEN
                fen = item['conversations'][1]['value']
                fen_positions.append(fen)

            print(f"Loaded {len(fen_positions)} FEN positions from JSON")
            return fen_positions

        else:
            # Load from PGN file
            if not HAVE_CHESS:
                raise RuntimeError("chess library required for PGN parsing. Install with: pip install chess")

            import chess.pgn

            fen_positions = []

            with open(self.fen_source, 'r') as pgn_file:
                game_count = 0

                while len(fen_positions) < self.max_positions:
                    game = chess.pgn.read_game(pgn_file)

                    if game is None:
                        # End of file
                        break

                    # Extract FEN from each position in the game
                    board = game.board()
                    for move in game.mainline_moves():
                        fen_positions.append(board.fen())
                        board.push(move)

                        if len(fen_positions) >= self.max_positions:
                            break

                    game_count += 1
                    if game_count % 100 == 0:
                        print(f"  Processed {game_count} games, extracted {len(fen_positions)} positions...")

            print(f"Loaded {len(fen_positions)} FEN positions from {game_count} PGN games")
            return fen_positions

    def get_piece_bounding_boxes(
        self,
        board,
        configuration: dict,
    ) -> list:
        """
        Calculate 2D bounding boxes for all pieces in camera view.

        Args:
            board: Board object with placed pieces
            configuration: Dict mapping cell names to piece info

        Returns:
            List of dicts with piece info and normalized bounding boxes
        """
        scene = bpy.context.scene
        render = scene.render
        camera = self.camera

        bboxes = []

        for cell_name, piece_info in configuration.items():
            # Get piece mesh object
            if cell_name not in board.cellsPieces:
                continue

            piece = board.cellsPieces[cell_name]
            if piece is None or piece.mesh is None:
                continue

            obj = piece.mesh

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
                # Convert world space to camera space
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

            # Calculate YOLO format (normalized x_center, y_center, width, height)
            x_center = (x_min + x_max) / 2 / render_width
            y_center = (y_min + y_max) / 2 / render_height
            width = (x_max - x_min) / render_width
            height = (y_max - y_min) / render_height

            # Get chess square name (e.g., "a1", "e4")
            square_name = cell_name.lower()

            # Get piece type and color from FEN character
            piece_type = piece_info["type"]  # e.g., "pawn_w", "knight_b"

            bboxes.append({
                "square": square_name,
                "piece_type": piece_type,
                "bbox": {
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "x_min": x_min / render_width,
                    "y_min": y_min / render_height,
                    "x_max": x_max / render_width,
                    "y_max": y_max / render_height,
                }
            })

        return bboxes

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
            Metadata dict with FEN, configuration, and bounding boxes
        """
        # Random board and piece set selection (ChessR-style)
        board = np.random.choice(self.all_boards)
        piece_set = np.random.choice(self.all_piece_sets)

        # Position board (modify original in place, we'll clean up after render)
        plateau_pos = Vector((-10, 10, 0))
        original_location = board.mesh.location.copy()
        board.mesh.location = plateau_pos

        # Random board rotation (ChessR-style)
        original_rotation = board.mesh.rotation_euler[2]
        board.mesh.rotation_euler[2] = np.random.uniform(2.0 * np.pi)

        # Convert FEN to configuration
        configuration = fen_to_configuration(fen)

        # Use the existing board instance (already has cells and corners)

        # Apply FEN configuration to board
        self.config_generator.applyConfigurationToBoard(
            configuration, board, piece_set
        )

        # Set random HDRI lighting (ChessR-style) - skip if no HDRIs
        if len(chessrGlobals.hdrisLoaded) > 0:
            self.config_generator.setRandomHDRI()

        # Position camera with random angle (ChessR-style)
        self.config_generator.positionCameraAroundBoardCenter(board, self.camera)

        # Calculate bounding boxes BEFORE rendering
        bounding_boxes = self.get_piece_bounding_boxes(board, configuration)

        # Render
        output_path = self.output_dir / f"chess_{image_index:07d}.jpg"
        bpy.context.scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)

        # Clean up scene (important to avoid memory buildup)
        # Delete all pieces that were created
        for cell_name in configuration.keys():
            if cell_name in board.cellsPieces:
                piece = board.cellsPieces[cell_name]
                if piece is not None and piece.mesh is not None:
                    bpy.data.objects.remove(piece.mesh, do_unlink=True)

        # Restore board to original state
        board.mesh.location = original_location
        board.mesh.rotation_euler[2] = original_rotation

        # Return metadata
        return {
            "id": f"hybrid_{image_index:07d}",
            "image": str(output_path),
            "fen": fen,
            "configuration": configuration,
            "board_style": board.mesh.name,
            "piece_set": piece_set.sourceCollection.name,
            "bounding_boxes": bounding_boxes,
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

            # Convert to Llava format for Cosmos training (with bounding boxes)
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
                "bounding_boxes": metadata["bounding_boxes"],  # Add bounding boxes for YOLO training
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
        "--fen-source",
        type=Path,
        default=Path("/Users/max/Code/cosmos-chessbot/data/value_llava/chess_fen_train.json"),
        help="Path to FEN source (PGN file or Llava JSON)"
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
    print(f"FEN source: {args.fen_source}")
    print(f"Output: {args.output_dir}")
    print(f"Images: {args.num_images}")
    print("=" * 60)
    print()

    # Generate dataset
    generator = HybridDatasetGenerator(
        fen_source=args.fen_source,
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
