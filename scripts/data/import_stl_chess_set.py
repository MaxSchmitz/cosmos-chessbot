#!/usr/bin/env python3
"""
Import STL chess pieces into Blender and organize them for the dataset generator.

Usage:
    blender --background --python import_stl_chess_set.py -- \
        --stl-dir /path/to/stl/files \
        --output-blend chess_set_imported.blend \
        --set-name "MyChessSet"
"""

import sys
import argparse
from pathlib import Path

try:
    import bpy
    import mathutils
except ImportError:
    print("Error: Must run inside Blender")
    sys.exit(1)

# Piece type mapping (filename patterns to piece types)
PIECE_PATTERNS = {
    'P': ['pawn_white', 'pawn_w', 'white_pawn', 'wpawn'],
    'p': ['pawn_black', 'pawn_b', 'black_pawn', 'bpawn'],
    'N': ['knight_white', 'knight_w', 'white_knight', 'wknight'],
    'n': ['knight_black', 'knight_b', 'black_knight', 'bknight'],
    'B': ['bishop_white', 'bishop_w', 'white_bishop', 'wbishop'],
    'b': ['bishop_black', 'bishop_b', 'black_bishop', 'bbishop'],
    'R': ['rook_white', 'rook_w', 'white_rook', 'wrook'],
    'r': ['rook_black', 'rook_b', 'black_rook', 'brook'],
    'Q': ['queen_white', 'queen_w', 'white_queen', 'wqueen'],
    'q': ['queen_black', 'queen_b', 'black_queen', 'bqueen'],
    'K': ['king_white', 'king_w', 'white_king', 'wking'],
    'k': ['king_black', 'king_b', 'black_king', 'bking'],
}


def detect_piece_type(filename: str) -> str:
    """Detect piece type from filename."""
    filename_lower = filename.lower()

    for piece_type, patterns in PIECE_PATTERNS.items():
        for pattern in patterns:
            if pattern in filename_lower:
                return piece_type

    return None


def import_stl_chess_set(stl_dir: Path, set_name: str = "ImportedChessSet"):
    """
    Import all STL files from directory and organize them by piece type.

    Args:
        stl_dir: Directory containing STL files
        set_name: Name for the collection

    Returns:
        Collection containing imported pieces
    """
    if not stl_dir.exists():
        print(f"Error: Directory {stl_dir} does not exist")
        return None

    # Create collection for this chess set
    collection = bpy.data.collections.new(set_name)
    bpy.context.scene.collection.children.link(collection)

    # Find all STL files
    stl_files = list(stl_dir.glob("*.stl")) + list(stl_dir.glob("*.STL"))
    print(f"\nFound {len(stl_files)} STL files in {stl_dir}")

    imported_pieces = {
        'P': [], 'N': [], 'B': [], 'R': [], 'Q': [], 'K': [],
        'p': [], 'n': [], 'b': [], 'r': [], 'q': [], 'k': [],
    }

    # Import each STL
    for stl_file in stl_files:
        piece_type = detect_piece_type(stl_file.stem)

        if not piece_type:
            print(f"  Skipping {stl_file.name} - couldn't detect piece type")
            continue

        print(f"  Importing {stl_file.name} as {piece_type}")

        # Import STL
        bpy.ops.import_mesh.stl(filepath=str(stl_file))

        # Get the imported object (most recently added)
        obj = bpy.context.selected_objects[0]

        # Rename to piece type
        obj.name = f"{piece_type}_{len(imported_pieces[piece_type])}"

        # Move to collection
        for coll in obj.users_collection:
            coll.objects.unlink(obj)
        collection.objects.link(obj)

        # Center and scale (STLs might be in different units)
        # This positions pieces at origin, ready to be placed on board
        obj.location = (0, 0, 0)

        # Optional: Apply scale if pieces are too large/small
        # Uncomment if needed:
        # obj.scale = (0.01, 0.01, 0.01)  # Adjust scale factor as needed
        # bpy.ops.object.transform_apply(scale=True)

        imported_pieces[piece_type].append(obj)

    # Summary
    print(f"\nImported pieces:")
    for piece_type, pieces in imported_pieces.items():
        if pieces:
            print(f"  {piece_type}: {len(pieces)} pieces")

    # Check if we have all required pieces
    required_types = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    missing = [pt for pt in required_types if not imported_pieces[pt]]

    if missing:
        print(f"\nWarning: Missing piece types: {missing}")
    else:
        print(f"\nSuccess! Complete chess set imported into collection '{set_name}'")

    return collection


def main():
    # Parse arguments after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Import STL chess pieces into Blender"
    )
    parser.add_argument(
        "--stl-dir",
        type=Path,
        required=True,
        help="Directory containing STL files"
    )
    parser.add_argument(
        "--output-blend",
        type=Path,
        default=None,
        help="Output .blend file (optional)"
    )
    parser.add_argument(
        "--set-name",
        type=str,
        default="ImportedChessSet",
        help="Name for the chess set collection"
    )

    args = parser.parse_args(argv)

    print("=" * 60)
    print("STL Chess Set Importer")
    print("=" * 60)
    print(f"STL directory: {args.stl_dir}")
    print(f"Set name: {args.set_name}")

    # Import the chess set
    collection = import_stl_chess_set(args.stl_dir, args.set_name)

    # Save if output specified
    if args.output_blend and collection:
        args.output_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.output_blend))
        print(f"\nSaved to {args.output_blend}")

    print("\nDone!")


if __name__ == "__main__":
    main()
