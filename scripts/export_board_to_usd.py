#!/usr/bin/env python3
"""
Export chess board and pieces from VALUE board.blend to USD for Isaac Sim.

Exports:
- board.usd -- chess board mesh
- pawn_w.usd, pawn_b.usd -- one of each piece type (12 total)

Usage:
    blender /Users/max/Code/VALUE-Dataset/rendering/board.blend -b \
        -P /Users/max/Code/cosmos-chessbot/scripts/export_board_to_usd.py \
        -- --output-dir data/usd
"""

import sys
import argparse
from pathlib import Path

try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    print("ERROR: Must run inside Blender")
    sys.exit(1)


# Map FEN piece characters to descriptive USD filenames
PIECE_TYPE_TO_FILENAME = {
    'P': 'pawn_w.usd',
    'N': 'knight_w.usd',
    'B': 'bishop_w.usd',
    'R': 'rook_w.usd',
    'Q': 'queen_w.usd',
    'K': 'king_w.usd',
    'p': 'pawn_b.usd',
    'n': 'knight_b.usd',
    'b': 'bishop_b.usd',
    'r': 'rook_b.usd',
    'q': 'queen_b.usd',
    'k': 'king_b.usd',
}


def export_object_to_usd(obj, output_path: Path) -> None:
    """Export a single Blender object to USD."""
    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')

    # Select only this object
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Export as USD
    bpy.ops.wm.usd_export(
        filepath=str(output_path),
        selected_objects_only=True,
        export_materials=True,
        export_uvmaps=True,
        evaluation_mode='RENDER',
        export_hair=False,
        export_animation=False,
    )

    print(f"  Exported: {obj.name} → {output_path.name}")


def main():
    """Export chess board and pieces to USD."""
    # Parse arguments (Blender passes args after "--")
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Export board and pieces to USD")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "data" / "usd",
        help="Output directory for USD files"
    )
    args = parser.parse_args(argv)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Exporting chess assets to USD for Isaac Sim")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print()

    # Export the board
    print("[1/13] Exporting board...")
    board_obj = bpy.data.objects.get('chessBoard')
    if board_obj:
        export_object_to_usd(board_obj, args.output_dir / "board.usd")
    else:
        print("  WARNING: 'chessBoard' object not found in scene")

    print()

    # Find all piece objects (2-char mesh names)
    all_pieces = {
        obj.name: obj
        for obj in bpy.data.objects
        if len(obj.name) == 2 and obj.type == 'MESH'
    }

    # Organize pieces by type
    pieces_by_type = {
        'P': [], 'N': [], 'B': [], 'R': [], 'Q': [], 'K': [],
        'p': [], 'n': [], 'b': [], 'r': [], 'q': [], 'k': [],
    }

    for name, obj in all_pieces.items():
        piece_type = name[0]
        if piece_type in pieces_by_type:
            pieces_by_type[piece_type].append(obj)

    print(f"Found {len(all_pieces)} piece objects:")
    for piece_type, pieces in pieces_by_type.items():
        print(f"  {piece_type}: {len(pieces)} pieces")
    print()

    # Export one of each piece type (12 total)
    print("Exporting pieces (one per type)...")
    exported_count = 0

    for idx, (piece_type, pieces) in enumerate(pieces_by_type.items(), start=2):
        if not pieces:
            print(f"[{idx}/13] WARNING: No {piece_type} pieces found")
            continue

        # Export the first instance of this piece type
        obj = pieces[0]
        output_filename = PIECE_TYPE_TO_FILENAME[piece_type]
        output_path = args.output_dir / output_filename

        print(f"[{idx}/13] Exporting {piece_type}...")
        export_object_to_usd(obj, output_path)
        exported_count += 1

    print()
    print("=" * 60)
    print(f"Export complete! {exported_count + 1} assets exported:")
    print(f"  Board: {args.output_dir / 'board.usd'}")
    print(f"  Pieces: {exported_count} types")
    print()
    print("Next steps:")
    print("  1. Copy data/usd/ to Isaac Sim project")
    print("  2. Load assets in Isaac env with stage.DefinePrim()")
    print("  3. Use fen_to_board_state() + get_square_position() for placement")
    print("=" * 60)


if __name__ == "__main__":
    if IN_BLENDER:
        main()
    else:
        print("ERROR: Must run inside Blender")
        print()
        print("Usage:")
        print("  blender /Users/max/Code/VALUE-Dataset/rendering/board.blend -b \\")
        print("    -P /Users/max/Code/cosmos-chessbot/scripts/export_board_to_usd.py \\")
        print("    -- --output-dir data/usd")
        sys.exit(1)
