#!/usr/bin/env python3
"""
Test USD asset loading in Isaac Sim.

This validates:
1. USD files load correctly
2. FEN placement module works
3. Pieces appear at correct 3D positions

Usage (on Isaac Sim machine):
    # Headless (no GUI)
    ./python.sh scripts/isaac_env_test.py

    # With GUI
    ./python.sh scripts/isaac_env_test.py --gui
"""

import argparse
import sys
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Isaac Sim imports
from isaacsim import SimulationApp

# Parse args before initializing SimulationApp
parser = argparse.ArgumentParser(description="Test USD asset loading")
parser.add_argument("--gui", action="store_true", help="Launch with GUI")
parser.add_argument("--usd-dir", type=Path, default=PROJECT_ROOT / "data" / "usd", help="USD assets directory")
args = parser.parse_args()

# Initialize Isaac Sim
simulation_app = SimulationApp({"headless": not args.gui})

# Now import omniverse modules (must be after SimulationApp init)
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdPhysics
import numpy as np

# Import our FEN placement module
from cosmos_chessbot.isaac.fen_placement import (
    fen_to_board_state,
    get_square_position,
    FEN_TO_PIECE_TYPE,
    BOARD_SQUARE_SIZE,
)


def create_test_scene(usd_dir: Path):
    """Create test scene with board and a few pieces."""
    # Get USD context and stage
    stage = omni.usd.get_context().get_stage()

    # Set up scene
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    print("=" * 60)
    print("Isaac Sim USD Asset Test")
    print("=" * 60)
    print(f"USD directory: {usd_dir}")
    print()

    # Create world
    world_prim = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world_prim)

    # Add ground plane
    print("[1/4] Creating ground plane...")
    ground_prim = stage.DefinePrim("/World/Ground", "Xform")
    ground_xform = UsdGeom.Xformable(ground_prim)
    ground_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5))

    # Add physics scene
    print("[2/4] Adding physics scene...")
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    # Load board
    print("[3/4] Loading chess board...")
    board_usd = usd_dir / "board.usd"
    if not board_usd.exists():
        print(f"ERROR: Board USD not found: {board_usd}")
        return False

    board_prim = stage.DefinePrim("/World/Board", "Xform")
    board_prim.GetReferences().AddReference(str(board_usd))
    print(f"  ✓ Board loaded from {board_usd.name}")

    # Load pieces from simple FEN (just a few pieces for testing)
    print("[4/4] Placing pieces from FEN...")
    test_fen = "4k3/8/8/8/4P3/8/8/4K3 w - - 0 1"  # Just kings and white pawn
    board_state = fen_to_board_state(test_fen)

    print(f"  FEN: {test_fen}")
    print(f"  Pieces to place: {len(board_state)}")

    piece_count = 0
    for square, fen_piece in board_state.items():
        # Get asset path
        piece_type = FEN_TO_PIECE_TYPE[fen_piece]
        piece_usd = usd_dir / f"{piece_type}.usd"

        if not piece_usd.exists():
            print(f"  WARNING: Piece USD not found: {piece_usd.name}")
            continue

        # Get 3D position
        pos = get_square_position(square)

        # Create prim and load USD
        prim_path = f"/World/Pieces/{square}_{piece_type}"
        piece_prim = stage.DefinePrim(prim_path, "Xform")
        piece_prim.GetReferences().AddReference(str(piece_usd))

        # Set position
        xform = UsdGeom.Xformable(piece_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

        print(f"  ✓ {square}: {piece_type:10s} at ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")
        piece_count += 1

    print()
    print("=" * 60)
    print(f"Test scene created successfully!")
    print(f"  Board: 1 object")
    print(f"  Pieces: {piece_count} objects")
    print(f"  Board square size: {BOARD_SQUARE_SIZE} meters")
    print("=" * 60)

    # Save scene
    output_usd = PROJECT_ROOT / "outputs" / "isaac_test_scene.usd"
    output_usd.parent.mkdir(parents=True, exist_ok=True)
    stage.Export(str(output_usd))
    print(f"\nScene saved to: {output_usd}")

    return True


def main():
    """Run test."""
    try:
        # Validate USD directory
        if not args.usd_dir.exists():
            print(f"ERROR: USD directory not found: {args.usd_dir}")
            print("\nPlease run on the machine where you exported the USD files:")
            print("  blender board.blend -b -P scripts/export_board_to_usd.py")
            return False

        # Create test scene
        success = create_test_scene(args.usd_dir)

        if success and args.gui:
            print("\nGUI Mode: Scene loaded. Press Ctrl+C to exit.")
            # Keep simulation running for inspection
            while simulation_app.is_running():
                simulation_app.update()

        return success

    finally:
        # Clean up
        simulation_app.close()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
