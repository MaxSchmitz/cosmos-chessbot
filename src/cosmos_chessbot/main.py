"""Main entry point for cosmos-chessbot."""

import argparse
from pathlib import Path

from .utils import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Cosmos Chessbot: Physical AI chess manipulation system"
    )
    parser.add_argument(
        "--overhead-camera",
        type=int,
        default=0,
        help="Egocentric camera device ID (default: 0)",
    )
    parser.add_argument(
        "--wrist-camera",
        type=int,
        default=1,
        help="Wrist camera device ID (default: 1)",
    )
    parser.add_argument(
        "--stockfish",
        type=str,
        default="stockfish",
        help="Path to Stockfish binary (default: stockfish)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Cosmos-Reason2-2B",
        help="Cosmos model to use (default: nvidia/Cosmos-Reason2-2B)",
    )
    parser.add_argument(
        "--cosmos-server",
        type=str,
        default=None,
        help="Remote Cosmos server URL (e.g., http://gpu-server:8000). If not set, runs locally.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for data storage (default: data/raw)",
    )
    parser.add_argument(
        "--moves",
        type=int,
        default=None,
        help="Number of moves to execute (default: run until game end)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="cosmos",
        choices=["pi05", "cosmos"],
        help="Policy to use: pi05 or cosmos (default: cosmos)",
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=Path,
        default=None,
        help="Path to policy checkpoint (optional, uses base model if not specified)",
    )
    parser.add_argument(
        "--enable-planning",
        action="store_true",
        default=True,
        help="Enable planning for Cosmos Policy (default: enabled)",
    )
    parser.add_argument(
        "--no-enable-planning",
        dest="enable_planning",
        action="store_false",
        help="Disable planning for Cosmos Policy",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="white",
        choices=["white", "black"],
        help="Color the robot plays as (default: white)",
    )
    parser.add_argument(
        "--game-mode",
        type=str,
        default="single-move",
        choices=["single-move", "full-game"],
        help="single-move: execute moves one at a time; full-game: full game loop with turn detection",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode (WARNING+ only)",
    )

    args = parser.parse_args()

    setup_logging(verbose=args.verbose, quiet=args.quiet)

    from .orchestrator import ChessOrchestrator, OrchestratorConfig

    config = OrchestratorConfig(
        egocentric_camera_id=args.overhead_camera,
        wrist_camera_id=args.wrist_camera,
        stockfish_path=args.stockfish,
        cosmos_model=args.model,
        cosmos_server_url=args.cosmos_server,
        data_dir=args.data_dir,
        policy_type=args.policy,
        policy_checkpoint=args.policy_checkpoint,
        enable_planning=args.enable_planning,
        color=args.color,
    )

    print("Initializing Cosmos Chessbot...")
    print(f"  Overhead camera: {args.overhead_camera}")
    print(f"  Wrist camera: {args.wrist_camera}")
    print(f"  Stockfish: {args.stockfish}")
    print(f"  Perception model: {args.model}")
    print(f"  Policy: {args.policy}")
    print(f"  Color: {args.color}")
    print(f"  Game mode: {args.game_mode}")
    if args.policy_checkpoint:
        print(f"  Policy checkpoint: {args.policy_checkpoint}")
    if args.policy == "cosmos":
        print(f"  Planning: {'enabled' if args.enable_planning else 'disabled'}")
    print()

    with ChessOrchestrator(config) as orchestrator:
        if args.game_mode == "full-game":
            orchestrator.run_game(max_moves=args.moves)
        else:
            move_count = 0
            max_moves = args.moves or float('inf')

            while move_count < max_moves:
                print(f"\n{'='*60}")
                print(f"Move {move_count + 1}")
                print('='*60)

                success = orchestrator.run_one_move()

                if not success:
                    print("Move failed. Stopping.")
                    break

                move_count += 1

            print(f"\nCompleted {move_count} moves")


if __name__ == "__main__":
    main()
