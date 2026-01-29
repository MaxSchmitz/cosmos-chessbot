"""Main entry point for cosmos-chessbot."""

import argparse
from pathlib import Path

from .orchestrator import ChessOrchestrator, OrchestratorConfig


def main():
    parser = argparse.ArgumentParser(
        description="Cosmos Chessbot: Physical AI chess manipulation system"
    )
    parser.add_argument(
        "--overhead-camera",
        type=int,
        default=0,
        help="Overhead camera device ID (default: 0)",
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

    args = parser.parse_args()

    config = OrchestratorConfig(
        overhead_camera_id=args.overhead_camera,
        wrist_camera_id=args.wrist_camera,
        stockfish_path=args.stockfish,
        cosmos_model=args.model,
        cosmos_server_url=args.cosmos_server,
        data_dir=args.data_dir,
    )

    print("Initializing Cosmos Chessbot...")
    print(f"  Overhead camera: {args.overhead_camera}")
    print(f"  Wrist camera: {args.wrist_camera}")
    print(f"  Stockfish: {args.stockfish}")
    print(f"  Model: {args.model}")
    print()

    with ChessOrchestrator(config) as orchestrator:
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
