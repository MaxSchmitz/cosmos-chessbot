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

    args = parser.parse_args()

    config = OrchestratorConfig(
        egocentric_camera_id=args.egocentric_camera,
        wrist_camera_id=args.wrist_camera,
        stockfish_path=args.stockfish,
        cosmos_model=args.model,
        cosmos_server_url=args.cosmos_server,
        data_dir=args.data_dir,
        policy_type=args.policy,
        policy_checkpoint=args.policy_checkpoint,
        enable_planning=args.enable_planning,
    )

    print("Initializing Cosmos Chessbot...")
    print(f"  Egocentric camera: {args.egocentric_camera}")
    print(f"  Wrist camera: {args.wrist_camera}")
    print(f"  Stockfish: {args.stockfish}")
    print(f"  Perception model: {args.model}")
    print(f"  Policy: {args.policy}")
    if args.policy_checkpoint:
        print(f"  Policy checkpoint: {args.policy_checkpoint}")
    if args.policy == "cosmos":
        print(f"  Planning: {'enabled' if args.enable_planning else 'disabled'}")
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
