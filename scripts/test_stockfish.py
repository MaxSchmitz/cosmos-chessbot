#!/usr/bin/env python3
"""Test the Stockfish engine wrapper."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from cosmos_chessbot.stockfish import StockfishEngine


def main():
    print("Testing Stockfish engine wrapper...")

    with StockfishEngine() as engine:
        # Test 1: Starting position
        print("\nTest 1: Best move from starting position")
        move = engine.get_best_move(depth=15)
        print(f"  Best move: {move}")

        # Test 2: Custom position (Scholar's mate setup)
        print("\nTest 2: Best move from custom FEN")
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        move = engine.get_best_move(fen=fen, depth=10)
        print(f"  FEN: {fen}")
        print(f"  Best move: {move}")
        print(f"  (Should be Qxf7# or similar winning move)")

        # Test 3: Position with move history
        print("\nTest 3: Best move after move sequence")
        engine.new_game()
        moves = ["e2e4", "e7e5", "g1f3"]
        move = engine.get_best_move(moves=moves, depth=15)
        print(f"  After moves: {' '.join(moves)}")
        print(f"  Best move: {move}")

    print("\nAll tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
