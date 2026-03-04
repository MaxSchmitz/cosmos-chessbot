#!/usr/bin/env python3
"""
Demo script showing Cosmos-Reason2 verification and recovery loop.

This demonstrates the key innovation for the Cosmos challenge:
1. Pre-action reasoning (obstacles, grasp strategy)
2. Action execution
3. Post-action verification (FEN comparison)
4. Recovery with correction reasoning

Usage:
    python scripts/demo_verification_loop.py
"""

import sys
from pathlib import Path

# Direct import to avoid loading heavy dependencies
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "cosmos_chessbot" / "reasoning"))

from fen_comparison import (
    compare_fen_states,
    calculate_expected_fen,
    generate_correction_move,
)


def demo_fen_comparison():
    """Demo FEN comparison and correction generation."""
    print("=" * 60)
    print("DEMO: FEN Comparison and Correction")
    print("=" * 60)

    # Scenario: Robot tried to move pawn from e2 to e4, but it ended up on d4
    current_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    planned_move = "e2e4"

    print(f"\n1. Current board state:")
    print(f"   FEN: {current_fen}")

    print(f"\n2. Planned move: {planned_move}")
    expected_fen = calculate_expected_fen(current_fen, planned_move)
    print(f"   Expected FEN: {expected_fen}")

    # Simulate actual FEN (piece on d4 instead of e4)
    actual_fen = "rnbqkbnr/pppppppp/8/8/3P4/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    print(f"\n3. Actual board state after execution:")
    print(f"   Actual FEN: {actual_fen}")

    # Compare
    print(f"\n4. Comparing expected vs actual:")
    comparison = compare_fen_states(expected_fen, actual_fen)
    print(f"   Match: {comparison.match}")
    print(f"   Differences: {len(comparison.differences)}")

    for diff in comparison.differences:
        print(f"     - {diff}")

    # Generate correction
    print(f"\n5. Generating correction move:")
    correction = generate_correction_move(comparison)
    if correction:
        print(f"   Correction: {correction}")
        print(f"   Explanation: Move piece from {correction[:2]} to {correction[2:4]}")

        # Verify correction would work
        corrected_fen = calculate_expected_fen(actual_fen, correction)
        final_comparison = compare_fen_states(expected_fen, corrected_fen)
        print(f"\n6. After correction:")
        print(f"   Match: {final_comparison.match}")
    else:
        print(f"   ⚠️  Cannot automatically generate correction (complex case)")


def demo_complex_scenario():
    """Demo a more complex scenario with multiple pieces."""
    print("\n\n" + "=" * 60)
    print("DEMO: Complex Scenario")
    print("=" * 60)

    # Scenario: Multiple pieces affected (e.g., knocked over adjacent piece)
    current_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    planned_move = "d7d5"

    print(f"\n1. Current position (after white's e4):")
    print(f"   FEN: {current_fen}")

    print(f"\n2. Black plays: {planned_move}")
    expected_fen = calculate_expected_fen(current_fen, planned_move)
    print(f"   Expected FEN: {expected_fen}")

    # Simulate: Robot knocked the e4 pawn while moving d7-d5
    actual_fen = "rnbqkbnr/ppp1pppp/8/3p4/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
    print(f"\n3. Actual state (e4 pawn knocked off):")
    print(f"   Actual FEN: {actual_fen}")

    # Compare
    print(f"\n4. Analysis:")
    comparison = compare_fen_states(expected_fen, actual_fen)
    print(f"   Match: {comparison.match}")
    print(f"   Issues found:")

    for diff in comparison.differences:
        print(f"     - {diff}")

    # In this case, correction is complex (need to put back e4 pawn)
    correction = generate_correction_move(comparison)
    print(f"\n5. Automatic correction: {correction if correction else 'Not possible (complex)'}")

    if not correction:
        print(f"   ℹ️  This scenario requires Cosmos reasoning:")
        print(f"      - Identify that e4 pawn was knocked off")
        print(f"      - Plan to first restore e4 pawn position")
        print(f"      - Then verify d5 pawn is correctly placed")


def demo_full_game_flow():
    """Demo the full orchestrator flow (pseudocode)."""
    print("\n\n" + "=" * 60)
    print("DEMO: Full Game Flow (Pseudocode)")
    print("=" * 60)

    print("""
    The complete Cosmos chess brain flow:

    1. WAIT FOR OPPONENT (Cosmos Reason2)
       ├─ Analyze video stream
       ├─ Question: "Has opponent finished their move?"
       ├─ Question: "Is their hand away from board?"
       └─ Output: should_robot_act = True/False

    2. EXTRACT STATE (YOLO26-DINO)
       ├─ Detect pieces with YOLO26
       ├─ Classify with DINO + MLP
       └─ Output: Current FEN

    3. PLAN MOVE (Stockfish)
       ├─ Input: Current FEN
       └─ Output: Best move (e.g., "e2e4")

    4. PRE-ACTION REASONING (Cosmos Reason2)
       ├─ Question: "What obstacles are in the path?"
       ├─ Question: "What grasp strategy should I use?"
       ├─ Question: "Will I knock over adjacent pieces?"
       └─ Output: ActionReasoning with strategy

    5. EXECUTE (Cosmos Policy or π₀.₅)
       ├─ Input: Images, robot state, task description
       └─ Output: Joint actions

    6. VERIFY (YOLO26-DINO + FEN Compare)
       ├─ Extract actual FEN
       ├─ Compare to expected FEN
       └─ Output: Differences if any

    7. RECOVERY (If needed - Cosmos Reason2)
       ├─ Question: "What went wrong physically?"
       ├─ Generate correction move from FEN diff
       ├─ Execute correction
       └─ Verify again (loop until correct)
    """)


if __name__ == "__main__":
    print("\n" + "🤖 " * 20)
    print("Cosmos Chess Brain - Verification & Recovery Demo")
    print("🤖 " * 20)

    demo_fen_comparison()
    demo_complex_scenario()
    demo_full_game_flow()

    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)
    print("\nThis demonstrates the key innovation for Cosmos challenge:")
    print("  - Precise state extraction (FEN from vision)")
    print("  - Physical reasoning (Cosmos understanding failures)")
    print("  - Automated recovery (correction generation)")
    print("  - Integration of symbolic (FEN) and physical (Cosmos) AI")
