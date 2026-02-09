"""Demo mode for evaluators without hardware.

Replays a famous chess game (the Immortal Game, Anderssen vs Kieseritzky 1851)
through the full orchestrator pipeline, printing rich terminal output for each
phase so evaluators can see the Cosmos Reason2 integration in action.

Usage:
    cosmos-chessbot --demo
    cosmos-chessbot --demo --demo-scenario failure-recovery
"""

import random
import time
from typing import Optional

import chess

from .reasoning.fen_comparison import (
    FENComparison,
    SquareDifference,
    compare_fen_states,
    calculate_expected_fen,
)


# The Immortal Game — Anderssen vs Kieseritzky, London 1851
IMMORTAL_GAME_MOVES = [
    "e2e4", "e7e5",
    "f2f4", "e5f4",
    "f1c4", "d8h4",
    "e1f1", "b7b5",
    "c4b5", "g8f6",
    "g1f3", "h4h6",
    "d2d3", "f6h5",
    "f3h4", "h6g5",
    "h4f5", "c7c6",
    "g2g4", "h5f6",
    "h1g1", "c6b5",
    "h2h4", "g5g6",
    "h4h5", "g6g5",
    "d1f3", "f6g8",
    "c1f4", "g5f6",
    "b1c3", "f8c5",
    "c3d5", "f6b2",
    "f4d6", "c5g1",
    "e4e5", "b2a1",
    "f1e2", "b8a6",
    "f5g7", "e8d8",
    "f3f6", "a6f6",
    "d6e7",
]


def _mock_perception_output(fen: str) -> dict:
    """Generate mock Cosmos perception output."""
    return {
        "fen": fen,
        "confidence": round(random.uniform(0.88, 0.97), 2),
        "anomalies": [],
        "raw_response": f'{{"fen": "{fen}", "confidence": 0.94, "anomalies": []}}',
    }


def _mock_action_reasoning(move_uci: str, board: chess.Board) -> dict:
    """Generate mock Cosmos action reasoning."""
    from_sq = move_uci[:2]
    to_sq = move_uci[2:4]
    piece = board.piece_at(chess.parse_square(from_sq))
    piece_name = chess.piece_name(piece.piece_type) if piece else "piece"

    # Find adjacent pieces for realism
    adjacent = []
    to_idx = chess.parse_square(to_sq)
    for attack_sq in board.attacks(to_idx):
        p = board.piece_at(attack_sq)
        if p:
            adjacent.append(f"{chess.piece_name(p.piece_type)} on {chess.square_name(attack_sq)}")

    return {
        "obstacles": adjacent[:2] if adjacent else ["none detected"],
        "adjacent_pieces": adjacent[:3],
        "grasp_strategy": f"Top-pinch grasp on {piece_name}, approach from above",
        "trajectory_advice": f"Lift {piece_name} 5cm, arc over to {to_sq}, lower gently",
        "risks": [
            "Adjacent pieces could be bumped during placement"
        ] if adjacent else ["Clear path, low risk"],
        "confidence": round(random.uniform(0.82, 0.95), 2),
        "reasoning": (
            f"The {piece_name} on {from_sq} needs to move to {to_sq}. "
            f"I observe {len(adjacent)} adjacent pieces. "
            f"Best approach: lift vertically, arc trajectory to avoid collisions."
        ),
    }


def _mock_turn_detection(whose_turn: str) -> dict:
    """Generate mock Cosmos turn detection output."""
    return {
        "whose_turn": whose_turn,
        "opponent_moving": whose_turn == "opponent",
        "should_robot_act": whose_turn == "robot",
        "reasoning": (
            "I observe the opponent's hand has returned to their side of the board. "
            "A piece has been moved. It is now the robot's turn to act."
            if whose_turn == "robot"
            else "The opponent is reaching toward the board. I should wait."
        ),
        "confidence": round(random.uniform(0.85, 0.96), 2),
    }


def _mock_move_detection(move_uci: str, board: chess.Board) -> dict:
    """Generate mock Cosmos move detection output."""
    from_sq = move_uci[:2]
    to_sq = move_uci[2:4]
    piece = board.piece_at(chess.parse_square(from_sq))
    piece_name = chess.piece_name(piece.piece_type) if piece else "unknown"

    return {
        "move_occurred": True,
        "from_square": from_sq,
        "to_square": to_sq,
        "piece_type": piece_name,
        "confidence": round(random.uniform(0.80, 0.94), 2),
        "reasoning": (
            f"I observed the opponent's hand move a {piece_name} from {from_sq} to {to_sq}. "
            f"The piece was picked up, moved across the board, and placed down."
        ),
    }


def _mock_trajectory_plan(move_uci: str, board: chess.Board) -> dict:
    """Generate mock trajectory plan output (Action CoT)."""
    from_sq = move_uci[:2]
    to_sq = move_uci[2:4]
    piece = board.piece_at(chess.parse_square(from_sq))
    piece_name = chess.piece_name(piece.piece_type) if piece else "piece"

    # Map algebraic square to approximate normalized pixel coords (0-1000)
    # Simple linear mapping for demo: a=125, h=875 (X); 1=875, 8=125 (Y)
    def sq_to_px(sq: str) -> tuple[int, int]:
        file_idx = ord(sq[0]) - ord("a")
        rank_idx = int(sq[1]) - 1
        x = 125 + file_idx * 107
        y = 875 - rank_idx * 107
        return (x, y)

    from_px = sq_to_px(from_sq)
    to_px = sq_to_px(to_sq)
    mid_y = min(from_px[1], to_px[1]) - 80  # lift above both squares

    return {
        "waypoints": [
            {"point_2d": [from_px[0], from_px[1] - 40], "label": f"above {from_sq}"},
            {"point_2d": list(from_px), "label": f"grasp {from_sq}"},
            {"point_2d": [from_px[0], mid_y], "label": "lift"},
            {"point_2d": [to_px[0], mid_y], "label": f"above {to_sq}"},
            {"point_2d": list(to_px), "label": f"place {to_sq}"},
        ],
        "move_uci": move_uci,
        "reasoning": (
            f"The {piece_name} needs to move from {from_sq} to {to_sq}. "
            f"I plan a vertical lift from {from_sq}, horizontal traverse at "
            f"safe height avoiding adjacent pieces, then vertical descent "
            f"to {to_sq}."
        ),
        "confidence": round(random.uniform(0.85, 0.95), 2),
    }


def _mock_goal_verification(success: bool = True) -> dict:
    """Generate mock goal verification output."""
    if success:
        return {
            "success": True,
            "reason": "Piece correctly placed on target square, stable and upright",
            "physical_issues": [],
            "confidence": round(random.uniform(0.90, 0.97), 2),
            "reasoning": (
                "Looking at the board after the move, the piece appears to be "
                "correctly centered on the target square. It is standing upright "
                "and no adjacent pieces were disturbed."
            ),
        }
    else:
        return {
            "success": False,
            "reason": "Piece placed but leaning against adjacent piece",
            "physical_issues": ["piece_unstable", "contact_with_adjacent"],
            "confidence": round(random.uniform(0.80, 0.92), 2),
            "reasoning": (
                "The piece appears to be on the correct square but is leaning "
                "against an adjacent piece. The gripper may have released at a "
                "slight angle. The piece needs to be straightened."
            ),
        }


def _mock_correction_reasoning() -> dict:
    """Generate mock correction reasoning output."""
    return {
        "physical_cause": "Piece slipped during placement — gripper released 2mm too early",
        "correction_needed": "Re-grasp piece from current position and place on target square",
        "obstacles": ["Piece is slightly off-center on wrong square"],
        "confidence": 0.88,
        "reasoning": (
            "The piece did not land on the intended square. Looking at the board, "
            "I can see it is offset by approximately one square. The gripper likely "
            "released prematurely. I need to pick it up and re-place it."
        ),
    }


def _print_section(title: str, char: str = "-") -> None:
    """Print a formatted section header."""
    print(f"\n{char * 60}")
    print(f"  {title}")
    print(f"{char * 60}")


def _print_json_block(label: str, data: dict) -> None:
    """Print a labeled JSON-like block."""
    import json
    print(f"\n  [{label}]")
    formatted = json.dumps(data, indent=4)
    for line in formatted.split("\n"):
        print(f"    {line}")


def _print_board(board: chess.Board) -> None:
    """Print ASCII chess board."""
    print()
    for line in str(board).split("\n"):
        print(f"    {line}")
    print()


def run_demo(scenario: str = "normal", max_moves: Optional[int] = None) -> None:
    """Run the demo mode.

    Args:
        scenario: 'normal' or 'failure-recovery'
        max_moves: Maximum total moves to replay (None = full game)
    """
    print("=" * 60)
    print("  COSMOS CHESSBOT — DEMO MODE")
    print("  No hardware required. Replaying the Immortal Game (1851).")
    print("=" * 60)
    print()
    print("This demo shows each phase of the orchestrator pipeline:")
    print("  1.  Turn detection       (Cosmos Reason2 video reasoning)")
    print("  2.  Move detection       (Cosmos Reason2 video reasoning)")
    print("  3.  Perception           (YOLO-DINO image -> FEN)")
    print("  4.  Planning             (Stockfish best move)")
    print("  5a. Action reasoning     (Cosmos Reason2 physical reasoning)")
    print("  5b. Trajectory planning  (Cosmos Reason2 Action CoT -> pixel waypoints)")
    print("  6a. Goal verification    (Cosmos Reason2 visual check)")
    print("  6b. FEN verification     (FEN comparison)")
    if scenario == "failure-recovery":
        print("  7.  Recovery             (Cosmos Reason2 correction reasoning)")
    print()

    board = chess.Board()
    moves = IMMORTAL_GAME_MOVES
    limit = max_moves if max_moves is not None else len(moves)
    # In failure-recovery mode, inject a failure on robot's 3rd move
    failure_injected = False
    robot_move_count = 0

    for i, move_uci in enumerate(moves):
        if i >= limit:
            break

        move = chess.Move.from_uci(move_uci)
        is_white_move = board.turn == chess.WHITE
        # Robot plays white in the demo
        is_robot_turn = is_white_move

        if not is_robot_turn:
            # ---- Opponent's turn ----
            _print_section(
                f"MOVE {board.fullmove_number} — OPPONENT'S TURN ({move_uci})", "="
            )

            # Phase 1: Turn detection
            _print_section("Phase 1: Turn Detection (Cosmos Reason2 Video)")
            print("  Capturing 4 video frames from egocentric camera...")
            time.sleep(0.3)
            detection = _mock_turn_detection("robot")
            _print_json_block("Cosmos Reason2 output", detection)

            # Phase 2: Move detection
            _print_section("Phase 2: Move Detection (Cosmos Reason2 Video)")
            print("  Capturing 8 video frames showing opponent's move...")
            time.sleep(0.3)
            move_det = _mock_move_detection(move_uci, board)
            _print_json_block("Cosmos Reason2 output", move_det)

            board.push(move)
            print(f"\n  Opponent played: {move_uci}")
            _print_board(board)

        else:
            # ---- Robot's turn ----
            robot_move_count += 1
            _print_section(
                f"MOVE {board.fullmove_number} — ROBOT'S TURN ({move_uci})", "="
            )

            # Phase 3: Perception
            _print_section("Phase 3: Perception (Cosmos Reason2 Image -> FEN)")
            current_fen = board.fen()
            perception = _mock_perception_output(current_fen)
            print(f"  Current board FEN: {current_fen}")
            _print_json_block("Cosmos Reason2 output", perception)

            # Phase 4: Planning
            _print_section("Phase 4: Planning (Stockfish)")
            print(f"  Stockfish best move: {move_uci}")
            expected_fen = calculate_expected_fen(current_fen, move_uci)
            print(f"  Expected FEN after move: {expected_fen}")

            # Phase 5a: Action reasoning
            _print_section("Phase 5a: Action Reasoning (Cosmos Reason2 Physical)")
            action = _mock_action_reasoning(move_uci, board)
            _print_json_block("Cosmos Reason2 output", action)

            # Phase 5b: Trajectory planning (Action CoT)
            _print_section("Phase 5b: Trajectory Planning (Cosmos Reason2 Action CoT)")
            trajectory = _mock_trajectory_plan(move_uci, board)
            _print_json_block("Cosmos Reason2 trajectory output", trajectory)

            # Execute
            print("\n  Executing robot action along planned trajectory...")
            time.sleep(0.5)
            print("  Robot action executed.")

            # Failure injection for failure-recovery scenario
            inject_failure = (
                scenario == "failure-recovery"
                and robot_move_count == 3
                and not failure_injected
            )

            if inject_failure:
                failure_injected = True
                # Simulate the piece landing on a wrong square
                wrong_board = board.copy()
                wrong_board.push(move)
                # Pretend verification sees wrong state (piece on adjacent square)
                wrong_fen = board.fen()  # pre-move = "piece didn't move"

                # Phase 6a: Goal verification (failure)
                _print_section("Phase 6a: Goal Verification (Cosmos Reason2 Visual) — FAILURE")
                goal_check = _mock_goal_verification(success=False)
                _print_json_block("Cosmos Reason2 visual check", goal_check)

                # Phase 6b: FEN verification (failure)
                _print_section("Phase 6b: FEN Verification — FAILURE")
                comparison = compare_fen_states(expected_fen, wrong_fen)
                print(f"  Expected: {expected_fen}")
                print(f"  Actual:   {wrong_fen}")
                print(f"  Match: {comparison.match}")
                print(f"\n  {comparison.summary()}")

                # Phase 7: Recovery
                _print_section("Phase 7: Recovery (Cosmos Reason2 Correction)")
                correction = _mock_correction_reasoning()
                _print_json_block("Cosmos Reason2 correction analysis", correction)

                print("\n  Executing correction move...")
                time.sleep(0.5)
                print("  Correction executed.")

                # Re-verify (success)
                _print_section("Phase 6a (retry): Goal Verification — SUCCESS")
                goal_check2 = _mock_goal_verification(success=True)
                _print_json_block("Cosmos Reason2 visual check", goal_check2)

                _print_section("Phase 6b (retry): FEN Verification — SUCCESS")
                board.push(move)
                actual_fen = board.fen()
                comparison2 = compare_fen_states(expected_fen, actual_fen)
                print(f"  Expected: {expected_fen}")
                print(f"  Actual:   {actual_fen}")
                print(f"  Match: {comparison2.match}")
                print("\n  Recovery successful!")
            else:
                # Phase 6a: Goal verification (success)
                _print_section("Phase 6a: Goal Verification (Cosmos Reason2 Visual)")
                goal_check = _mock_goal_verification(success=True)
                _print_json_block("Cosmos Reason2 visual check", goal_check)

                # Phase 6b: FEN verification (success)
                board.push(move)
                actual_fen = board.fen()

                _print_section("Phase 6b: FEN Verification")
                comparison = compare_fen_states(expected_fen, actual_fen)
                print(f"  Expected: {expected_fen}")
                print(f"  Actual:   {actual_fen}")
                print(f"  Match: {comparison.match}")
                if comparison.match:
                    print("\n  Move verified successfully!")

            _print_board(board)

        # Small delay for readability
        time.sleep(0.2)

    # Game summary
    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)
    if board.is_game_over():
        print(f"  Result: {board.result()}")
        if board.is_checkmate():
            print("  Checkmate!")
    else:
        print(f"  Replayed {min(limit, len(moves))} moves of the Immortal Game.")
    print(f"\n  Final position:")
    _print_board(board)
