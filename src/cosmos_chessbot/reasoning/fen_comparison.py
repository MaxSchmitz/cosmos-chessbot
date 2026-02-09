"""FEN state comparison and correction utilities.

Compares expected vs actual board states after move execution,
identifies misplaced/missing pieces, and generates correction moves.
"""

from dataclasses import dataclass, field
from typing import Optional

import chess


@dataclass
class SquareDifference:
    """A difference between expected and actual board state on one square."""

    square: str
    """Square in algebraic notation (e.g., 'e4')."""

    expected_piece: Optional[str]
    """Expected piece symbol (e.g., 'P' for white pawn), or None if empty."""

    actual_piece: Optional[str]
    """Actual piece symbol, or None if empty."""

    def __str__(self) -> str:
        exp = self.expected_piece or "empty"
        act = self.actual_piece or "empty"
        return f"{self.square}: expected {exp}, found {act}"


@dataclass
class FENComparison:
    """Result of comparing two FEN positions."""

    match: bool
    """Whether the positions match (piece placement only)."""

    expected_fen: str
    """The expected FEN string."""

    actual_fen: str
    """The actual FEN string."""

    differences: list[SquareDifference] = field(default_factory=list)
    """List of squares that differ."""

    def summary(self) -> str:
        """Human-readable summary of differences."""
        if self.match:
            return "Board states match."
        lines = [f"Found {len(self.differences)} difference(s):"]
        for diff in self.differences:
            lines.append(f"  {diff}")
        return "\n".join(lines)


def _piece_symbol(piece: Optional[chess.Piece]) -> Optional[str]:
    """Convert chess.Piece to symbol string, or None."""
    if piece is None:
        return None
    return piece.symbol()


def compare_fen_states(expected_fen: str, actual_fen: str) -> FENComparison:
    """Compare two FEN strings and identify differences.

    Only compares piece placement (first component of FEN).

    Args:
        expected_fen: Expected board state FEN
        actual_fen: Actual board state FEN

    Returns:
        FENComparison with match status and list of differences
    """
    expected_board = chess.Board(expected_fen)
    actual_board = chess.Board(actual_fen)

    differences: list[SquareDifference] = []

    for sq in chess.SQUARES:
        expected_piece = expected_board.piece_at(sq)
        actual_piece = actual_board.piece_at(sq)

        if expected_piece != actual_piece:
            differences.append(SquareDifference(
                square=chess.square_name(sq),
                expected_piece=_piece_symbol(expected_piece),
                actual_piece=_piece_symbol(actual_piece),
            ))

    return FENComparison(
        match=len(differences) == 0,
        expected_fen=expected_fen,
        actual_fen=actual_fen,
        differences=differences,
    )


def calculate_expected_fen(fen: str, move_uci: str) -> str:
    """Calculate expected FEN after applying a move.

    Args:
        fen: Current board FEN
        move_uci: Move in UCI format (e.g., 'e2e4')

    Returns:
        FEN string after the move
    """
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    board.push(move)
    return board.fen()


def find_misplaced_piece(comparison: FENComparison) -> Optional[SquareDifference]:
    """Find a piece that is on the wrong square.

    A misplaced piece is one where the actual square has a piece that
    doesn't belong there (actual_piece is not None and differs from expected).

    Args:
        comparison: FEN comparison result

    Returns:
        The first misplaced piece difference, or None
    """
    for diff in comparison.differences:
        if diff.actual_piece is not None and diff.actual_piece != diff.expected_piece:
            return diff
    return None


def find_missing_piece(comparison: FENComparison) -> Optional[SquareDifference]:
    """Find a square that should have a piece but is empty.

    Args:
        comparison: FEN comparison result

    Returns:
        The first missing piece difference, or None
    """
    for diff in comparison.differences:
        if diff.expected_piece is not None and diff.actual_piece is None:
            return diff
    return None


def generate_correction_move(comparison: FENComparison) -> Optional[str]:
    """Generate a UCI correction move for a simple misplacement.

    Handles the case where exactly one piece needs to be moved from
    its current (wrong) position to its expected position.

    Args:
        comparison: FEN comparison result

    Returns:
        UCI move string (e.g., 'e4e2') to correct the position,
        or None if the situation is too complex for automatic correction
    """
    if comparison.match:
        return None

    # Simple case: one piece is somewhere it shouldn't be, one square
    # is missing the piece it should have, and they're the same piece type.
    # This means the piece landed on the wrong square.
    wrong_squares = []  # squares with unexpected pieces
    empty_squares = []  # squares that should have pieces but don't

    for diff in comparison.differences:
        if diff.actual_piece is not None and diff.expected_piece is None:
            # Piece here that shouldn't be
            wrong_squares.append(diff)
        elif diff.actual_piece is None and diff.expected_piece is not None:
            # Empty but should have a piece
            empty_squares.append(diff)
        elif diff.actual_piece is not None and diff.expected_piece is not None:
            # Wrong piece type — too complex
            return None

    # Simple correction: one piece in wrong spot, one empty target
    if len(wrong_squares) == 1 and len(empty_squares) == 1:
        src = wrong_squares[0]
        dst = empty_squares[0]
        # Verify the piece types match
        if src.actual_piece == dst.expected_piece:
            return f"{src.square}{dst.square}"

    return None
