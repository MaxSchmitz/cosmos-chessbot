"""
FEN-to-3D chess piece placement logic.

This module is renderer-agnostic -- it maps FEN positions to 3D coordinates
without any dependencies on Blender or Isaac Sim. Both rendering systems
can import and use these functions.

Key constants from VALUE board.blend:
- Board center: (0, 0, 0) in Blender world space
- Square size: 0.106768 meters (standard tournament board)
"""

import numpy as np
from typing import Dict, Tuple


# FEN piece character to piece type name
FEN_TO_PIECE_TYPE = {
    # White pieces (uppercase)
    'P': 'pawn_w',
    'N': 'knight_w',
    'B': 'bishop_w',
    'R': 'rook_w',
    'Q': 'queen_w',
    'K': 'king_w',
    # Black pieces (lowercase)
    'p': 'pawn_b',
    'n': 'knight_b',
    'b': 'bishop_b',
    'r': 'rook_b',
    'q': 'queen_b',
    'k': 'king_b',
}

# Physical collision radii for each piece type (in meters)
# Used for PhysX collision shapes and reward proximity thresholds
PIECE_COLLISION_RADII = {
    'P': 0.020, 'p': 0.020,  # Pawns
    'N': 0.025, 'n': 0.025,  # Knights
    'B': 0.027, 'b': 0.027,  # Bishops
    'R': 0.028, 'r': 0.028,  # Rooks
    'Q': 0.035, 'q': 0.035,  # Queens
    'K': 0.035, 'k': 0.035,  # Kings
}

# Standard chess starting position piece counts
STARTING_PIECE_COUNTS = {
    'P': 8, 'N': 2, 'B': 2, 'R': 2, 'Q': 1, 'K': 1,
    'p': 8, 'n': 2, 'b': 2, 'r': 2, 'q': 1, 'k': 1,
}

# Board geometry (from VALUE board.blend)
BOARD_SQUARE_SIZE = 0.106768  # meters (tournament board standard)


def fen_to_board_state(fen: str) -> Dict[str, str]:
    """
    Parse FEN string to board state.

    Args:
        fen: FEN notation (e.g., "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    Returns:
        Dict mapping square names (A1-H8) to FEN piece characters (P/p/N/n/etc.)

    Example:
        >>> state = fen_to_board_state("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        >>> state['E1']
        'K'
        >>> state['E8']
        'k'
    """
    # Extract board position (first field before space)
    board_part = fen.split()[0]
    ranks = board_part.split('/')

    state = {}

    # FEN ranks are listed from 8 to 1 (top to bottom from white's view)
    for rank_idx, rank in enumerate(ranks):
        file_idx = 0

        for char in rank:
            if char.isdigit():
                # Empty squares
                file_idx += int(char)
            else:
                # Piece on this square
                file_letter = chr(ord('A') + file_idx)
                rank_number = str(8 - rank_idx)
                square = f"{file_letter}{rank_number}"

                state[square] = char
                file_idx += 1

    return state


def get_square_position(
    square_name: str,
    board_center: np.ndarray = np.array([0.0, 0.0, 0.0]),
    square_size: float = BOARD_SQUARE_SIZE
) -> np.ndarray:
    """
    Get 3D position for a chess square.

    Coordinate system:
    - X axis: A to H (left to right from white's view)
    - Y axis: 1 to 8 (bottom to top from white's view)
    - Z axis: vertical (up)
    - Origin: board center

    Args:
        square_name: Chess square (e.g., "A1", "E4", "H8")
        board_center: 3D position of board center [x, y, z]
        square_size: Size of each square in meters

    Returns:
        3D position [x, y, z] for the square center

    Example:
        >>> pos = get_square_position("E4")
        >>> float(round(pos[0], 3)), float(round(pos[1], 3))
        (0.053, -0.053)
        >>> pos = get_square_position("A1")  # Bottom-left corner (white's view)
        >>> bool(pos[0] < 0 and pos[1] < 0)  # Negative X and Y
        True
    """
    file_letter = square_name[0].upper()
    rank_number = square_name[1]

    # Convert to 0-7 indices
    file_idx = ord(file_letter) - ord('A')  # 0=A, 7=H
    rank_idx = int(rank_number) - 1  # 0=1, 7=8

    # Calculate position offset from board center
    # Center of board is between D4-E4 and D5-E5 (offset 3.5 squares in each direction)
    x_offset = (file_idx - 3.5) * square_size
    y_offset = (rank_idx - 3.5) * square_size

    return board_center + np.array([x_offset, y_offset, 0.0])


def get_captured_pieces(board_state: Dict[str, str]) -> Dict[str, int]:
    """
    Calculate how many pieces of each type have been captured.

    Args:
        board_state: Dict from fen_to_board_state() mapping squares to pieces

    Returns:
        Dict mapping piece types to capture counts

    Example:
        >>> # Starting position
        >>> state = fen_to_board_state("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        >>> captured = get_captured_pieces(state)
        >>> all(count == 0 for count in captured.values())
        True
        >>> # After white pawn on e4 captured black pawn (only 7 black pawns remain)
        >>> state = fen_to_board_state("rnbqkbnr/ppp1pppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")
        >>> captured = get_captured_pieces(state)
        >>> captured['p']  # One black pawn captured
        1
    """
    # Count pieces currently on board
    pieces_on_board = {}
    for piece in board_state.values():
        pieces_on_board[piece] = pieces_on_board.get(piece, 0) + 1

    # Calculate captured counts
    captured = {}
    for piece_type, starting_count in STARTING_PIECE_COUNTS.items():
        current_count = pieces_on_board.get(piece_type, 0)
        captured[piece_type] = max(0, starting_count - current_count)

    return captured


def validate_fen(fen: str) -> bool:
    """
    Validate FEN is a legal chess position.

    Basic validation without python-chess dependency:
    - Has at least 6 fields (position, turn, castling, en passant, halfmove, fullmove)
    - Position has 8 ranks separated by '/'
    - Each rank has valid piece characters and empty square counts

    For full validation, use python-chess: chess.Board(fen).is_valid()

    Args:
        fen: FEN string

    Returns:
        True if FEN passes basic validation

    Example:
        >>> validate_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        True
        >>> validate_fen("invalid")
        False
    """
    if not isinstance(fen, str) or len(fen) < 10:
        return False

    fields = fen.split()
    if len(fields) < 6:
        return False

    # Validate position field
    position = fields[0]
    ranks = position.split('/')

    if len(ranks) != 8:
        return False

    valid_pieces = set('pnbrqkPNBRQK')
    for rank in ranks:
        file_count = 0
        for char in rank:
            if char.isdigit():
                file_count += int(char)
            elif char in valid_pieces:
                file_count += 1
            else:
                return False  # Invalid character
        if file_count != 8:
            return False  # Rank doesn't have 8 squares

    return True


if __name__ == "__main__":
    # Quick test
    import doctest
    doctest.testmod()

    # Demo
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    state = fen_to_board_state(starting_fen)

    print(f"Starting position has {len(state)} pieces:")
    for square, piece in sorted(state.items()):
        pos = get_square_position(square)
        piece_type = FEN_TO_PIECE_TYPE[piece]
        print(f"  {square}: {piece_type:10s} at ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")
