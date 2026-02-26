"""Chess engine and board state tools."""

from __future__ import annotations

import json

import chess
from mcp.server.fastmcp import Context

from .server import mcp
from .state import ServerState


@mcp.tool()
def get_best_move(ctx: Context, fen: str = "", depth: int = 20) -> str:
    """Get Stockfish's best move for a position.

    Args:
        fen: FEN string. If empty, uses the current internal board state.
             Board-only FEN (no turn/castling info) will have metadata appended
             from the internal board.
        depth: Search depth (default 20).

    Returns JSON with best_move in UCI format and the FEN used.
    """
    state: ServerState = ctx.request_context.lifespan_context
    try:
        if not fen:
            fen = state.board.fen()
        elif len(fen.split()) == 1:
            turn = "w" if state.board.turn == chess.WHITE else "b"
            fen = f"{fen} {turn} KQkq - 0 1"

        best = state.engine.get_best_move(fen=fen, depth=depth)
        return json.dumps({"best_move": best, "fen_used": fen})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_board_state(ctx: Context) -> str:
    """Get the current internal chess board state.

    Returns FEN, move history, legal moves, and game status.
    This is the internal tracking board -- it may differ from the
    physical board if moves haven't been pushed.
    """
    state: ServerState = ctx.request_context.lifespan_context
    board = state.board
    legal = [m.uci() for m in board.legal_moves]
    history = [m.uci() for m in board.move_stack]

    return json.dumps({
        "fen": board.fen(),
        "board_fen": board.board_fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "fullmove_number": board.fullmove_number,
        "legal_moves": legal,
        "move_history": history,
        "is_check": board.is_check(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
        "is_game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
    }, indent=2)


@mcp.tool()
def push_move(ctx: Context, move_uci: str) -> str:
    """Push a UCI move to the internal board for tracking.

    Use this after executing a move on the physical board (robot or opponent)
    to keep the internal state in sync.

    Args:
        move_uci: Move in UCI format (e.g. 'e2e4', 'g1f3').
    """
    state: ServerState = ctx.request_context.lifespan_context
    try:
        move = chess.Move.from_uci(move_uci)
        if move not in state.board.legal_moves:
            return json.dumps({
                "success": False,
                "error": f"Illegal move: {move_uci}",
                "legal_moves": [m.uci() for m in state.board.legal_moves],
            })
        state.board.push(move)
        return json.dumps({
            "success": True,
            "move": move_uci,
            "fen": state.board.fen(),
            "turn": "white" if state.board.turn == chess.WHITE else "black",
        })
    except (chess.InvalidMoveError, ValueError) as e:
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
def reset_board(ctx: Context, fen: str = "") -> str:
    """Reset the internal board to starting position or a custom FEN.

    Also resets the Stockfish engine's game state.

    Args:
        fen: Custom FEN to set. If empty, resets to standard starting position.
    """
    state: ServerState = ctx.request_context.lifespan_context
    try:
        if fen:
            state.board.set_fen(fen)
        else:
            state.board.reset()
        state.engine.new_game()
        return json.dumps({"success": True, "fen": state.board.fen()})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
