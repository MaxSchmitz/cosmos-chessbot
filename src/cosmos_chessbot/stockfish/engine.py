"""UCI protocol wrapper for Stockfish chess engine."""

import subprocess
import time
from pathlib import Path
from typing import Optional


class StockfishEngine:
    """Wrapper for Stockfish chess engine via UCI protocol."""

    def __init__(self, engine_path: str = "stockfish", timeout: float = 5.0):
        """Initialize Stockfish engine.

        Args:
            engine_path: Path to stockfish binary (or 'stockfish' if in PATH)
            timeout: Default timeout for commands in seconds
        """
        self.engine_path = engine_path
        self.timeout = timeout
        self._process: Optional[subprocess.Popen] = None

    def __enter__(self):
        """Start engine on context entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop engine on context exit."""
        self.stop()

    def start(self) -> None:
        """Start the Stockfish engine process."""
        self._process = subprocess.Popen(
            [self.engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )

        # Initialize UCI
        self._send("uci")
        self._read_until("uciok")

        self._send("isready")
        self._read_until("readyok")

    def stop(self) -> None:
        """Stop the Stockfish engine process."""
        if self._process is not None:
            self._send("quit")
            self._process.wait(timeout=2)
            self._process = None

    def new_game(self) -> None:
        """Start a new game."""
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")

    def get_best_move(
        self,
        fen: Optional[str] = None,
        moves: Optional[list[str]] = None,
        depth: int = 20,
        movetime: Optional[int] = None,
    ) -> str:
        """Get best move for current position.

        Args:
            fen: FEN string for position (if None, uses startpos)
            moves: List of moves in UCI format (e.g., ['e2e4', 'e7e5'])
            depth: Search depth
            movetime: Time to search in milliseconds

        Returns:
            Best move in UCI format (e.g., 'e2e4')
        """
        # Set position
        if fen:
            pos_cmd = f"position fen {fen}"
        else:
            pos_cmd = "position startpos"

        if moves:
            pos_cmd += " moves " + " ".join(moves)

        self._send(pos_cmd)

        # Start search
        if movetime:
            self._send(f"go movetime {movetime}")
        else:
            self._send(f"go depth {depth}")

        # Wait for bestmove
        lines = self._read_until("bestmove", timeout=30.0)
        best_line = [ln for ln in lines if ln.startswith("bestmove")][-1]

        # Parse: "bestmove e2e4" or "bestmove e2e4 ponder e7e5"
        return best_line.split()[1]

    def _send(self, command: str) -> None:
        """Send command to engine."""
        if self._process is None:
            raise RuntimeError("Engine not started")
        self._process.stdin.write((command + "\n").encode())
        self._process.stdin.flush()

    def _read_until(self, token: str, timeout: Optional[float] = None) -> list[str]:
        """Read lines until token is found.

        Args:
            token: Token to search for
            timeout: Timeout in seconds

        Returns:
            List of lines read (including the one with token)
        """
        if self._process is None:
            raise RuntimeError("Engine not started")

        timeout = timeout or self.timeout
        lines = []
        t0 = time.time()

        while time.time() - t0 < timeout:
            line = self._process.stdout.readline().decode(errors="replace").strip()
            if not line:
                continue
            lines.append(line)
            if token in line:
                return lines

        raise TimeoutError(
            f"Did not see token '{token}' within {timeout}s. Got: {lines[-10:]}"
        )
