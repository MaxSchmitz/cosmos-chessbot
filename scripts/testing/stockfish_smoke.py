# scripts/stockfish_smoke.py
import subprocess
import sys
import time

def send(p: subprocess.Popen, line: str) -> None:
    p.stdin.write((line + "\n").encode())
    p.stdin.flush()

def read_until(p: subprocess.Popen, token: str, timeout_s: float = 5.0) -> list[str]:
    lines = []
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        line = p.stdout.readline().decode(errors="replace").strip()
        if not line:
            continue
        lines.append(line)
        if token in line:
            return lines
    raise TimeoutError(f"Did not see token '{token}' within {timeout_s}s. Got: {lines[-10:]}")

def main() -> int:
    engine_path = sys.argv[1] if len(sys.argv) > 1 else "stockfish"
    p = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )

    send(p, "uci")
    read_until(p, "uciok", timeout_s=5)

    send(p, "isready")
    read_until(p, "readyok", timeout_s=5)

    send(p, "ucinewgame")
    send(p, "isready")
    read_until(p, "readyok", timeout_s=5)

    send(p, "position startpos")
    send(p, "go depth 10")

    # Wait for bestmove
    lines = read_until(p, "bestmove", timeout_s=10)
    best = [ln for ln in lines if ln.startswith("bestmove")][-1]
    print(best)

    send(p, "quit")
    p.wait(timeout=2)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

