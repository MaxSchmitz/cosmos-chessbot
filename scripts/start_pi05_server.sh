#!/bin/bash
# Start the pi0.5 inference server on the brev GPU machine.
#
# Usage (on brev):
#   ./scripts/start_pi05_server.sh          # foreground
#   ./scripts/start_pi05_server.sh --bg     # background (nohup)
#
# From local machine, set up SSH tunnel:
#   ssh -f -N -L 8001:localhost:8001 ubuntu@isaacsim
#
# Then in Python:
#   from cosmos_chessbot.policy.pi05_policy import PI05Policy
#   policy = PI05Policy(host="localhost", port=8001)

set -euo pipefail

PORT="${PI05_PORT:-8001}"
OPENPI_DIR="${OPENPI_DIR:-$HOME/openpi}"
UV="${UV:-$HOME/.local/bin/uv}"

cd "$OPENPI_DIR"

if [[ "${1:-}" == "--bg" ]]; then
    echo "Starting pi0.5 server in background on port $PORT..."
    nohup "$UV" run scripts/serve_policy.py --env=DROID --port="$PORT" > /tmp/pi05_server.log 2>&1 &
    echo "PID: $!"
    echo "Logs: /tmp/pi05_server.log"
    echo "Waiting for model download + load..."
    tail -f /tmp/pi05_server.log &
    TAIL_PID=$!
    # Wait until server is listening
    while ! grep -q "server listening" /tmp/pi05_server.log 2>/dev/null; do
        sleep 2
    done
    kill $TAIL_PID 2>/dev/null || true
    echo ""
    echo "Server ready on port $PORT"
else
    echo "Starting pi0.5 server on port $PORT (foreground)..."
    exec "$UV" run scripts/serve_policy.py --env=DROID --port="$PORT"
fi
