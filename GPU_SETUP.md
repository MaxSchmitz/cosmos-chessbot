# GPU Server Setup Guide

This guide explains how to run Cosmos-Reason2 on a remote GPU server (H100/GB200) while developing locally.

## Architecture

```
Local Machine (SO-101 + Cameras)    GPU Server (H100/GB200)
┌────────────────────────────┐     ┌──────────────────────┐
│                            │     │                      │
│  Camera Capture            │     │  Cosmos-Reason2      │
│  Stockfish                 │────▶│  Inference Server    │
│  π₀.₅ (future)            │◀────│                      │
│  Orchestrator              │     │  FastAPI Endpoint    │
│                            │     │                      │
└────────────────────────────┘     └──────────────────────┘
       HTTP Requests                  GPU Inference
```

## GPU Server Setup

### 1. Clone Repository on GPU Server

```bash
ssh your-gpu-server

git clone <your-repo-url>
cd cosmos-chessbot
```

### 2. Install Dependencies

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

### 3. Verify GPU

```bash
nvidia-smi

# Should show H100, GB200, or other supported GPU
# Check CUDA version matches requirements (12.8+ or 13.0+)
```

### 4. Start Cosmos Server

```bash
# Default: binds to 0.0.0.0:8000
uv run python scripts/cosmos_server.py

# Custom port
uv run python scripts/cosmos_server.py --port 8080

# Custom host/port
uv run python scripts/cosmos_server.py --host 0.0.0.0 --port 8000
```

Server will load Cosmos-Reason2 model on startup (this takes a minute or two).

Once you see:
```
Model loaded and ready!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

The server is ready for requests.

### 5. Test Server Health

From your local machine:

```bash
curl http://your-gpu-server:8000/health

# Should return:
# {"status":"healthy","model_loaded":true}
```

## Local Machine Setup

### 1. Test Remote Perception

```bash
# Test with a chess board image
uv run python scripts/test_remote_perception.py \
  http://your-gpu-server:8000 \
  assets/test_board.jpg
```

### 2. Run Orchestrator with Remote Cosmos

```bash
uv run cosmos-chessbot \
  --cosmos-server http://your-gpu-server:8000 \
  --overhead-camera 0 \
  --wrist-camera 1
```

The orchestrator will automatically use the remote server instead of loading Cosmos locally.

## Network Configuration

### Firewall Rules

Ensure port 8000 (or your chosen port) is open on the GPU server:

```bash
# Example for ufw
sudo ufw allow 8000/tcp
```

### SSH Tunnel (Alternative)

If you can't open firewall ports, use SSH tunneling:

```bash
# On local machine, create tunnel
ssh -L 8000:localhost:8000 your-gpu-server

# Then use localhost in your commands
uv run cosmos-chessbot --cosmos-server http://localhost:8000
```

## Production Tips

### Running Server in Background

Use tmux or screen to keep server running:

```bash
# Start tmux session
tmux new -s cosmos-server

# Start server
uv run python scripts/cosmos_server.py

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t cosmos-server
```

### Systemd Service (Optional)

For production deployment, create a systemd service:

```ini
# /etc/systemd/system/cosmos-inference.service
[Unit]
Description=Cosmos Inference Server
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/cosmos-chessbot
ExecStart=/home/your-user/.local/bin/uv run python scripts/cosmos_server.py
Restart=always

[Install]
WantedBy=multi-tier.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable cosmos-inference
sudo systemctl start cosmos-inference
```

### Monitoring

Monitor GPU usage:

```bash
watch -n 1 nvidia-smi
```

Monitor server logs:

```bash
# If using systemd
sudo journalctl -u cosmos-inference -f
```

## API Endpoints

### GET /health

Health check and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### POST /perceive

Run board state perception.

**Request:**
```json
{
  "image_base64": "<base64 encoded PNG/JPG>",
  "max_new_tokens": 2048,
  "temperature": 0.1
}
```

**Response:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "confidence": 0.95,
  "anomalies": ["white knight on f3 is tilted"],
  "raw_response": "..."
}
```

## Troubleshooting

### Server won't start
- Check CUDA version: `nvcc --version`
- Check GPU visibility: `nvidia-smi`
- Verify disk space for model download: `df -h`

### Connection refused from local machine
- Check firewall rules
- Verify server is listening: `netstat -tuln | grep 8000`
- Try SSH tunnel as workaround

### Out of memory
- Reduce image size in camera config
- Reduce `max_vision_tokens` in perception.py
- Use smaller model (if available)

### Slow inference
- Check GPU utilization: `nvidia-smi dmon`
- Consider batching (future enhancement)
- Verify not running on CPU by accident

## Performance Expectations

On H100:
- Model loading: ~30-60 seconds
- Single image inference: ~0.5-2 seconds
- Throughput: ~1-2 FPS

Network latency will add ~10-50ms depending on your setup.
