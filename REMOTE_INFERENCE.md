# Quick Start: Remote Cosmos Inference

## Overview

You can now run Cosmos-Reason2 on a remote GPU server while developing locally on your SO-101 setup.

## Setup Steps

### On GPU Server (H100/GB200)

```bash
# 1. Clone and setup
git clone <your-repo> && cd cosmos-chessbot
uv sync

# 2. Start server (will load model on startup)
uv run python scripts/cosmos_server.py

# Server will bind to 0.0.0.0:8000 by default
```

Wait for: `Model loaded and ready!`

### On Local Machine (SO-101)

```bash
# 1. Test connection
curl http://your-gpu-server:8000/health

# 2. Test perception with chess board image
uv run python scripts/test_remote_perception.py \
  http://your-gpu-server:8000 \
  assets/your-board.jpg

# 3. Run full orchestrator
uv run cosmos-chessbot \
  --cosmos-server http://your-gpu-server:8000 \
  --overhead-camera 0
```

## What Changed

### New Files
- `scripts/cosmos_server.py` - FastAPI server for GPU
- `src/cosmos_chessbot/vision/remote_perception.py` - HTTP client
- `scripts/test_remote_perception.py` - Test remote inference
- `GPU_SETUP.md` - Detailed setup guide

### Modified Files
- Orchestrator now supports `--cosmos-server` flag
- Vision module exports `RemoteCosmosPerception`
- Added httpx, fastapi, uvicorn dependencies

## Testing Your Chess Board Image

You mentioned you have a chess board image in `assets/`. Test it:

```bash
# If testing locally (once you have GPU access)
uv run python scripts/test_perception.py assets/<your-image-name>

# If testing remotely (recommended for now)
# 1. Start server on GPU machine
# 2. From local machine:
uv run python scripts/test_remote_perception.py \
  http://your-gpu-server:8000 \
  assets/<your-image-name>
```

The script accepts any common image format (.jpg, .png, .bmp, etc.) - no specific naming required.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Local Machine (Your SO-101 Setup)                      │
│                                                         │
│  ┌─────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Cameras │──▶│ Orchestrator │──│ RemoteCosmos    │──┼──┐
│  └─────────┘  └──────────────┘  │ Perception      │  │  │
│                     │             └─────────────────┘  │  │
│                     ▼                                   │  │
│               ┌──────────┐                             │  │
│               │Stockfish │                             │  │
│               └──────────┘                             │  │
└─────────────────────────────────────────────────────────┘  │
                                                              │
                                         HTTP                 │
                                                              │
┌─────────────────────────────────────────────────────────┐  │
│ GPU Server (H100/GB200)                                 │  │
│                                                         │  │
│  ┌──────────────────┐          ┌──────────────────┐   │  │
│  │ cosmos_server.py │◀─────────│ Cosmos-Reason2   │   │◀─┘
│  │  FastAPI         │          │ (nvidia/2B)      │   │
│  └──────────────────┘          └──────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Next Steps

1. **Spin up GPU server** (H100/GB200 with CUDA 12.8+)
2. **Start Cosmos server** on GPU
3. **Test with your board image** using test_remote_perception.py
4. **Iterate on prompts** if FEN extraction needs tuning
5. **Begin data collection** for π₀.₅ training

See `GPU_SETUP.md` for detailed server configuration, firewall setup, and production deployment options.
