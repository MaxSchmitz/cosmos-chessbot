# Quick Start Guide

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Ensure Stockfish is installed:
```bash
# macOS
brew install stockfish

# Ubuntu
sudo apt install stockfish

# Or download from https://stockfishchess.org/download/
```

3. Verify Stockfish works:
```bash
uv run python scripts/stockfish_smoke.py
```

## Testing Components

### Test Camera Capture

Capture a frame from your overhead camera:
```bash
uv run python scripts/capture_frame.py 0 data/raw/test_frame.jpg
```

### Test Cosmos Perception

Run perception on a static chess board image:
```bash
# First, place a chess board image in assets/
uv run python scripts/test_perception.py assets/board.jpg
```

## Running the Full System

Run the orchestrator (requires cameras and robot):
```bash
uv run cosmos-chessbot --moves 5
```

Or run programmatically:
```bash
uv run python -m cosmos_chessbot.main
```

## Project Structure

```
cosmos-chessbot/
├── src/cosmos_chessbot/
│   ├── orchestrator/      # Main control loop
│   ├── vision/            # Camera + Cosmos perception
│   ├── stockfish/         # UCI engine wrapper
│   ├── policy/            # π₀.₅ interface (TODO)
│   └── schemas/           # Data schemas
├── scripts/               # Test and utility scripts
├── data/
│   ├── raw/              # Captured images
│   ├── episodes/         # Teleoperation data
│   └── eval/             # Evaluation results
└── assets/               # Test images and videos
```

## Next Steps

1. **Camera Setup**: Verify overhead and wrist cameras work
2. **Perception Testing**: Test Cosmos on various board states
3. **Data Collection**: Start collecting teleoperation episodes
4. **Policy Integration**: Integrate π₀.₅ for manipulation
5. **Closed Loop**: Test full sense → perceive → plan → act → verify loop
