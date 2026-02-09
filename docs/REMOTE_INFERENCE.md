# Remote Cosmos-Reason2 Inference (Brev GPU Server)

## Overview

The local machine (connected to the SO-101 arm) runs perception (YOLO-DINO) and manipulation (PPO policy) locally. All Cosmos-Reason2 reasoning is offloaded to a brev GPU server via HTTP.

```
 Local Machine (SO-101)                    Brev GPU Server (H100)
 ──────────────────────                    ──────────────────────

 YOLO-DINO perception (local)              Cosmos-Reason2-8B
 PPO policy (local)                         ├── /reason/analyze_game
 Stockfish (local)                          ├── /reason/detect_move
 Cameras + Robot (local)                    ├── /reason/action
                     ── HTTP POST ─────►    ├── /reason/correction
                     ◄── JSON response ──   └── /perceive (optional)
```

## Setup

### On Brev Server (H100 / L40S)

```bash
git clone https://github.com/MaxSchmitz/cosmos-chessbot.git
cd cosmos-chessbot
uv sync

# Authenticate with HuggingFace (Cosmos-Reason2 is gated)
uvx huggingface-cli login

# Start server
uv run python scripts/cosmos_server.py --host 0.0.0.0 --port 8000
```

Wait for: `Models loaded and ready!`

### On Local Machine (SO-101)

```bash
# Verify connection
curl http://your-brev-server:8000/health

# Run with remote reasoning
uv run cosmos-chessbot \
  --cosmos-server http://your-brev-server:8000 \
  --game-mode full-game --color white
```

## Server Endpoints

### `GET /health`

Health check.

```bash
curl http://your-brev-server:8000/health
```

```json
{
    "status": "healthy",
    "perception_loaded": true,
    "reasoning_loaded": true
}
```

### `POST /perceive`

Board state perception (FEN extraction from image). Optional — the local machine typically uses YOLO-DINO for perception instead.

```bash
curl -X POST http://your-brev-server:8000/perceive \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<base64-png>", "temperature": 0.1}'
```

```json
{
    "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "confidence": 0.94,
    "anomalies": [],
    "raw_response": "..."
}
```

### `POST /reason/action`

Pre-action physical reasoning — obstacles, grasp strategy, trajectory.

```bash
curl -X POST http://your-brev-server:8000/reason/action \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<base64-png>",
    "move_uci": "e2e4",
    "from_square": "e2",
    "to_square": "e4"
  }'
```

```json
{
    "obstacles": ["pawn on d3"],
    "adjacent_pieces": ["pawn on d2", "pawn on f2"],
    "grasp_strategy": "Top-pinch grasp on pawn, approach from above",
    "trajectory_advice": "Lift 5cm, arc over to e4",
    "risks": ["Adjacent pieces could be bumped"],
    "confidence": 0.91,
    "reasoning": "The pawn on e2 needs to move to e4..."
}
```

### `POST /reason/trajectory`

Action CoT trajectory planning — outputs normalized 2D pixel waypoints (0-1000) for the gripper trajectory.

```bash
curl -X POST http://your-brev-server:8000/reason/trajectory \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<base64-png>",
    "move_uci": "e2e4",
    "from_square": "e2",
    "to_square": "e4",
    "piece_type": "pawn"
  }'
```

```json
{
    "waypoints": [
        {"point_2d": [553, 728], "label": "above e2"},
        {"point_2d": [553, 768], "label": "grasp e2"},
        {"point_2d": [553, 474], "label": "lift"},
        {"point_2d": [553, 474], "label": "above e4"},
        {"point_2d": [553, 554], "label": "place e4"}
    ],
    "move_uci": "e2e4",
    "reasoning": "The pawn needs to move from e2 to e4. Vertical lift, horizontal traverse...",
    "confidence": 0.88
}
```

### `POST /reason/verify_goal`

Post-action visual goal verification — checks whether the move physically succeeded.

```bash
curl -X POST http://your-brev-server:8000/reason/verify_goal \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<base64-png>",
    "move_uci": "e2e4",
    "from_square": "e2",
    "to_square": "e4",
    "piece_type": "pawn"
  }'
```

```json
{
    "success": true,
    "reason": "Piece correctly placed on target square, stable and upright",
    "physical_issues": [],
    "confidence": 0.93,
    "reasoning": "Looking at the board after the move, the piece appears correctly centered..."
}
```

### `POST /reason/analyze_game`

Turn detection from video frames.

```bash
curl -X POST http://your-brev-server:8000/reason/analyze_game \
  -H "Content-Type: application/json" \
  -d '{"frames_base64": ["<frame1>", "<frame2>", "<frame3>", "<frame4>"]}'
```

```json
{
    "whose_turn": "robot",
    "opponent_moving": false,
    "should_robot_act": true,
    "reasoning": "The opponent's hand has returned to their side...",
    "confidence": 0.92
}
```

### `POST /reason/detect_move`

Opponent move detection from video frames.

```bash
curl -X POST http://your-brev-server:8000/reason/detect_move \
  -H "Content-Type: application/json" \
  -d '{"frames_base64": ["<f1>", "<f2>", "<f3>", "<f4>", "<f5>", "<f6>", "<f7>", "<f8>"]}'
```

```json
{
    "move_occurred": true,
    "from_square": "e7",
    "to_square": "e5",
    "piece_type": "pawn",
    "confidence": 0.88,
    "reasoning": "I observed the opponent move a pawn from e7 to e5..."
}
```

### `POST /reason/correction`

Post-failure correction planning.

```bash
curl -X POST http://your-brev-server:8000/reason/correction \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<base64-png>",
    "expected_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "actual_fen": "rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "differences": ["e4: expected P, found empty", "e3: expected empty, found P"]
  }'
```

```json
{
    "physical_cause": "Piece slipped during placement",
    "correction_needed": "Re-grasp piece from e3 and place on e4",
    "obstacles": ["Piece is off-center"],
    "confidence": 0.88,
    "reasoning": "The piece did not land on the intended square..."
}
```

## How It Works

The orchestrator uses `RemoteChessGameReasoning` (in `reasoning/remote_reasoning.py`) as a drop-in replacement for the local `ChessGameReasoning`. The switch is automatic:

```python
# In orchestrator.__init__:
if config.cosmos_server_url:
    self.game_reasoning = RemoteChessGameReasoning(server_url=config.cosmos_server_url)
else:
    self.game_reasoning = ChessGameReasoning(model_name=config.cosmos_model)
```

Both classes expose the same interface (`analyze_game_state`, `detect_move`, `reason_about_action`, `plan_correction`), so the rest of the orchestrator is unchanged.
