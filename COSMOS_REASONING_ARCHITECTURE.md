# Cosmos-Reason2 Embodied Reasoning Architecture

## Overview

Cosmos Chessbot uses **Cosmos-Reason2** exclusively for embodied reasoning — not perception. Following the same egocentric reasoning pattern as the [IntBot showcase](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/reason2/intbot_showcase/inference.html), we frame all prompts from the robot's first-person perspective to leverage Cosmos-Reason2's core strength: understanding multi-agent physical interactions.

Perception (image → FEN) is handled locally by YOLO-DINO, which provides fast, reliable bounding box detection without a GPU server round-trip.

## Architecture

```
 Local Machine (SO-101)                    Brev GPU Server (H100)
 ──────────────────────                    ──────────────────────

 ┌──────────────────────┐                  ┌──────────────────────────┐
 │ Cameras              │                  │ cosmos_server.py         │
 │  egocentric + wrist  │                  │  (FastAPI)               │
 └────────┬─────────────┘                  │                          │
          │                                │  Cosmos-Reason2-8B       │
          ▼                                │  ┌────────────────────┐  │
 ┌──────────────────────┐                  │  │ analyze_game_state │  │
 │ YOLO-DINO            │                  │  │ detect_move        │  │
 │ Board segmentation   │                  │  │ reason_about_action│  │
 │ + piece detection    │                  │  │ plan_correction    │  │
 │ → FEN                │                  │  └────────────────────┘  │
 └────────┬─────────────┘                  └────────────▲─────────────┘
          │                                             │
          ▼                                        HTTP POST
 ┌──────────────────────┐                        (images/video
 │ Stockfish (UCI)      │                         + prompts)
 │ → best move          │                             │
 └────────┬─────────────┘                             │
          │                                           │
          ▼                                           │
 ┌──────────────────────┐                             │
 │ Orchestrator         ├─────────────────────────────┘
 │  compile_intent()    │  ← action reasoning
 │  wait_for_opponent() │  ← turn detection
 │  detect_opponent()   │  ← move detection
 │  recover()           │  ← correction planning
 └────────┬─────────────┘
          │
          ▼
 ┌──────────────────────┐
 │ PPO Policy           │
 │ (trained in Isaac)   │
 │ → joint commands     │
 └────────┬─────────────┘
          │
          ▼
 ┌──────────────────────┐
 │ SO-101 Robot Arm     │
 └──────────────────────┘
```

## Why Cosmos-Reason2 for Reasoning (Not Perception)

| Capability | Cosmos-Reason2 | YOLO-DINO |
|-----------|----------------|-----------|
| Egocentric social reasoning | Excellent | N/A |
| Temporal video understanding | Excellent | N/A |
| Physical cause-effect reasoning | Excellent | N/A |
| Object detection / bounding boxes | Possible (future) | Excellent |
| FEN extraction from board | Not trained for this | Reliable via geometry |
| Latency | ~2-5s (LLM inference) | ~15ms (local) |

Cosmos-Reason2 excels at understanding **what is happening** in a scene — whose turn it is, what a human is doing, what went wrong physically. YOLO-DINO excels at **what is where** — detecting and locating chess pieces. Using each where it's strongest gives the best overall system.

## The Four Reasoning Modes

### 1. Turn Detection — `analyze_game_state(video_frames)`

**Purpose**: Determine whose turn it is by watching egocentric video. This is multi-agent social reasoning — understanding human intent and turn-taking dynamics.

**Robot-centric prompt** (from `game_reasoning.py`):

```
You are an embodied chess-playing robot with an egocentric camera view.
The camera view is YOUR view of the chess board and your opponent.

Watch this video from my egocentric camera and reason about the chess game.
The camera view is MY view as the robot. I am playing chess against a human opponent.

Analyze the video and answer:
1. Whose turn is it? (mine or my opponent's)
2. Is my opponent currently making a move? (reaching for pieces, moving a piece)
3. Should I make my move now, or should I wait?
```

**Input**: `list[PIL.Image]` — 4-8 video frames captured ~0.15s apart

**Output**:
```json
{
    "whose_turn": "robot" | "opponent" | "unknown",
    "opponent_moving": true | false,
    "should_robot_act": true | false,
    "reasoning": "I observe the opponent's hand has returned to their side...",
    "confidence": 0.92
}
```

**Used in**: `orchestrator.wait_for_opponent_turn()` — polls this until `should_robot_act=True`

### 2. Move Detection — `detect_move(video_frames)`

**Purpose**: Identify what move the opponent just made. Temporal reasoning about piece movement across frames.

**Robot-centric prompt**:

```
Watch this video from my egocentric camera. My opponent just made a chess move.
The camera view is MY view as the robot.

Identify what move they made:
1. Which piece did they move?
2. Where did it start? (square in algebraic notation like 'e2')
3. Where did it end? (square in algebraic notation like 'e4')
```

**Input**: `list[PIL.Image]` — 8 video frames showing the move

**Output**:
```json
{
    "move_occurred": true,
    "from_square": "e7",
    "to_square": "e5",
    "piece_type": "pawn",
    "reasoning": "I observed the opponent's hand move a pawn from e7 to e5...",
    "confidence": 0.88
}
```

**Used in**: `orchestrator.detect_opponent_move()` → pushes detected move to internal `chess.Board`

### 3. Action Reasoning — `reason_about_action(image, move_uci, from_square, to_square)`

**Purpose**: Plan physical execution of a chess move. Obstacle avoidance, grasp strategy, trajectory planning.

**Robot-centric prompt**:

```
I need to execute a chess move: e2e4
Pick up the piece from e2 and place it on e4.

Looking at my egocentric camera view, reason about:
1. What obstacles are in the path between e2 and e4?
2. What pieces are adjacent to e2 and e4?
3. What grasp strategy should I use for this piece?
4. What's the safest trajectory to avoid knocking over other pieces?
5. Are there any physical risks or challenges I should be aware of?
```

**Input**: `PIL.Image` (current egocentric view) + move details

**Output**:
```json
{
    "obstacles": ["pawn on d3", "knight on f3"],
    "adjacent_pieces": ["pawn on d2", "pawn on f2"],
    "grasp_strategy": "Top-pinch grasp on pawn, approach from above",
    "trajectory_advice": "Lift 5cm, arc over knight on f3",
    "risks": ["Adjacent pieces could be bumped during placement"],
    "confidence": 0.91
}
```

**Used in**: `orchestrator.compile_intent()` — informs the physical manipulation plan

### 4. Correction Planning — `plan_correction(image, expected_fen, actual_fen, differences)`

**Purpose**: Diagnose what went wrong physically after a failed move and plan correction.

**Robot-centric prompt**:

```
I tried to execute a chess move but the result is incorrect.

Expected board state (FEN): rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR
Actual board state (FEN): rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR

Differences found:
  e4: expected P, found empty
  e3: expected empty, found P

Looking at my egocentric camera view, reason about:
1. What physically went wrong?
2. Why did the piece end up in the wrong position?
3. What physical correction is needed to fix this?
```

**Input**: `PIL.Image` + expected/actual FEN + list of `SquareDifference`

**Output**:
```json
{
    "physical_cause": "Piece slipped during placement — gripper released 2mm too early",
    "correction_needed": "Re-grasp piece from e3 and place on e4",
    "obstacles": ["Piece is slightly off-center on wrong square"],
    "confidence": 0.88,
    "reasoning": "The piece did not land on the intended square..."
}
```

**Used in**: `orchestrator.recover()` — correction reasoning before executing a fix

## Remote Server Endpoints

When running Cosmos-Reason2 on a brev GPU server, the `RemoteChessGameReasoning` client in `reasoning/remote_reasoning.py` mirrors the local `ChessGameReasoning` interface:

| Local Method | Server Endpoint | Request Body |
|-------------|-----------------|--------------|
| `analyze_game_state(frames)` | `POST /reason/analyze_game` | `frames_base64: list[str]` |
| `detect_move(frames)` | `POST /reason/detect_move` | `frames_base64: list[str]` |
| `reason_about_action(image, ...)` | `POST /reason/action` | `image_base64, move_uci, from_square, to_square` |
| `plan_correction(image, ...)` | `POST /reason/correction` | `image_base64, expected_fen, actual_fen, differences` |

The orchestrator automatically uses `RemoteChessGameReasoning` when `--cosmos-server` is set, or `ChessGameReasoning` (local) otherwise.

## Full Game Loop Flow

```
Game starts
    │
    ├── Robot is white? ──► Robot moves first
    │                         │
    │                         ▼
    │                    ┌─────────────┐
    │                    │ sense       │ Camera capture
    │                    │ perceive    │ YOLO-DINO → FEN
    │                    │ plan        │ Stockfish → move
    │                    │ compile     │ Cosmos reasoning (action)
    │                    │ act         │ PPO policy → robot
    │                    │ verify      │ FEN comparison
    │                    │ recover?    │ Cosmos reasoning (correction)
    │                    └──────┬──────┘
    │                           │
    │                           ▼
    │                    ┌─────────────────────────────┐
    │                    │ wait_for_opponent_turn()     │
    │                    │   Cosmos: analyze_game_state │
    │                    │   (poll until should_act)    │
    │                    └──────┬──────────────────────┘
    │                           │
    │                           ▼
    │                    ┌─────────────────────────────┐
    │                    │ detect_opponent_move()       │
    │                    │   Cosmos: detect_move        │
    │                    │   Push to internal board     │
    │                    └──────┬──────────────────────┘
    │                           │
    │                           ▼
    │                    Loop back to robot's turn
    │
    └── Robot is black? ──► Wait for opponent first, then same loop
```

## References

- [Cosmos Reason2 IntBot Showcase](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/reason2/intbot_showcase/inference.html) — Embodied egocentric reasoning examples
- [Cosmos Cookbook](https://nvidia-cosmos.github.io/cosmos-cookbook/) — Recipes for Cosmos models
- [NVIDIA Cosmos](https://github.com/NVIDIA/Cosmos) — Physical AI models and tooling
