# Cosmos Chessbot

**Bridging Moravec's Paradox with Physical AI Reasoning**

## Overview

In 1996, a computer defeated the world chess champion -- but needed a human to move the pieces.
In 2016, AlphaGo defeated the world champion in Go -- but still could not act in the physical world.
Today, AI systems can solve Olympiad-level math problems, yet struggle with tasks humans find trivial, like picking up a chess piece.

This mismatch is known as **Moravec's Paradox**:

> Tasks that are cognitively "hard" for humans are often easy for machines, while tasks that are physically "easy" for humans remain exceptionally difficult for machines.

Cosmos Chessbot demonstrates how modern Physical AI models can help close this gap.

We build a robotic chess system where:
- **Symbolic intelligence** (chess strategy) is solved
- **Physical manipulation** is imperfect and uncertain
- **Cosmos-Reason2** acts as a reasoning supervisor that connects the two

The result is a robot that does not just choose the right move -- it can reason about the physical world, detect failures, and recover when reality disagrees with intention.

## Quick Start

### Prerequisites

You need two machines:

1. **GPU server** -- An NVIDIA GPU (H100 or L40S) to run Cosmos-Reason2 inference. We use [Brev](https://docs.nvidia.com/brev/latest/quick-start.html) with the Isaac Sim container.

2. **Local machine** -- A computer with:
   - An [SO-101 follower arm](https://github.com/huggingface/lerobot/blob/main/examples/robots/so101_follower/README.md) connected via USB
   - Two cameras: one overhead/egocentric camera and one wrist/gripper camera
   - [Stockfish](https://stockfishchess.org/download/) installed
   - [uv](https://docs.astral.sh/uv/getting-started/installation/) installed

### 1. Set up the GPU server

You need a GPU server with ~34GB VRAM for Cosmos-Reason2 and ~8GB for pi0.5 inference. We use [Brev](https://docs.nvidia.com/brev/latest/quick-start.html) with an L40S instance.

```bash
ssh <your-brev-instance>
git clone https://github.com/MaxSchmitz/cosmos-chessbot.git
cd cosmos-chessbot
pip install uv && uv sync
```

Log in to Hugging Face (Cosmos-Reason2 and PaliGemma are gated models):

```bash
uvx huggingface-cli login
```

Start both servers:

```bash
# Cosmos-Reason2 reasoning server (port 8000)
uv run python scripts/cosmos_server.py --host 0.0.0.0 --port 8000 &

# Pi0.5 VLA inference server (port 8001)
uv run python scripts/serve_pi05.py \
    --checkpoint outputs/pi05_hud/checkpoints/005000/pretrained_model/ \
    --port 8001 &
```

Set up SSH tunnels from your local machine:

```bash
ssh -L 8000:localhost:8000 -L 8001:localhost:8001 <your-brev-instance>
```

Verify both servers:

```bash
curl http://localhost:8000/health
# Pi0.5 health check is implicit -- it sends metadata on WebSocket connect
```

### 2. Set up the local machine

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already.

Clone the repo and install dependencies:

```bash
git clone https://github.com/MaxSchmitz/cosmos-chessbot.git
cd cosmos-chessbot
uv sync
```

### 3. Run the system

Single move with pi0.5 + HUD (recommended):

```bash
uv run cosmos-chessbot \
  --policy pi05 \
  --cosmos-server http://localhost:8000 \
  --pi05-server ws://localhost:8001 \
  --overhead-camera 1 --wrist-camera 0
```

Full game as white:

```bash
uv run cosmos-chessbot \
  --policy pi05 \
  --cosmos-server http://localhost:8000 \
  --pi05-server ws://localhost:8001 \
  --game-mode full-game --color white
```

Full game as black (limit to 20 moves):

```bash
uv run cosmos-chessbot \
  --policy pi05 \
  --cosmos-server http://localhost:8000 \
  --pi05-server ws://localhost:8001 \
  --game-mode full-game --color black --moves 20
```

## System Architecture

```
 Local Machine (SO-101)                     Brev GPU Server (L40S)
 ──────────────────────                     ──────────────────────
 Cameras (egocentric + wrist)               Cosmos-Reason2 Reasoning
         │                                    ├── Turn detection (video)
         ▼                                    ├── Move detection (video)
 YOLO-DINO (perception -> FEN)               ├── Action reasoning (image)
         │                                    └── Correction planning (image)
         ▼                                          ▲
 Stockfish (best move)                              │
         │                                     HTTP (reasoning
         ▼                                      queries only)
 Cosmos-Reason2 (intent + constraints) ─────────────┘
         │
         ▼                                   Pi0.5 Inference Server
 HUD overlay (green=pick, magenta=place)      (WebSocket, port 8001)
         │                                          ▲
         ▼                                          │
 Pi0.5 VLA Policy ──────────────────────────────────┘
   (chunked closed-loop control)
         │
         ▼
 SO-101 Robot Arm (physical execution)
         │
         ▼
 Verification (FEN comparison)
         │
         ▼
 Cosmos-Reason2 (failure diagnosis) ────────► Brev Server
```

### Key Design Principles

- **Cosmos-Reason2** never outputs motor commands -- it reasons about the physical world
- **Pi0.5** never reasons about chess or rules -- it executes manipulation guided by HUD markers
- **HUD overlay** decouples "what to move" from "how to move" -- a single visual policy handles any chess move
- **YOLO-DINO** handles perception locally -- fast, reliable, no GPU server round-trip
- **Cosmos-Reason2** handles reasoning remotely -- egocentric embodied AI on brev GPU
- Each model is used exactly where it is strongest

## How Cosmos-Reason2 Is Used

Cosmos-Reason2 is used as an **embodied reasoning supervisor**, following the same egocentric reasoning pattern as the [IntBot showcase](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/reason2/intbot_showcase/inference.html). All prompts use first-person, robot-centric framing.

### 1. Turn Detection (Video -> Game State)

Watches egocentric video to determine whose turn it is -- multi-agent social reasoning.

```
Input:  4-8 video frames from egocentric camera
Prompt: "Watch this video from my egocentric camera...
         Whose turn is it? Is my opponent currently making a move?
         Should I make my move now, or should I wait?"

Output: { "whose_turn": "robot",
          "opponent_moving": false,
          "should_robot_act": true,
          "confidence": 0.92 }
```

### 2. Move Detection (Video -> Opponent's Move)

Identifies what move the opponent just made from video -- temporal physical reasoning.

```
Input:  8 video frames showing opponent's move
Prompt: "My opponent just made a chess move...
         Which piece did they move? Where did it start? Where did it end?"

Output: { "move_occurred": true,
          "from_square": "e7",
          "to_square": "e5",
          "piece_type": "pawn",
          "confidence": 0.88 }
```

### 3. Action Reasoning (Image -> Physical Plan)

Plans how to physically execute a move -- obstacle avoidance, grasp strategy.

```
Input:  Current egocentric camera image + move (e.g., "e2e4")
Prompt: "I need to execute chess move e2e4...
         What obstacles are in the path? What grasp strategy should I use?
         What's the safest trajectory?"

Output: { "obstacles": ["pawn on d3", "knight on f3"],
          "grasp_strategy": "Top-pinch grasp on pawn, approach from above",
          "trajectory_advice": "Lift 5cm, arc to avoid knight",
          "risks": ["Adjacent pieces could be bumped"],
          "confidence": 0.91 }
```

### 4. Correction Planning (Image -> Recovery)

Diagnoses what went wrong physically and plans correction after a failed move.

```
Input:  Current image + expected FEN + actual FEN + differences
Prompt: "I tried to execute a chess move but the result is incorrect...
         What physically went wrong? What correction is needed?"

Output: { "physical_cause": "Piece slipped during placement",
          "correction_needed": "Re-grasp from current position",
          "obstacles": ["Piece is off-center"],
          "confidence": 0.88 }
```

### 5. Trajectory Planning -- Action CoT (Image -> Pixel Waypoints)

Plans a 2D end-effector trajectory in pixel coordinates, following the Cosmos-Reason2 Action CoT format. The chess board is a known flat plane, so pixel waypoints convert to 3D board-plane coordinates via homography.

```
Input:  Current egocentric camera image + move (e.g., "e2e4")
Prompt: "I need to execute chess move e2e4. Specify the 2D trajectory
         my gripper should follow as pixel coordinates in the image..."

Output: { "waypoints": [
            {"point_2d": [420, 380], "label": "above e2"},
            {"point_2d": [420, 410], "label": "descend to e2"},
            {"point_2d": [420, 410], "label": "grasp e2"},
            {"point_2d": [420, 350], "label": "lift clear"},
            {"point_2d": [420, 260], "label": "above e4"},
            {"point_2d": [420, 290], "label": "place e4"},
            {"point_2d": [420, 290], "label": "release"}
          ],
          "reasoning": "Vertical lift, horizontal traverse at safe height...",
          "confidence": 0.88 }
```

### 6. Goal Verification (Post-Action Image -> Success/Failure)

Visually verifies the physical outcome after the robot executes a move. Catches issues that FEN comparison misses: piece tipped over, adjacent pieces bumped, gripper didn't release.

```
Input:  Post-action egocentric camera image + move details
Prompt: "I just attempted chess move e2e4. Has the robot successfully
         placed the pawn on e4? Is it stable? Were adjacent pieces bumped?"

Output: { "success": true,
          "reason": "Piece correctly placed, stable and upright",
          "physical_issues": [],
          "confidence": 0.93 }
```

## Core Components

### 1. YOLO-DINO (Local Perception)

Two-stage piece detection running locally (no GPU server needed):
- **YOLO26** detects piece bounding boxes (mAP50=0.986) and board corners (mAP50=0.995)
- **DINO vits8 + MLP** classifies piece types from cropped detections (91% accuracy)
- Selective DINO: only invoked on low-confidence YOLO detections, cutting inference from 5s to 1.5s
- Board corners detected via YOLO pose model, mapped to chess coordinates via homography

### 2. Stockfish (Symbolic Chess Engine)

Standard UCI chess engine for move selection:
- Deterministic, well-understood symbolic planning
- No perception and no embodiment -- by design

### 3. Pi0.5 VLA Policy (Fine-tuned)

A [pi0.5](https://www.physicalintelligence.company/blog/pi0-5) Vision-Language-Action model fine-tuned on real-robot chess demonstrations:
- **Architecture**: PaliGemma-3B backbone with action head, outputting 50-step action chunks
- **Training data**: ~50 episodes of pick-and-place demonstrations recorded with the SO-101
- **Task encoding**: Visual HUD overlay on the egocentric camera -- green circle on the source piece, magenta circle on the target square
- **Inference**: Remote WebSocket server on GPU, chunked execution at 30fps on the local robot
- **Key insight**: One visual policy handles any chess move. The HUD overlay encodes "what to move where" visually, so the policy only needs to learn "pick highlighted, place at target."

### 4. HUD Overlay System

The HUD (Heads-Up Display) overlay bridges symbolic chess decisions and physical execution:

1. The chess engine decides a move (e.g., e2 to e4)
2. YOLO pose model detects the 4 board corners in the camera image
3. A homography maps chess squares to pixel coordinates
4. Green and magenta circles are drawn on the camera image at source and target
5. The pi0.5 policy sees these markers and executes the pick-and-place

This decouples move selection from motor control -- the same policy handles any move, captures (remove then place), and even castling (two sequential moves).

### 5. SO-101 Robotic Arm

- 5 arm joints + 1 gripper
- Egocentric camera for global perception
- Wrist camera for close-range grasp verification

## Control Loop

### Single-Move Mode (`--game-mode single-move`)

Each move follows this loop:

1. **Sense** -- Capture egocentric and wrist camera images
2. **Perceive** -- YOLO-DINO extracts board state (FEN)
3. **Plan** -- Stockfish computes best move via UCI
4. **Compile Intent** -- Cosmos-Reason2 reasons about physical constraints + plans 2D pixel trajectory (Action CoT)
5. **Act** -- Pi0.5 policy executes the manipulation on SO-101 (HUD overlay guides the pick-and-place)
6. **Verify** -- Two-stage: Cosmos visual goal verification, then FEN comparison
7. **Recover** -- If verification fails, Cosmos-Reason2 diagnoses the physical failure and plans correction

### Full-Game Mode (`--game-mode full-game`)

Adds turn detection and opponent move detection between robot moves:

1. **Wait for Opponent** -- Cosmos-Reason2 watches video, detects when opponent's turn is complete
2. **Detect Opponent Move** -- Cosmos-Reason2 identifies what piece moved and where
3. **Robot's Turn** -- Full single-move loop (sense -> perceive -> plan -> act -> verify -> recover)
4. **Repeat** -- Natural turn-taking until game over

This is the key Cosmos-Reason2 showcase: temporal video reasoning about multi-agent physical interactions.

## Why Chess?

Chess is not the goal -- it is the **testbed**.

Chess provides:
- A well-defined symbolic world (rules, moves, legality)
- A physically challenging manipulation task (small pieces, precise placement)
- Clear success/failure criteria (FEN comparison)
- Natural multi-agent interaction (turn-taking with opponent)
- Easy evaluation and reproducibility

By choosing chess, we isolate the hardest unsolved problem: **making symbolic decisions real in the physical world**.

## Repository Structure

```
cosmos-chessbot/
├── src/cosmos_chessbot/          # Main package
│   ├── orchestrator/             # Control loop
│   ├── vision/                   # Perception (YOLO-DINO, board segmentation)
│   ├── reasoning/                # Cosmos-Reason2 reasoning
│   ├── stockfish/                # UCI engine wrapper
│   ├── policy/                   # Manipulation policies
│   └── schemas/                  # Pydantic I/O models
├── scripts/
│   ├── cosmos_server.py          # Cosmos-Reason2 GPU server
│   ├── training/                 # Model training scripts
│   ├── evaluation/               # Benchmarking and evaluation
│   ├── data/                     # Dataset generation and conversion
│   ├── hardware/                 # Robot calibration and deployment
│   ├── testing/                  # Integration tests and smoke tests
│   └── visualization/            # Visualization and inspection
├── docs/                         # Guides, architecture docs, model cards
├── data/                         # Datasets and USD assets
├── models/                       # Pre-trained model weights
└── config/                       # Configuration files
```

## Reference

### Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/perceive` | POST | Board state perception (FEN extraction) |
| `/reason/action` | POST | Pre-action physical reasoning |
| `/reason/trajectory` | POST | Action CoT trajectory planning (2D pixel waypoints) |
| `/reason/verify_goal` | POST | Post-action visual goal verification |
| `/reason/analyze_game` | POST | Turn detection from video frames |
| `/reason/detect_move` | POST | Opponent move detection from video |
| `/reason/correction` | POST | Post-failure correction planning |
| `/reason/critique_episode` | POST | Full episode video critique (RL critic) |
| `/reason/analyze_board` | POST | Board scene analysis (position, phase, condition) |

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--game-mode` | `single-move` | `single-move` or `full-game` |
| `--color` | `white` | Robot plays as `white` or `black` |
| `--policy` | `pi05` | Policy: `pi05` (VLA with HUD overlay) |
| `--cosmos-server` | none | Brev Cosmos server URL (e.g., `http://gpu:8000`) |
| `--pi05-server` | `ws://localhost:8001` | Pi0.5 WebSocket server URL |
| `--pi05-fps` | `30` | Pi0.5 action execution rate |
| `--pi05-steps` | `500` | Max action steps per pi0.5 move |
| `--overhead-camera` | `0` | Egocentric camera device ID |
| `--wrist-camera` | `1` | Wrist camera device ID |
| `--stockfish` | `stockfish` | Path to Stockfish binary |
| `--perception` | `yolo` | Perception: `yolo` (YOLO-DINO FEN detection) |
| `--moves` | unlimited | Maximum moves to execute |
| `--dry-run` | off | Skip robot connection (vision + planning only) |
| `--verbose` / `--quiet` | normal | Logging verbosity |

### Pi0.5 Inference Server

Start the pi0.5 server on your GPU machine, pointing to the fine-tuned HUD checkpoint:

```bash
# On GPU server
uv run python scripts/serve_pi05.py \
    --checkpoint outputs/pi05_hud/checkpoints/005000/pretrained_model/ \
    --port 8001
```

Then tunnel the port to your local machine:

```bash
ssh -L 8001:localhost:8001 <your-brev-instance>
```

### Pi0.5 Standalone Episode

Run pi0.5 directly (outside the orchestrator) for testing:

```bash
uv run python scripts/hardware/run_pi05_episode.py \
    --source e2 --target e4 \
    --server-url ws://localhost:8001
```

## Evaluation

### Perception Accuracy

Evaluated on [ChessReD2k](https://arxiv.org/abs/2310.04086) test set (209 images):

| Metric | Score |
|--------|-------|
| Per-square accuracy | 99.67% |
| Exact FEN match | 83.7% |
| YOLO piece detection mAP50 | 0.986 |
| YOLO corner detection mAP50 | 0.995 |
| DINO-MLP piece classification | 91.0% val accuracy |

```bash
# Run FEN evaluation on ChessReD2k
uv run python scripts/evaluation/evaluate_fen_accuracy.py \
    --dataset data/chessred2k_pose/ \
    --yolo-pieces models/yolo_pieces.pt \
    --yolo-corners models/yolo_corners.pt \
    --dino-mlp models/dino_mlp/dino_mlp_best.pth
```

### Cosmos Reasoning

Cosmos-Reason2 reasoning quality is evaluated qualitatively through the demo -- chain-of-thought outputs are shown alongside camera footage. Each endpoint produces structured JSON with a confidence score.

## Reproducing from Scratch

### 1. Train perception models

```bash
# YOLO piece detection
uv run python scripts/training/train_yolo26_pieces.py \
    --data data/combined_pieces/data.yaml --epochs 100

# YOLO corner pose detection
uv run python scripts/training/train_yolo26_corners.py \
    --data data/combined_corners/data.yaml --epochs 100

# DINO-MLP piece classifier
uv run python scripts/training/train_dino_mlp_classifier.py \
    --data-dir data/combined_pieces/
```

### 2. Collect robot demonstrations

Record ~50 pick-and-place episodes with teleoperation:

```bash
uv run python scripts/hardware/collect_episodes.py \
    --repo-id <your-hf-repo> --fps 30
```

### 3. Apply HUD overlays to training data

```bash
uv run python scripts/data/apply_hud_to_dataset.py \
    --src-repo-id <raw-dataset> --dst-repo-id <hud-dataset>
```

### 4. Fine-tune Pi0.5

```bash
# On GPU server
uv run python scripts/training/train_pi05.py \
    --dataset <hud-dataset> --steps 12000
```

## Current Status

- [x] Stockfish UCI integration
- [x] YOLO-DINO board perception (local, real-time, mAP50=0.984)
- [x] Cosmos-Reason2 embodied reasoning (9 endpoints: turn detection, move detection, action reasoning, trajectory planning, goal verification, correction planning, episode critique, board analysis, perception)
- [x] Remote reasoning server with full endpoint coverage
- [x] Pi0.5 VLA fine-tuning on real-robot demonstrations
- [x] HUD overlay system for visual move encoding
- [x] Pi0.5 chunked closed-loop execution via WebSocket server
- [x] Full orchestrator: sense -> perceive -> plan -> compile -> act -> verify -> recover
- [x] Full game loop with turn detection and opponent move detection
- [x] FEN verification and automated recovery
- [x] MCP server with 20 tools for interactive control

## License

- **Cosmos-Reason2**: NVIDIA Open Model License (Apache 2.0 compatible)
- **Pi0.5 / PaliGemma**: Respective model licenses (used for VLA fine-tuning)
- **Stockfish**: GPLv3 (used as an external engine)

## Acknowledgements

- [NVIDIA Cosmos](https://github.com/NVIDIA/Cosmos) team for Physical AI models and tooling
- [Physical Intelligence](https://www.physicalintelligence.company/) for the pi0.5 VLA architecture
- [Hugging Face LeRobot](https://github.com/huggingface/lerobot) for the robot learning framework and pi0.5 integration
- [The Robot Studio / Hugging Face](https://github.com/TheRobotStudio/SO-ARM100) for the SO-100/SO-101 robot arm
- [ChessReD2k](https://arxiv.org/abs/2310.04086) (Masouris et al., VISAPP 2024) for the chess recognition dataset used to train our YOLO piece detection and board corner models
- [Stockfish](https://stockfishchess.org/) developers for the UCI chess engine
- [Claude Code](https://claude.ai/claude-code) (Anthropic) for assistance writing the codebase
