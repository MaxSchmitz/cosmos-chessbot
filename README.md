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

## System Architecture

```
 Local Machine (SO-101)                     Brev GPU Server (H100)
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
         ▼
 PPO Policy (trained in Isaac Sim)
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
- **The PPO policy** never reasons about chess or rules -- it executes manipulation
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

Plans a 2D end-effector trajectory in normalized pixel coordinates (0-1000), following the Cosmos-Reason2 Action CoT format. The chess board is a known flat plane, so pixel waypoints convert to 3D board-plane coordinates via homography.

```
Input:  Current egocentric camera image + move (e.g., "e2e4")
Prompt: "I need to execute chess move e2e4. Specify the 2D trajectory
         my gripper should follow in normalized pixel coordinates (0-1000)..."

Output: { "waypoints": [
            {"point_2d": [553, 728], "label": "above e2"},
            {"point_2d": [553, 768], "label": "grasp e2"},
            {"point_2d": [553, 474], "label": "lift"},
            {"point_2d": [553, 474], "label": "above e4"},
            {"point_2d": [553, 554], "label": "place e4"}
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

Board segmentation (Ultimate V2) + piece detection (YOLO) running locally:
- Image -> board crop -> piece bounding boxes -> FEN
- Fast inference, no GPU server round-trip needed
- Reliable bounding box detection for chess pieces

### 2. Stockfish (Symbolic Chess Engine)

Standard UCI chess engine for move selection:
- Deterministic, well-understood symbolic planning
- No perception and no embodiment -- by design

### 3. PPO RL Policy (Isaac Sim)

A PPO reinforcement learning policy trained in simulation for physical manipulation:
- **Architecture**: ActorCritic with shared backbone (256 hidden dim)
- **Observations** (21-dim): joint positions (5) + gripper (1) + end-effector pose (7) + piece-relative (3) + target-relative (3) + grasp flag (1) + phase (1)
- **Actions** (6-dim): joint targets (5) + gripper command (1)
- **Training**: Isaac Sim with rigid body physics, kinematic grasping, 32-piece typed piece pool
- **Reward curriculum**: approach -> grasp -> lift -> transport -> success, with collision and action-rate penalty ramps

### 4. SO-101 Robotic Arm

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
5. **Act** -- PPO policy executes the manipulation on SO-101
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
├── src/cosmos_chessbot/
│   ├── orchestrator/           # Main control loop (single-move + full-game)
│   │   └── orchestrator.py     # ChessOrchestrator with game loop
│   ├── vision/                 # Perception (YOLO-DINO, board segmentation)
│   │   ├── fen_detection.py    # YOLO piece detection -> FEN
│   │   ├── board_segmentation.py  # Ultimate V2 board crop
│   │   ├── llm_fen_detector.py # Claude/GPT/Gemini FEN fallback
│   │   └── camera.py           # Camera capture interface
│   ├── reasoning/              # Cosmos-Reason2 game reasoning
│   │   ├── game_reasoning.py   # Turn detection, move detection, action/correction reasoning
│   │   ├── fen_comparison.py   # FEN diff, expected FEN calculation, correction moves
│   │   └── remote_reasoning.py # HTTP client for brev GPU server
│   ├── stockfish/              # UCI engine wrapper
│   ├── policy/                 # Manipulation policies (PPO, Cosmos Policy)
│   ├── isaac/                  # Isaac Sim RL environment
│   ├── schemas/                # Pydantic models for server I/O
│   └── main.py                 # CLI entry point
├── scripts/
│   ├── cosmos_server.py        # Cosmos-Reason2 GPU server (perception + reasoning)
│   ├── train_chess_rl.py       # PPO training in Isaac Sim
│   ├── eval_policy.py          # Policy evaluation
│   └── run_sim_policy_on_real_robot.py  # Sim-to-real deployment
├── data/
│   ├── usd/                    # 13 USD assets (board + 12 piece types)
│   └── chess_fen_*.jsonl       # FEN datasets (train/val/test)
└── README.md
```

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

Log in to your Brev instance running the Isaac Sim container:

```bash
brev shell isaacsim
```

Clone the repo and install dependencies:

```bash
git clone https://github.com/MaxSchmitz/cosmos-chessbot.git
cd cosmos-chessbot
pip install uv && uv sync
```

Log in to Hugging Face (Cosmos-Reason2 is a gated model):

```bash
uvx huggingface-cli login
```

Open port 8000 so your local machine can reach the server. In the Brev dashboard, add port 8000 to the instance's open ports, or use SSH tunneling:

```bash
# Option A: SSH tunnel from your local machine
ssh -L 8000:localhost:8000 <your-brev-instance>

# Option B: Open port in Brev dashboard, then use the public URL
```

Start the Cosmos-Reason2 server:

```bash
uv run python scripts/cosmos_server.py --host 0.0.0.0 --port 8000
```

Verify it's running:

```bash
curl http://localhost:8000/health
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

Single move (robot executes one move, then stops):

```bash
uv run cosmos-chessbot \
  --cosmos-server http://<your-brev-server>:8000 \
  --overhead-camera 0 --wrist-camera 1
```

Full game as white:

```bash
uv run cosmos-chessbot \
  --cosmos-server http://<your-brev-server>:8000 \
  --game-mode full-game --color white
```

Full game as black:

```bash
uv run cosmos-chessbot \
  --cosmos-server http://<your-brev-server>:8000 \
  --game-mode full-game --color black --moves 20
```

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

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--game-mode` | `single-move` | `single-move` or `full-game` |
| `--color` | `white` | Robot plays as `white` or `black` |
| `--cosmos-server` | none | Brev server URL (e.g., `http://gpu:8000`) |
| `--model` | `nvidia/Cosmos-Reason2-2B` | Cosmos model for local inference |
| `--overhead-camera` | `0` | Egocentric camera device ID |
| `--wrist-camera` | `1` | Wrist camera device ID |
| `--stockfish` | `stockfish` | Path to Stockfish binary |
| `--policy` | `cosmos` | Policy: `cosmos` or `pi05` |
| `--moves` | unlimited | Maximum moves to execute |
| `--verbose` / `--quiet` | normal | Logging verbosity |

### RL Training (Isaac Sim)

```bash
# Inside Isaac Sim container
OMNI_KIT_ACCEPT_EULA=yes /isaac-sim/python.sh scripts/train_chess_rl.py \
    --num-envs 64 --num-steps 500000
```

### Sim-to-Real Deployment

```bash
uv run python scripts/run_sim_policy_on_real_robot.py \
    --checkpoint data/eval/policy_final.pt \
    --robot-port /dev/ttyUSB0 \
    --task "e2 e4" --continuous
```

## Current Status

- [x] Stockfish UCI integration
- [x] YOLO-DINO board perception (local, real-time)
- [x] Cosmos-Reason2 embodied reasoning (6 modes: turn/move/action/trajectory/verification/correction)
- [x] Remote reasoning server with full endpoint coverage
- [x] PPO RL training in Isaac Sim (reward curriculum, typed piece pool)
- [x] Sim-to-real policy deployment
- [x] Full game loop with turn detection and opponent move detection
- [x] FEN verification and automated recovery
- [ ] Full closed-loop autonomous play on physical robot

## License

- **Cosmos-Reason2**: NVIDIA Open Model License (Apache 2.0 compatible)
- **Isaac Sim / IsaacLab**: NVIDIA EULA (used for RL training)
- **Stockfish**: GPLv3 (used as an external engine)

## Acknowledgements

- [NVIDIA Cosmos](https://github.com/NVIDIA/Cosmos) team for Physical AI models and tooling
- [NVIDIA Isaac Sim / IsaacLab](https://github.com/isaac-sim/IsaacLab) for the simulation and RL training framework
- [Stockfish](https://stockfishchess.org/) developers for the UCI chess engine
