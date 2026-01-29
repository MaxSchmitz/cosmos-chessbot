Cosmos Chessbot

Bridging Moravec’s Paradox with Physical AI Reasoning

Overview

In 1996, a computer defeated the world chess champion — but needed a human to move the pieces.
In 2016, AlphaGo defeated the world champion in Go — but still could not act in the physical world.
Today, AI systems can solve Olympiad-level math problems, yet struggle with tasks humans find trivial, like picking up a chess piece.

This mismatch is known as Moravec’s Paradox:

tasks that are cognitively “hard” for humans are often easy for machines, while tasks that are physically “easy” for humans remain exceptionally difficult for machines.

Cosmos Chessbot demonstrates how modern Physical AI models can help close this gap.

We build a robotic chess system where:

symbolic intelligence (chess strategy) is solved,

physical manipulation is imperfect and uncertain,

and Cosmos-Reason2 acts as a reasoning supervisor that connects the two.

The result is a robot that does not just choose the right move — it can reason about the physical world, detect failures, and recover when reality disagrees with intention.

System Architecture

Cosmos Chessbot is a modular, hierarchical Physical AI system:

Cameras (overhead + wrist)
        ↓
Cosmos-Reason2 (perception + physical reasoning)
        ↓
Stockfish (symbolic planning / best move)
        ↓
Cosmos-Reason2 (intent compilation + constraints)
        ↓
π₀.₅ (vision-language-action policy)
        ↓
SO-101 Robot Arm (physical execution)
        ↓
Cosmos-Reason2 (verification + recovery)

Key Design Principle

Cosmos-Reason2 never outputs motor commands

π₀.₅ never reasons about chess or rules

Each model is used exactly where it is strongest

Core Components
1. Cosmos-Reason2 (Physical Reasoning Supervisor)

Cosmos-Reason2 receives camera input and performs:

Board state extraction (FEN + confidence)

Detection of physical anomalies (tilted pieces, pieces between squares, occlusion)

Translation of symbolic chess moves into physical manipulation intent

Post-action verification and failure diagnosis

Recovery planning when execution fails

Cosmos is used as a world model and deliberative planner, not as a controller.

2. Stockfish (Symbolic Chess Engine)

Stockfish is used via the UCI protocol to:

Select the best chess move from the current board state

Provide deterministic, well-understood symbolic planning

Stockfish has no perception and no embodiment — by design.

3. π₀.₅ (Vision-Language-Action Policy)

π₀.₅ executes physical actions on the SO-101 arm:

Picking up chess pieces

Placing pieces accurately on target squares

Performing recovery behaviors (regrasp, nudge, retry)

The policy is fine-tuned using teleoperation data collected on the real robot.

4. SO-101 Robotic Arm

Single-arm manipulation setup

Compliant gripper suitable for small objects

Overhead camera for global perception

Wrist camera for close-range grasp verification

Control Loop

Each move follows this loop:

Sense
Capture overhead (and wrist) camera images.

Perceive (Cosmos-Reason2)
Extract board state (FEN), confidence, and anomalies.

Plan (Stockfish)
Compute best move using UCI.

Compile Intent (Cosmos-Reason2)
Convert symbolic move into:

pick square

place square

constraints (approach, clearance, avoidance)

recovery strategy

Act (π₀.₅)
Execute the manipulation.

Verify (Cosmos-Reason2)
Confirm expected board state change.

Recover if Needed
If verification fails, Cosmos diagnoses the failure and updates the action plan.

This loop explicitly addresses the intelligence–embodiment gap highlighted by Moravec’s Paradox.

Why Chess?

Chess is not the goal — it is the testbed.

Chess provides:

A well-defined symbolic world (rules, moves, legality)

A physically challenging manipulation task

Clear success/failure criteria

Easy evaluation and reproducibility

By choosing chess, we isolate the hardest unsolved problem:

making symbolic decisions real in the physical world

What This Demonstrates

Physical AI reasoning over long horizons

Separation of reasoning and control

Robust perception under real-world imperfections

Failure detection and recovery

A concrete resolution of Moravec’s Paradox in a real robot system

Repository Structure
cosmos-chessbot/
├── src/
│   └── cosmos_chessbot/
│       ├── orchestrator/   # Main control loop
│       ├── vision/         # Camera capture + Cosmos prompts
│       ├── stockfish/      # UCI engine wrapper
│       ├── policy/         # π₀.₅ adapter
│       ├── schemas/        # JSON schemas for Cosmos outputs
│       └── utils/
├── scripts/                # CLI tools and demos
├── data/
│   ├── raw/                # Recorded camera data
│   ├── episodes/           # Teleop training data
│   └── eval/
├── configs/
├── tests/
└── README.md

Installation

This project uses uv for environment management.

uv init cosmos-chessbot
uv sync


Additional dependencies are installed incrementally as needed.

Current Status

✅ Stockfish UCI integration working

✅ Project scaffolding complete

⏳ Cosmos-Reason2 board perception

⏳ π₀.₅ fine-tuning with teleoperation data

⏳ Full closed-loop demo

Evaluation

We evaluate the system on:

Board state accuracy

Successful move execution rate

Recovery success after induced failures

End-to-end robustness over multiple moves

License

Cosmos-Reason2: NVIDIA Open Model License (Apache 2.0 compatible)

π₀.₅ (LeRobot): Apache 2.0

Stockfish: GPLv3 (used as an external engine)

Acknowledgements

NVIDIA Cosmos team for Physical AI models and tooling

Physical Intelligence for π₀.₅ and OpenPI

Stockfish developers for the UCI chess engine
