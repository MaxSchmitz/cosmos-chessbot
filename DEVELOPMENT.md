# Development Guide

## What We Built

### Core Architecture

**Modular Control Loop**: sense → perceive → plan → compile → act → verify → recover

**Key Components Implemented**:

1. **Vision Module** (`src/cosmos_chessbot/vision/`)
   - `Camera`: Interface for overhead and wrist camera capture
   - `CosmosPerception`: Cosmos-Reason2 integration for board state extraction
   - Outputs structured `BoardState` with FEN, confidence, and anomalies

2. **Stockfish Module** (`src/cosmos_chessbot/stockfish/`)
   - UCI protocol wrapper
   - Get best move from any board position
   - Already tested and working

3. **Orchestrator** (`src/cosmos_chessbot/orchestrator/`)
   - Main control loop
   - Coordinates all components
   - Implements the full pipeline

4. **Utility Scripts** (`scripts/`)
   - `stockfish_smoke.py`: Test Stockfish (✅ working)
   - `capture_frame.py`: Test camera capture
   - `test_perception.py`: Test Cosmos perception
   - `inference_sample.py`: Reference Cosmos example

## Development Priorities

### Week 1 Tasks

**Priority 1: Camera + Cosmos Perception (Days 1-3)**
- [ ] Test camera capture on your SO-101 setup
- [ ] Capture test images of chess board
- [ ] Run Cosmos perception and iterate on prompts
- [ ] Measure FEN extraction accuracy

**Priority 2: Data Collection Pipeline (Days 1-7)**
- [ ] Set up teleoperation recording
- [ ] Define episode structure (start state, actions, end state)
- [ ] Start collecting pick-and-place demonstrations
- [ ] Target: 100-200 episodes by end of week

**Priority 3: Orchestrator Testing (Days 4-7)**
- [ ] Test with mock policy (prints instead of executing)
- [ ] Verify sense → perceive → plan flow
- [ ] Debug any camera/Cosmos issues

### Testing the Perception

```bash
# 1. Capture a test frame from overhead camera
uv run python scripts/capture_frame.py 0 assets/test_board.jpg

# 2. Run perception on it
uv run python scripts/test_perception.py assets/test_board.jpg
```

Expected output:
- FEN string of current board state
- Confidence score
- List of any anomalies detected

### What's Not Implemented Yet

**π₀.₅ Policy** (`src/cosmos_chessbot/policy/`)
- Need to integrate LeRobot/Physical Intelligence framework
- Fine-tune on collected teleoperation data
- Interface: `execute(pick_square, place_square, constraints) -> success`

**Intent Compilation** (in orchestrator)
- Currently just parses UCI move to pick/place squares
- Should use Cosmos-Reason2 to generate:
  - Approach strategy (from above, from side)
  - Clearance constraints
  - Collision avoidance
  - Recovery strategies

**Verification Logic** (in orchestrator)
- Currently just FEN string comparison
- Should be more sophisticated:
  - Fuzzy matching (pieces slightly off-center is OK)
  - Handle ambiguous states
  - Confidence thresholds

**Recovery System** (in orchestrator)
- Currently just a stub
- Should use Cosmos-Reason2 to:
  - Diagnose failure mode
  - Generate recovery plan
  - Execute recovery behaviors

## Prompt Engineering for Cosmos

The current perception prompt (in `vision/perception.py`) requests JSON output with:
- `fen`: Board state
- `confidence`: 0-1 score
- `anomalies`: List of physical issues

You may need to iterate on this prompt based on actual performance. Consider:
- Adding few-shot examples
- Being more explicit about FEN format
- Adding reasoning steps ("First, identify each piece...")

## Architecture Decisions

**Why separate Cosmos calls?**
- Perception: FEN extraction
- Intent compilation: Move → physical constraints
- Verification: Post-action diagnosis
- Recovery: Failure analysis

Each uses Cosmos differently. Keeping them separate allows independent prompt tuning.

**Why local Cosmos inference?**
- Lower latency (critical for real-time control)
- No API costs
- Full control over generation parameters
- Can batch if needed

**Why Stockfish over neural chess engine?**
- Deterministic and understood
- Fast and reliable
- Doesn't distract from the Physical AI focus
- Can be upgraded later if desired

## Next Steps

1. **Test cameras** - Ensure both overhead and wrist work
2. **Test Cosmos** - Run on real board images, iterate prompts
3. **Start data collection** - Begin teleoperation ASAP
4. **Plan π₀.₅ integration** - Design the policy interface

## Monitoring Progress

Track these metrics:
- FEN extraction accuracy (compare to ground truth)
- Anomaly detection rate (how often catches issues)
- End-to-end move success rate (after policy integration)
- Recovery success rate (after recovery system)

## Questions?

Key unknowns to resolve:
- Are your cameras USB or CSI? (may need different cv2 backend)
- What's your SO-101 control interface?
- Do you have existing teleoperation code to build on?
- Will you fine-tune π₀.₅ from scratch or use pre-trained base?
