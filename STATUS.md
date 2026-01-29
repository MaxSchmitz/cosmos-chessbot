# Project Status

**Last Updated**: 2026-01-29
**Timeline**: 1 month (just started)
**Target**: Cosmos Cookoff submission

## What's Working ✅

### Infrastructure
- [x] Project structure created
- [x] Dependencies installed and tested
- [x] Module imports working
- [x] Git repository initialized

### Stockfish Integration
- [x] UCI protocol wrapper implemented
- [x] Context manager interface for safe cleanup
- [x] Support for FEN positions and move sequences
- [x] Tested and working (see `scripts/test_stockfish.py`)

### Camera Module
- [x] Abstract camera interface implemented
- [x] Support for multiple cameras (overhead + wrist)
- [x] Frame capture and save functionality
- [x] PIL Image output format

### Cosmos-Reason2 Perception
- [x] Model loading infrastructure
- [x] Structured prompt for board state extraction
- [x] JSON parsing with fallback
- [x] BoardState dataclass (FEN, confidence, anomalies)
- [x] **Remote inference server** (FastAPI) for GPU deployment
- [x] **Remote client** for local development
- [x] Automatic local/remote switching via config

### Orchestrator
- [x] Main control loop structure
- [x] Component integration (cameras, Cosmos, Stockfish)
- [x] sense → perceive → plan pipeline
- [x] Stubs for act → verify → recover

### Testing Tools
- [x] `scripts/stockfish_smoke.py` - Basic UCI test
- [x] `scripts/test_stockfish.py` - Wrapper test
- [x] `scripts/capture_frame.py` - Camera capture
- [x] `scripts/test_perception.py` - Cosmos testing (local)
- [x] `scripts/test_remote_perception.py` - Remote Cosmos testing
- [x] `scripts/cosmos_server.py` - GPU inference server
- [x] `scripts/inference_sample.py` - Reference example

## What's Not Yet Implemented ⏳

### High Priority (Week 1)
- [ ] Test camera capture on actual SO-101 setup
- [ ] Test Cosmos perception on real board images
- [ ] Prompt engineering for FEN extraction
- [ ] Teleoperation recording infrastructure
- [ ] Episode data format definition

### Medium Priority (Week 2)
- [ ] π₀.₅ policy interface
- [ ] Intent compilation (Cosmos: move → constraints)
- [ ] Verification logic improvements
- [ ] FEN comparison with fuzzy matching

### Lower Priority (Week 3+)
- [ ] π₀.₅ fine-tuning pipeline
- [ ] Recovery system (Cosmos failure diagnosis)
- [ ] End-to-end testing
- [ ] Metrics and evaluation

## File Structure

```
cosmos-chessbot/
├── src/cosmos_chessbot/
│   ├── __init__.py          ✅ Package init
│   ├── main.py              ✅ CLI entry point
│   ├── orchestrator/
│   │   ├── __init__.py      ✅
│   │   └── orchestrator.py  ✅ Main control loop
│   ├── vision/
│   │   ├── __init__.py      ✅
│   │   ├── camera.py        ✅ Camera interface
│   │   └── perception.py    ✅ Cosmos perception
│   ├── stockfish/
│   │   ├── __init__.py      ✅
│   │   └── engine.py        ✅ UCI wrapper
│   ├── policy/
│   │   └── __init__.py      ⏳ π₀.₅ interface (TODO)
│   ├── schemas/
│   │   └── __init__.py      ⏳ Data schemas (TODO)
│   └── utils/
│       └── __init__.py      ✅
├── scripts/
│   ├── stockfish_smoke.py   ✅ Working
│   ├── test_stockfish.py    ✅ Working
│   ├── capture_frame.py     ✅ Ready to test
│   ├── test_perception.py   ✅ Ready to test
│   └── inference_sample.py  ✅ Reference
├── data/
│   ├── raw/                 ✅ For captured images
│   ├── episodes/            ✅ For teleoperation data
│   └── eval/                ✅ For evaluation results
├── assets/                  ✅ For test images
├── pyproject.toml           ✅ Dependencies configured
├── README.md                ✅ Project overview
├── QUICKSTART.md            ✅ Getting started guide
├── DEVELOPMENT.md           ✅ Development guide
└── STATUS.md                ✅ This file
```

## Immediate Next Steps

1. **Test Hardware Integration**
   ```bash
   # Capture from overhead camera (adjust device ID if needed)
   uv run python scripts/capture_frame.py 0 assets/board_test.jpg

   # Capture from wrist camera
   uv run python scripts/capture_frame.py 1 assets/wrist_test.jpg
   ```

2. **Test Cosmos Perception**
   ```bash
   # Run perception on captured board image
   uv run python scripts/test_perception.py assets/board_test.jpg
   ```

3. **Iterate on Prompts**
   - Edit prompt in `src/cosmos_chessbot/vision/perception.py`
   - Test on various board configurations
   - Measure accuracy against ground truth

4. **Start Data Collection**
   - Design episode recording format
   - Set up leader-follower teleoperation
   - Begin collecting pick-and-place demos

## Key Decisions Made

**Architecture**: Hierarchical separation - Cosmos for reasoning, π₀.₅ for control, Stockfish for strategy

**Cosmos Usage**: Local inference (not API) for latency and cost

**Modular Design**: Each component can be tested independently

**Data Format**: FEN for board state, UCI for moves

## Open Questions

- [ ] Camera device IDs for overhead and wrist on SO-101?
- [ ] π₀.₅ training: from scratch or fine-tune pre-trained?
- [ ] Teleoperation framework: custom or existing?
- [ ] Success criteria: accuracy thresholds?
- [ ] Demo format: full game or multi-move sequence?

## Resources

- Cosmos-Reason2: `nvidia/Cosmos-Reason2-2B` on HuggingFace
- π₀.₅: Physical Intelligence / LeRobot
- Stockfish: GPL v3 engine via UCI
- Hardware: SO-101 arm with leader + follower
