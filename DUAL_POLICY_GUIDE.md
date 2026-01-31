# Dual-Policy Chess Manipulation System

This guide explains how to use the dual-policy system that supports both **π₀.₅** and **Cosmos Policy** for chess manipulation tasks.

## Overview

The system allows empirical comparison of two manipulation policies:

1. **π₀.₅** - Vision-Language-Action policy from LeRobot
2. **Cosmos Policy** - World model-based policy with planning capability

Both policies train on the same LeRobot dataset, enabling fair comparison.

## Architecture

```
Data Collection (Once)
        ↓
    LeRobot Episodes
        ↓
    ┌───────┴────────┐
    ↓                ↓
Train π₀.₅     Train Cosmos
    ↓                ↓
Run & Compare Policies
```

## Quick Start

### 1. Collect Data (Once)

Collect teleoperation demonstrations using LeRobot:

```bash
# Set up LeRobot environment (separate from main env)
python3 -m venv lerobot_env
source lerobot_env/bin/activate
pip install lerobot gymnasium opencv-python pillow numpy

# Collect episodes
python scripts/collect_episodes.py --num-episodes 100
```

Episodes are saved to `data/episodes/cosmos-chessbot/chess-manipulation/`

### 2. Train π₀.₅

```bash
# Activate LeRobot environment
source lerobot_env/bin/activate

# Train π₀.₅ (50,000 steps, ~2-4 hours on H100)
python scripts/train_pi05.py

# Custom training
python scripts/train_pi05.py \
  --steps 100000 \
  --batch-size 8 \
  --output-dir checkpoints/pi05_custom
```

### 3. Train Cosmos Policy

```bash
# Install Cosmos Policy (when available)
# Follow NVIDIA Cosmos documentation

# Train Cosmos (30,000 steps)
python scripts/train_cosmos.py

# Custom training
python scripts/train_cosmos.py \
  --steps 50000 \
  --batch-size 4 \
  --horizon 15 \
  --output-dir checkpoints/cosmos_custom
```

### 4. Run with π₀.₅

```bash
# Using fine-tuned π₀.₅
uv run cosmos-chessbot \
  --policy pi05 \
  --policy-checkpoint checkpoints/pi05_chess/final.pt \
  --cosmos-server http://192.241.168.72:8080 \
  --moves 10
```

### 5. Run with Cosmos Policy

```bash
# Direct action (no planning)
uv run cosmos-chessbot \
  --policy cosmos \
  --policy-checkpoint checkpoints/cosmos_chess/final.pt \
  --no-enable-planning \
  --cosmos-server http://192.241.168.72:8080 \
  --moves 10

# With planning (recommended)
uv run cosmos-chessbot \
  --policy cosmos \
  --policy-checkpoint checkpoints/cosmos_chess/final.pt \
  --enable-planning \
  --cosmos-server http://192.241.168.72:8080 \
  --moves 10
```

### 6. Compare Policies

```bash
# Full comparison (50 test episodes per policy)
python scripts/compare_policies.py

# Custom comparison
python scripts/compare_policies.py \
  --num-episodes 100 \
  --pi05-checkpoint checkpoints/pi05_chess/final.pt \
  --cosmos-checkpoint checkpoints/cosmos_chess/final.pt \
  --output data/eval/my_comparison.json
```

## CLI Reference

### Main Application

```bash
cosmos-chessbot [OPTIONS]

Options:
  --policy {pi05,cosmos}          Policy to use (default: cosmos)
  --policy-checkpoint PATH        Path to checkpoint (optional)
  --enable-planning               Enable planning for Cosmos (default: on)
  --no-enable-planning            Disable planning
  --cosmos-server URL             Remote Cosmos perception server
  --moves INT                     Number of moves to execute
  --overhead-camera INT           Overhead camera ID
  --wrist-camera INT              Wrist camera ID
```

### Training Scripts

**π₀.₅ Training:**
```bash
python scripts/train_pi05.py [OPTIONS]

Options:
  --dataset-path PATH       Dataset location (default: data/episodes)
  --output-dir PATH         Checkpoint output (default: checkpoints/pi05_chess)
  --steps INT               Training steps (default: 50000)
  --batch-size INT          Batch size (default: 4)
  --lr FLOAT                Learning rate (default: 1e-4)
```

**Cosmos Training:**
```bash
python scripts/train_cosmos.py [OPTIONS]

Options:
  --dataset-path PATH       Dataset location (default: data/episodes)
  --output-dir PATH         Checkpoint output (default: checkpoints/cosmos_chess)
  --steps INT               Training steps (default: 30000)
  --batch-size INT          Batch size (default: 4)
  --horizon INT             Planning horizon (default: 10)
```

**Policy Comparison:**
```bash
python scripts/compare_policies.py [OPTIONS]

Options:
  --num-episodes INT          Test episodes per policy (default: 50)
  --pi05-checkpoint PATH      π₀.₅ checkpoint path
  --cosmos-checkpoint PATH    Cosmos checkpoint path
  --output PATH               Results JSON output
  --skip-pi05                 Skip π₀.₅ evaluation
  --skip-cosmos               Skip Cosmos evaluation
```

## Policy Comparison

### π₀.₅ (LeRobot)

**Strengths:**
- Mature implementation, proven on manipulation tasks
- Language conditioning (can use natural language instructions)
- Lower training compute requirements
- Good generalization with diverse demonstrations

**Limitations:**
- No explicit planning
- Single action prediction (greedy)
- Doesn't predict future states

### Cosmos Policy

**Strengths:**
- World model planning (predicts future states)
- Multi-candidate action generation
- Value-based action selection
- Better handling of occlusions and uncertainty

**Limitations:**
- Higher training compute requirements
- More complex to debug
- No language conditioning (vision-only)

### Expected Performance

Based on plan, we expect:
- **π₀.₅**: 70-80% success rate on diverse moves
- **Cosmos (direct)**: 75-85% success rate
- **Cosmos (planning)**: 80-90% success rate

Planning should provide advantage for:
- Occluded pieces
- Captures (removing defending piece)
- Recovery from failures

## File Structure

```
cosmos-chessbot/
├── src/cosmos_chessbot/
│   ├── policy/
│   │   ├── base_policy.py          # Abstract policy interface
│   │   ├── pi05_policy.py          # π₀.₅ implementation
│   │   └── cosmos_policy.py        # Cosmos implementation
│   └── orchestrator/
│       └── orchestrator.py         # Updated with policy selection
├── scripts/
│   ├── collect_episodes.py         # Data collection
│   ├── train_pi05.py               # π₀.₅ training
│   ├── train_cosmos.py             # Cosmos training
│   └── compare_policies.py         # Policy comparison
├── checkpoints/
│   ├── pi05_chess/                 # π₀.₅ checkpoints
│   └── cosmos_chess/               # Cosmos checkpoints
└── data/
    ├── episodes/                   # Training data
    └── eval/                       # Evaluation results
```

## Troubleshooting

### π₀.₅ Import Errors

```
ImportError: lerobot not found
```

**Solution:** Install LeRobot in separate environment:
```bash
python3 -m venv lerobot_env
source lerobot_env/bin/activate
pip install lerobot
```

### Cosmos Policy Not Found

```
ImportError: Cosmos Policy package not found
```

**Solution:** Install from NVIDIA:
```bash
git clone https://github.com/NVIDIA/Cosmos.git
cd Cosmos
pip install -e .
```

### Training OOM Errors

**Solution:** Reduce batch size:
```bash
# π₀.₅
python scripts/train_pi05.py --batch-size 2

# Cosmos
python scripts/train_cosmos.py --batch-size 2
```

### Policy Checkpoint Not Loading

Check checkpoint path:
```bash
ls -lh checkpoints/pi05_chess/
ls -lh checkpoints/cosmos_chess/
```

Ensure checkpoint file exists (e.g., `final.pt`, `checkpoint.pt`)

## Development Status

### ✅ Completed
- Abstract policy interface (`base_policy.py`)
- π₀.₅ wrapper with BasePolicy implementation
- Cosmos Policy wrapper with planning
- Orchestrator policy selection
- CLI arguments for policy choice
- Training scripts for both policies
- Comparison evaluation script

### 🚧 In Progress
- Data collection (need real SO-101 robot)
- Policy training (need collected data)
- Robot control integration (`_execute_robot_action`)

### 📋 TODO
- Collect 150-200 teleoperation episodes
- Train both policies
- Implement robot state reading (`_get_robot_state`)
- Implement robot action execution
- Run full comparative evaluation
- Record demo video

## Next Steps

1. **Data Collection** (Week 1)
   - Set up SO-101 with LeRobot
   - Collect 50 episodes
   - Test with real chess board

2. **π₀.₅ Training** (Week 2)
   - Train on collected data
   - Test in orchestrator
   - Collect more data if needed

3. **Cosmos Training** (Week 3)
   - Set up Cosmos Policy package
   - Train on same dataset
   - Test planning capability

4. **Comparison & Demo** (Week 4)
   - Run comparative evaluation
   - Choose best policy
   - Record submission video

## References

- [LeRobot Documentation](https://github.com/huggingface/lerobot)
- [NVIDIA Cosmos](https://github.com/NVIDIA/Cosmos)
- [Plan Document](See plan in conversation history)
