# LeRobot Data Collection Guide

This guide covers using LeRobot for teleoperation data collection and π₀.₅ training.

## Overview

LeRobot (from Hugging Face) provides:
- ✅ Teleoperation recording with leader/follower arms
- ✅ Standard dataset format
- ✅ Built-in visualization tools
- ✅ Policy training pipelines
- ✅ Integration with SO-100/SO-101 robots

## Setup

### 1. Configure Robot

LeRobot needs to know about your SO-101 setup. Create a robot configuration:

```bash
# Check if SO-100/SO-101 is supported
lerobot list-robots

# If not listed, you may need to create a custom config
```

For SO-101 with leader/follower arms, you'll need:
- Robot type identifier
- Serial ports or device IDs
- Camera configurations (overhead + wrist)
- Workspace bounds

## Data Collection Workflow

### Step 1: Collect Episodes

```bash
# Collect 10 episodes
uv run python scripts/collect_episodes.py \
  --robot-path so100 \
  --num-episodes 10 \
  --fps 30 \
  --episode-time-s 30

# Options:
#   --display-cameras    Show camera feeds while recording
#   --warmup-time-s 5    Warmup before each episode
#   --root data/episodes Custom storage location
```

**What to demonstrate:**
- Pick up different chess pieces (pawn, knight, rook, etc.)
- Place on various target squares
- Different approach angles
- Captures (picking up and removing opponent pieces)
- Vary starting positions

**Tips:**
- Keep movements smooth and deliberate
- Complete each pick-and-place in one episode
- Reset board between episodes
- Aim for diverse demonstrations

### Step 2: Visualize Episodes

```bash
# View first episode
uv run python scripts/visualize_episodes.py --episode-index 0

# View episode 5
uv run python scripts/visualize_episodes.py --episode-index 5
```

This shows:
- Camera feeds
- Robot joint positions
- Gripper state
- Timestamps

### Step 3: Review Data Quality

Check for:
- ✅ Smooth trajectories (no jerky movements)
- ✅ Successful grasps (piece doesn't slip)
- ✅ Accurate placements (pieces centered on squares)
- ✅ Good camera visibility (no occlusions)

Delete bad episodes:
```bash
# LeRobot datasets are in data/episodes/cosmos-chessbot/chess-manipulation/
# Each episode is in a subfolder: episode_000000, episode_000001, etc.
rm -rf data/episodes/cosmos-chessbot/chess-manipulation/episode_000005
```

### Step 4: Collect More Data

Repeat until you have 100-200 high-quality episodes.

**Data collection targets:**
- Week 1: 50 episodes
- Week 2: 100 episodes (50 more)
- Week 3: 150-200 episodes (50-100 more)

## Dataset Format

LeRobot stores episodes in a standard format:

```
data/episodes/cosmos-chessbot/chess-manipulation/
├── meta_data/
│   └── info.json              # Dataset metadata
├── episode_000000/
│   ├── observation.state.txt  # Robot state
│   ├── observation.images.overhead.mp4
│   ├── observation.images.wrist.mp4
│   └── action.txt             # Recorded actions
├── episode_000001/
│   └── ...
└── stats/                     # Computed statistics
```

Each episode contains:
- **Observations**: Camera images + robot state (joint angles, gripper)
- **Actions**: Motor commands sent to follower arm
- **Metadata**: Timestamps, episode length, success flag

## Training π₀.₅

Once you have 100+ episodes:

```bash
# Train policy
lerobot train \
  --policy pi0 \
  --dataset-repo-id cosmos-chessbot/chess-manipulation \
  --root data/episodes \
  --num-epochs 100 \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --output-dir checkpoints/pi0_chess

# This will:
# 1. Load your collected episodes
# 2. Fine-tune π₀.₅ on chess manipulation
# 3. Save checkpoints periodically
# 4. Log metrics (loss, success rate)
```

Training time: ~6-12 hours on H100 for 100 episodes, 100 epochs.

## Evaluation

Test trained policy:

```bash
# Evaluate on held-out episodes
lerobot eval \
  --policy pi0 \
  --checkpoint checkpoints/pi0_chess/final.pt \
  --dataset-repo-id cosmos-chessbot/chess-manipulation \
  --root data/episodes \
  --num-episodes 10
```

## Integration with Orchestrator

Once trained, integrate with the main system:

```python
from cosmos_chessbot.policy import LeRobotPolicy

# Load trained policy
policy = LeRobotPolicy(
    robot_type="so100",
    policy_path=Path("checkpoints/pi0_chess/final.pt")
)

# Execute action
success = policy.execute_action(
    pick_square="e2",
    place_square="e4",
    constraints={"approach": "from_above"}
)
```

## Troubleshooting

### Robot not connecting
- Check serial ports: `ls /dev/tty*`
- Verify permissions: `sudo usermod -a -G dialout $USER`
- Check LeRobot config: `lerobot list-robots`

### Poor policy performance
- Collect more diverse data (100+ episodes)
- Check data quality (smooth movements, no failures)
- Increase training epochs
- Adjust learning rate

### Camera sync issues
- Ensure all cameras are same FPS
- Check timestamps in dataset
- Verify cameras are triggered simultaneously

## Resources

- [LeRobot Documentation](https://github.com/huggingface/lerobot)
- [SO-100 Robot Setup](https://github.com/huggingface/lerobot/tree/main/lerobot/configs/robot/so100.yaml)
- [Policy Training Guide](https://github.com/huggingface/lerobot/blob/main/examples/1_train_policy.md)

## Next Steps After Data Collection

1. **Integrate with orchestrator** - Connect trained policy to main control loop
2. **Test closed-loop** - Run full sense → perceive → plan → act → verify
3. **Add verification** - Use Cosmos to check if move succeeded
4. **Implement recovery** - Handle failures and retry logic
