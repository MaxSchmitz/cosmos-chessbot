# Quick Start: Isaac Sim Setup

You now have everything needed to test USD assets in Isaac Sim.

## What's Ready

✅ **USD Assets** (13 files, 27MB total)
- Location: `data/usd/`
- Exported from VALUE board.blend
- Includes: board + 12 piece types with PBR materials

✅ **FEN Placement Module**
- Location: `src/cosmos_chessbot/isaac/fen_placement.py`
- Tested and working (all doctests pass)
- Converts FEN → 3D positions

✅ **Test Script**
- Location: `scripts/isaac_env_test.py`
- Loads USD assets and places pieces from FEN
- Works headless or with GUI

✅ **Documentation**
- `ISAAC_SIM_SETUP.md` - Full RL environment architecture
- `BREV_ISAAC_SETUP.md` - Brev machine setup steps

## On Your Brev Machine Now

Follow `BREV_ISAAC_SETUP.md` step-by-step:

1. **Clone repo** (or git pull if already cloned)
2. **Install package** in Isaac Sim's Python environment
3. **Run test script** to validate USD loading
4. **Check output** in `outputs/isaac_test_scene.usd`

If test passes → you're ready to build the RL environment.

## Commands Cheat Sheet

```bash
# On Brev machine
cd ~/cosmos-chessbot
git pull  # Get latest with USD assets

# Install in Isaac Python
~/.local/share/ov/pkg/isaac-sim-*/python.sh -m pip install -e .

# Run test
~/.local/share/ov/pkg/isaac-sim-*/python.sh scripts/isaac_env_test.py

# Expected: "Test scene created successfully!" with 3 pieces
```

## What Success Looks Like

```
============================================================
Test scene created successfully!
  Board: 1 object
  Pieces: 3 objects
  Board square size: 0.106768 meters
============================================================

Scene saved to: /home/ubuntu/cosmos-chessbot/outputs/isaac_test_scene.usd
```

If you see this → USD pipeline works → ready for RL env implementation.

## Next: Build RL Environment

After test passes, create `src/cosmos_chessbot/isaac/chess_env.py`:

```python
from omni.isaac.lab.envs import DirectRLEnv
from cosmos_chessbot.isaac.fen_placement import fen_to_board_state, get_square_position

class ChessPickPlaceEnv(DirectRLEnv):
    def __init__(self, cfg, **kwargs):
        # Load USD assets
        # Set up SO-101 robot
        # Initialize FEN dataset
        super().__init__(cfg, **kwargs)

    def reset(self):
        # Sample FEN
        # Place pieces
        # Sample target move
        return observation

    def step(self, action):
        # Apply action to robot
        # Step physics
        # Compute reward
        # Check termination
        return obs, reward, done, info
```

See `ISAAC_SIM_SETUP.md` for full environment pseudocode.
