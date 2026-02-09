# Running Isaac Sim on Brev

Quick setup guide for testing USD assets on your Brev Isaac Sim machine.

## 1. Clone Repo on Brev Machine

```bash
# SSH into Brev machine
brev shell <machine-name>

# Clone repo
cd ~
git clone https://github.com/<your-username>/cosmos-chessbot.git
cd cosmos-chessbot
```

## 2. Copy USD Assets to Brev Machine

The USD assets were exported on your local Mac. Transfer them to Brev:

```bash
# On your LOCAL machine (Mac)
cd ~/Code/cosmos-chessbot
tar -czf usd_assets.tar.gz data/usd/

# Copy to Brev (replace with your machine name/IP)
scp usd_assets.tar.gz <brev-machine>:~/cosmos-chessbot/

# On BREV machine
cd ~/cosmos-chessbot
tar -xzf usd_assets.tar.gz
ls data/usd/  # Should see board.usd, pawn_w.usd, etc.
```

Alternatively, if the repo has the USD files committed:
```bash
# On Brev machine
cd ~/cosmos-chessbot
git pull  # Get latest with USD assets
```

## 3. Install Dependencies

Isaac Sim has its own Python environment. Install our package there:

```bash
# On Brev machine
cd ~/cosmos-chessbot

# Find Isaac Sim Python (usually in ~/.local/share/ov/pkg/isaac-sim-*)
ISAAC_PYTHON=~/.local/share/ov/pkg/isaac-sim-*/python.sh

# Install package in Isaac's Python
$ISAAC_PYTHON -m pip install -e .
```

## 4. Run Test Script

Test that USD assets load correctly:

```bash
cd ~/cosmos-chessbot

# Headless mode (no GUI, faster)
~/.local/share/ov/pkg/isaac-sim-*/python.sh scripts/isaac_env_test.py

# With GUI (if you have remote desktop/VNC)
~/.local/share/ov/pkg/isaac-sim-*/python.sh scripts/isaac_env_test.py --gui
```

**Expected output:**
```
============================================================
Isaac Sim USD Asset Test
============================================================
USD directory: /home/ubuntu/cosmos-chessbot/data/usd

[1/4] Creating ground plane...
[2/4] Adding physics scene...
[3/4] Loading chess board...
  ✓ Board loaded from board.usd
[4/4] Placing pieces from FEN...
  FEN: 4k3/8/8/8/4P3/8/8/4K3 w - - 0 1
  Pieces to place: 3
  ✓ E1: king_w     at (+0.053, -0.374, +0.000)
  ✓ E4: pawn_w     at (+0.053, -0.053, +0.000)
  ✓ E8: king_b     at (+0.053, +0.374, +0.000)

============================================================
Test scene created successfully!
  Board: 1 object
  Pieces: 3 objects
  Board square size: 0.106768 meters
============================================================

Scene saved to: /home/ubuntu/cosmos-chessbot/outputs/isaac_test_scene.usd
```

## 5. Inspect Output Scene

If the test succeeds, you can open the exported scene in Isaac Sim:

```bash
# Launch Isaac Sim GUI
~/.local/share/ov/pkg/isaac-sim-*/isaac-sim.sh

# Then: File → Open → outputs/isaac_test_scene.usd
```

You should see:
- Chess board at origin
- White king at E1
- White pawn at E4
- Black king at E8

## 6. Next Steps

Once the test passes, you're ready to:

1. **Build the RL environment** (see `ISAAC_SIM_SETUP.md`)
   - Create `ChessPickPlaceEnv` class
   - Implement `reset()`, `step()`, `compute_reward()`

2. **Add SO-101 robot**
   - Use LeIsaac's SO-101 USD or convert URDF
   - Test gripper collision with pieces

3. **Test reward function**
   - Hand-design a few trajectories
   - Verify reward gradients point toward solution

4. **Train RL agent**
   - Start with HIL-SERL (single-env SAC)
   - Target 80%+ success rate before real deployment

## Troubleshooting

### "Module 'isaacsim' not found"
Isaac Sim Python must be invoked via `python.sh`, not system Python:
```bash
# Wrong
python scripts/isaac_env_test.py

# Right
~/.local/share/ov/pkg/isaac-sim-*/python.sh scripts/isaac_env_test.py
```

### "USD file not found"
Check that `data/usd/` was transferred correctly:
```bash
ls -lh data/usd/
# Should see 13 USD files totaling ~6MB
```

### "Cannot connect to display"
If running headless without GUI, make sure you're NOT using `--gui` flag:
```bash
# Headless (no X11 needed)
python.sh scripts/isaac_env_test.py

# GUI requires X11 forwarding or VNC
python.sh scripts/isaac_env_test.py --gui
```

### Isaac Sim Python path varies
Brev might install Isaac Sim in a different location. Find it with:
```bash
find ~ -name "python.sh" -path "*/isaac-sim*/python.sh" 2>/dev/null
```

## Resources

- [Isaac Sim Python API](https://docs.omniverse.nvidia.com/isaacsim/latest/core_api_tutorials/index.html)
- [USD Assets Guide](https://docs.omniverse.nvidia.com/isaacsim/latest/features/environment_setup/assets/usd_assets_overview.html)
- [Brev Docs](https://docs.brev.dev/)
