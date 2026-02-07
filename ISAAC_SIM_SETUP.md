# Isaac Sim RL Environment Setup

This guide covers using the exported USD assets and FEN placement logic to build an Isaac Sim RL environment for chess manipulation.

## What We Have

### 1. USD Assets (exported from Blender)

Location: `data/usd/`

```
data/usd/
├── board.usd           # Chess board (17KB)
├── pawn_w.usd          # White pawn (122KB)
├── pawn_b.usd          # Black pawn (122KB)
├── knight_w.usd        # White knight (214KB)
├── knight_b.usd        # Black knight (214KB)
├── bishop_w.usd        # White bishop (189KB)
├── bishop_b.usd        # Black bishop (189KB)
├── rook_w.usd          # White rook (2.2MB)
├── rook_b.usd          # Black rook (2.2MB)
├── queen_w.usd         # White queen (183KB)
├── queen_b.usd         # Black queen (183KB)
├── king_w.usd          # White king (199KB)
└── king_b.usd          # Black king (199KB)
```

**Origin:** Exported from VALUE-Dataset `board.blend` using `scripts/export_board_to_usd.py`

**Materials:** PBR materials (base color, roughness, metallic) preserved from Blender. Isaac OmniPBR will interpret them -- may not match Blender pixel-perfect, but sufficient for RL.

**Textures:** `data/usd/textures/lighting.exr` (21MB HDRI) can be ignored -- Isaac uses its own RTX lighting.

### 2. FEN-to-3D Placement Module

Location: `src/cosmos_chessbot/isaac/fen_placement.py`

**Renderer-agnostic Python module** with no Blender or Isaac dependencies. Can be imported by both Blender scripts and Isaac environments.

**Key functions:**

```python
from cosmos_chessbot.isaac.fen_placement import (
    fen_to_board_state,
    get_square_position,
    get_captured_pieces,
    FEN_TO_PIECE_TYPE,
    PIECE_COLLISION_RADII,
)

# Parse FEN to board state
state = fen_to_board_state("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
# state == {"A1": "R", "A2": "P", ..., "E8": "k", ...}

# Get 3D position for a square
pos = get_square_position("E4")  # np.array([0.053, -0.053, 0.0])

# Map piece character to USD asset name
asset = FEN_TO_PIECE_TYPE['P']  # 'pawn_w'
asset = FEN_TO_PIECE_TYPE['n']  # 'knight_b'

# Get piece collision radius (for PhysX and reward shaping)
radius = PIECE_COLLISION_RADII['Q']  # 0.035 meters (queen)
```

**Constants:**
- `BOARD_SQUARE_SIZE = 0.106768` meters (standard tournament board)
- Board center at origin (0, 0, 0)
- X axis: A→H (left to right from white's view)
- Y axis: 1→8 (bottom to top from white's view)
- Z axis: vertical (up)

## Isaac Sim Environment Architecture

### Environment Structure

```python
# Pseudocode for IsaacLab or EnvHub environment

class ChessPickPlaceEnv(DirectRLEnv):
    def __init__(self):
        # Load USD assets once
        self.board_prim = stage.DefinePrim("/World/Board", "Xform")
        self.board_prim.GetReferences().AddReference("data/usd/board.usd")

        # Load piece assets (12 types)
        self.piece_assets = {}
        for fen_char, asset_name in FEN_TO_PIECE_TYPE.items():
            asset_path = f"data/usd/{asset_name}.usd"
            self.piece_assets[fen_char] = asset_path

        # SO-101 robot arm (from LeIsaac)
        self.robot = SO101Robot(prim_path="/World/Robot")

    def reset(self):
        # Sample random FEN from dataset
        fen = self.sample_fen()

        # Parse FEN to board state
        board_state = fen_to_board_state(fen)

        # Place pieces on board
        for square, fen_piece in board_state.items():
            pos = get_square_position(square)
            asset_path = self.piece_assets[fen_piece]

            # Instantiate piece at position
            piece_prim = stage.DefinePrim(f"/World/Pieces/{square}", "Xform")
            piece_prim.GetReferences().AddReference(asset_path)
            piece_prim.GetAttribute("xformOp:translate").Set(pos)

            # Add PhysX collision
            radius = PIECE_COLLISION_RADII[fen_piece]
            UsdPhysics.CollisionAPI.Apply(piece_prim)
            # ... set collision shape, mass, etc.

        # Sample target move from FEN
        self.target_piece_square = sample_piece_to_move(board_state)
        self.target_destination_square = sample_valid_destination()

        return observation

    def step(self, action):
        # Apply action to robot (EE-space or joint-space)
        self.robot.apply_action(action)

        # Step physics sim
        self.physics_sim.step()

        # Calculate reward
        reward = self.compute_reward()

        # Check termination
        done = self.check_done()

        return observation, reward, done, info

    def compute_reward(self):
        # Shaped reward for RL
        gripper_pos = self.robot.get_gripper_position()
        target_piece_pos = self.get_piece_position(self.target_piece_square)
        target_square_pos = get_square_position(self.target_destination_square)

        # Approach phase: reward proximity to target piece
        approach_reward = -np.linalg.norm(gripper_pos - target_piece_pos)

        # Placement phase: reward proximity to target square (only when grasped)
        if self.is_piece_grasped(self.target_piece_square):
            placement_reward = -10 * np.linalg.norm(target_piece_pos - target_square_pos)
        else:
            placement_reward = 0

        # Collision penalty: penalize touching non-target pieces
        collision_penalty = -5 * self.count_collisions_with_other_pieces()

        # Success bonus: sparse reward on successful placement
        if self.is_placement_successful():
            success_bonus = 100
        else:
            success_bonus = 0

        return approach_reward + placement_reward + collision_penalty + success_bonus
```

### FEN Dataset Integration

Use the existing FEN datasets from the vision pipeline:

```python
import json

# Load FEN positions from VALUE hybrid dataset
with open("data/value_llava/chess_fen_train.json") as f:
    fen_data = json.load(f)

# Extract FENs
fens = [item['conversations'][1]['value'] for item in fen_data]

# Sample during reset
def sample_fen():
    return np.random.choice(fens)
```

**Diversity:** 10,000+ unique positions from Lichess games. Ensures RL agent sees wide variety of board configurations.

## Training Workflow

### Option 1: HIL-SERL (Single-Env SAC)

LeRobot's HIL-SERL pipeline adapted for Isaac Sim:

```bash
# 1. Collect demos (optional warmstart)
uv run python scripts/collect_episodes.py \
    --robot-type so101 \
    --repo-id cosmos-chessbot/chess-manipulation

# 2. Train SAC in Isaac Sim
# Replace HIL-SERL actor with Isaac env
# Keep reward classifier, EE-space actions, safety bounds

# 3. Evaluate in sim
# Track: success rate, placement accuracy, collision rate

# 4. Deploy to real robot
# Sim-to-real transfer with domain randomization
```

**Pros:** Easier to wire up, single-env debugging, reuses LeRobot HIL-SERL components.

**Cons:** Slower than GPU-parallel rollouts.

### Option 2: IsaacLab GPU-Parallel RL

Thousands of parallel envs on H100:

```bash
# Train with IsaacLab RL workflows (PPO, SAC, etc.)
# 4096 envs @ 60 Hz = ~250k steps/sec

# Requires:
# - Linux + NVIDIA GPU + Isaac Sim
# - IsaacLab DirectRLEnv or ManagerBasedRLEnv
# - Vectorized environment with shared scene graph
```

**Pros:** 100x+ faster training.

**Cons:** More complex setup, requires Isaac Sim Linux install.

**Recommendation:** Start with Option 1 (single-env HIL-SERL) to validate the environment design, then scale to Option 2 if training time is a bottleneck.

## Domain Randomization (Sim-to-Real)

Randomize at each `reset()` for transfer to real robot:

### Visual Randomization
- Piece colors (reuse `PIECE_COLOR_SCHEMES` from Blender script)
- Board materials (wood, marble, plastic)
- Lighting (RTX dome light intensity, rotation)
- Camera viewpoint (match real SO-101 head camera FOV)

### Physics Randomization
- Piece mass (±20%)
- Friction coefficients (table, pieces)
- Gripper force limits
- Action latency (communication delay)

### Geometry Randomization
- Piece placement noise (±2mm on square)
- Board orientation (±5° rotation)
- Table height variation

**Implementation:**
```python
# In reset()
piece_mass = np.random.uniform(0.02, 0.05)  # kg
table_friction = np.random.uniform(0.3, 0.7)
lighting_intensity = np.random.uniform(800, 1200)
```

## Next Steps

1. **Set up Isaac Sim environment** (Linux + NVIDIA GPU required)
   ```bash
   # Install Isaac Sim 2023.1.1+ (requires Ubuntu 22.04)
   # Follow: https://docs.omniverse.nvidia.com/isaacsim/latest/installation.html
   ```

2. **Create minimal env** that loads the USD assets and places pieces from FEN
   - Test with `scripts/isaac_env_test.py` (to be written)
   - Validate piece positions match Blender coordinates

3. **Add SO-101 robot** from LeIsaac or create custom URDF→USD conversion
   - Test gripper collision with pieces
   - Validate forward kinematics matches real robot

4. **Implement reward function** with approach + placement + collision terms
   - Test reward shaping on hand-designed trajectories
   - Verify gradients guide policy toward solution

5. **Train SAC agent** (single-env first)
   - Target: 80%+ success rate in sim before real deployment
   - Log: success rate, mean reward, collision rate, episode length

6. **Domain randomization** tuning
   - Compare sim policy on real robot
   - Adjust randomization ranges to minimize sim-to-real gap

## References

- [LeIsaac SO-101 Tasks](https://github.com/LightwheelAI/LeIsaac/tree/main/source/extensions/leisaac.tasks)
- [IsaacLab DirectRLEnv Tutorial](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html)
- [HIL-SERL Paper](https://arxiv.org/abs/2410.20027)
- [USD Import in Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/features/environment_setup/assets/usd_assets_overview.html)
