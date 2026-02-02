# Hybrid Chess Dataset Generator

## Overview

This approach combines the best of both ChessR and VALUE datasets:

**From ChessR** (Visual Diversity):
- Multiple board styles and materials
- Multiple piece sets
- Random camera angles (0-20° from top, 360° rotation)
- Random HDRI lighting environments
- Motion blur simulation
- Piece rotation and jitter within squares

**From VALUE** (Realistic Positions):
- Real Lichess game FEN positions (not random placement)
- 200K+ realistic chess positions from actual games

## Why This Approach?

The VALUE dataset has realistic positions but limited visual variation (single board style, single camera angle). ChessR has great visual diversity but uses random piece placement (unrealistic positions).

By combining them, we get:
- **Realistic positions** that actually occur in games
- **Visual robustness** to different boards, pieces, lighting, angles
- **Better generalization** to real robot camera views

## Prerequisites

### 1. Install ChessR Dependencies

```bash
# Install Blender 2.8+ (if not already installed)
# Download from https://www.blender.org/download/

# Install chess library for FEN parsing
pip install chess
```

### 2. Verify Directory Structure

```
/Users/max/Code/
├── ChessR/
│   ├── ChessR_datagen.blend          # Main Blender file
│   └── src/
│       ├── BoardImagesGenerator.py
│       ├── Board.py
│       └── globals.py
├── VALUE-Dataset/
│   └── rendering/
│       ├── data/
│       │   └── Dec18.pgn             # Lichess games
│       └── utils/
│           └── ChessReader.py
└── cosmos-chessbot/
    └── scripts/
        └── generate_hybrid_dataset.py
```

### 3. Set Up ChessR Blender File

The `ChessR_datagen.blend` file must have these collections:
- **plateaux** - Contains board styles (multiple boards for variation)
- **Pieces sets** - Contains piece sets (multiple sets for variation)
- **piecesTypes** - Categorizes pieces by type (pawn, knight, etc.)

If you haven't customized ChessR yet, the default file should work.

## Usage

### Basic Usage

Generate 10,000 images with hybrid approach:

```bash
cd /Users/max/Code/ChessR

blender ChessR_datagen.blend -b -P /Users/max/Code/cosmos-chessbot/scripts/generate_hybrid_dataset.py -- --num-images 10000
```

### Advanced Options

```bash
blender ChessR_datagen.blend -b -P /Users/max/Code/cosmos-chessbot/scripts/generate_hybrid_dataset.py -- \
  --pgn-file /Users/max/Code/VALUE-Dataset/rendering/data/Dec18.pgn \
  --output-dir /Users/max/Code/cosmos-chessbot/data/hybrid_dataset \
  --num-images 50000
```

**Arguments:**
- `--pgn-file`: Path to Lichess PGN file (default: VALUE Dec18.pgn)
- `--output-dir`: Where to save images and annotations (default: cosmos-chessbot/data/hybrid_dataset)
- `--num-images`: Number of images to generate (default: 10000)

### Expected Output

```
cosmos-chessbot/data/hybrid_dataset/
├── chess_0000000.jpg
├── chess_0000001.jpg
├── ...
├── chess_0009999.jpg
└── annotations.json          # Llava format, ready for Cosmos-RL
```

**annotations.json format:**
```json
[
  {
    "id": "hybrid_0000000",
    "image": "/Users/max/Code/cosmos-chessbot/data/hybrid_dataset/chess_0000000.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat is the FEN position of this chess board?"
      },
      {
        "from": "gpt",
        "value": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
      }
    ]
  }
]
```

## Training with Hybrid Dataset

### Update Cosmos-RL Config

Edit `scripts/cosmos_rl/chess_sft_config.toml`:

```toml
[custom.dataset]
annotation_path = "/Users/max/Code/cosmos-chessbot/data/hybrid_dataset/annotations.json"
media_path = ""  # Empty because paths are absolute
system_prompt = "You are a helpful chess analysis assistant."
```

### Train Model

```bash
# On GPU server
cd ~/cosmos-reason2/examples/cosmos_rl
source .venv/bin/activate

cosmos-rl --config ~/cosmos-chessbot/scripts/cosmos_rl/chess_sft_config.toml \
           ~/cosmos-chessbot/scripts/cosmos_rl/chess_sft.py
```

## Recommended Dataset Sizes

Based on Cosmos Cookbook and data requirements:

| Dataset Size | Use Case | Training Time (8x A100) |
|-------------|----------|------------------------|
| 10,000 | Quick test, proof of concept | ~1 hour |
| 50,000 | Good baseline, diverse coverage | ~3 hours |
| 100,000 | Strong performance | ~6 hours |
| 200,000 | Maximum performance (matches VALUE) | ~12 hours |

**Recommendation**: Start with 10,000 to verify pipeline, then scale to 50-100K for production.

## Rendering Performance

**Single-threaded (Blender):**
- ~10-20 seconds per image (including variations)
- 10,000 images: ~28-56 hours
- 50,000 images: ~6-12 days

**Tips for Faster Rendering:**

### 1. Reduce Render Samples

Edit ChessR's render settings in Blender or modify the script:
```python
bpy.context.scene.cycles.samples = 64  # Lower from default 128
```

### 2. Use GPU Rendering

Enable GPU in Blender preferences or via script:
```python
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.scene.cycles.device = 'GPU'
```

### 3. Parallel Rendering (Advanced)

Split dataset into batches and render on multiple machines:

**Machine 1:**
```bash
blender ChessR_datagen.blend -b -P generate_hybrid_dataset.py -- --num-images 25000
```

**Machine 2:**
```bash
# Modify script to start from offset
blender ChessR_datagen.blend -b -P generate_hybrid_dataset.py -- --num-images 25000 --start-index 25000
```

Then combine annotations manually.

## Customization

### Add More Board Styles

1. Open `ChessR_datagen.blend` in Blender
2. Import new board 3D model
3. Add to `plateaux` collection
4. Ensure it has cell empties (A1-H8, corners, center)
5. Save blend file

### Add More Piece Sets

1. Open `ChessR_datagen.blend`
2. Import new piece 3D models
3. Create new collection in `Pieces sets`
4. Add all piece types (pawn, knight, bishop, rook, queen, king)
5. Link them to corresponding `piecesTypes` subcollections
6. Save blend file

### Adjust Camera Angles

Modify in `generate_hybrid_dataset.py`:

```python
# In positionCameraAroundBoardCenter() call
# Edit MAX_CAM_ANGLE_FROM_UPWARDS in ChessR/src/globals.py
# Default: 0-20° from top
# For more extreme angles: increase to 0-45°
```

### Adjust Lighting Variety

Add more HDRI files to ChessR's HDRI folder:
```bash
# Download HDRIs from polyhaven.com
# Add to: /Users/max/Code/ChessR/hdri/
# Script will automatically pick random ones
```

## Validation

### Check Sample Quality

```bash
# View first 10 images
open /Users/max/Code/cosmos-chessbot/data/hybrid_dataset/chess_000000*.jpg

# Verify FEN annotations
cat /Users/max/Code/cosmos-chessbot/data/hybrid_dataset/annotations.json | jq '.[0:5]'
```

### Verify FEN Correctness

Create quick validation script:

```python
import json
import chess

with open('data/hybrid_dataset/annotations.json') as f:
    data = json.load(f)

# Check all FENs are valid
for sample in data:
    fen = sample['conversations'][1]['value']
    try:
        board = chess.Board(fen)
        assert board.is_valid()
    except:
        print(f"Invalid FEN: {fen} in {sample['id']}")
```

## Troubleshooting

### Error: "chess library not available"

```bash
pip install chess
```

### Error: "ChessR modules not found"

Make sure you're running from `/Users/max/Code/ChessR` directory and the Blender file is `ChessR_datagen.blend`.

### Error: "No HDRIs found"

Set HDRI path in ChessR's `src/globals.py`:
```python
HDRI_FOLDER = "/path/to/hdri/files"
```

Or disable HDRI randomization in `generate_hybrid_dataset.py` by commenting out:
```python
# self.config_generator.setRandomHDRI()
```

### Rendering Too Slow

- Lower render samples (see "Rendering Performance" above)
- Enable GPU rendering
- Use fewer images initially (10K instead of 100K)
- Consider cloud GPU rendering (AWS, Vast.ai)

## Next Steps

1. **Generate dataset** (start with 10K images)
2. **Verify quality** (check samples manually)
3. **Train Cosmos-RL** (update config, run training)
4. **Evaluate** (test on robot camera images)
5. **Iterate** (adjust variations, add more samples if needed)

## Comparison to Alternatives

| Approach | Positions | Visual Diversity | Pros | Cons |
|----------|-----------|------------------|------|------|
| **VALUE only** | Real Lichess ✅ | Low ❌ | Ready to use, proven | Won't generalize to different boards/lighting |
| **ChessR only** | Random ❌ | High ✅ | Great visual variation | Unrealistic positions, poor FEN learning |
| **Hybrid (this)** | Real Lichess ✅ | High ✅ | Best of both worlds | Requires rendering time |

## Expected Results

Based on Cosmos Cookbook intelligent-transportation recipe (92% accuracy with 5K samples):

**With 10K hybrid samples:**
- Expected FEN accuracy: 85-90%
- Training time: 1-2 hours (8x A100)
- Good for proof of concept

**With 50K hybrid samples:**
- Expected FEN accuracy: 90-95%
- Training time: 3-4 hours (8x A100)
- Production-ready for robot deployment

**With 100K+ hybrid samples:**
- Expected FEN accuracy: 95%+
- Training time: 6-8 hours (8x A100)
- Maximum performance, robust to edge cases
