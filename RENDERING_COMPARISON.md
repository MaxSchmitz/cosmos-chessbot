# Chess Dataset Rendering Comparison

## Test Results (February 2, 2026)

### 1. Hybrid Approach (Our Generator)

**Location**: `/Users/max/Code/cosmos-chessbot/data/hybrid_test/`

**Characteristics**:
- **Lighting**: Random HDRI selection from 3 studio HDRIs
- **Boards**: 5 different board styles (randomized)
- **Pieces**: 4 different piece sets (randomized)
- **Camera**: Random spherical position (0-20° from top, 360° rotation)
- **FEN Source**: Real Lichess game positions from VALUE dataset
- **Render Time**: ~1-1.5 minutes per image
- **File Size**: 55-99KB per image
- **Resolution**: Default ChessR settings (need to verify)

**Pros**:
- Real game positions (realistic piece distributions)
- High visual diversity (boards, pieces, lighting, angles)
- Egocentric camera angles possible
- Customizable to match robot camera

**Cons**:
- Slower rendering (~1.5 min/image = ~10 days for 10K images)
- Requires Blender setup
- Complex pipeline (multiple dependencies)

**Quality Notes**:
- First test WITHOUT HDRI: Purple/magenta tint (bad lighting)
- After adding HDRIs: Proper lighting and colors
- Shows importance of environment lighting for realistic rendering

---

### 2. VALUE Dataset (Pre-rendered)

**Location**: `/Users/max/Code/cosmos-chessbot/data/data/`

**Sample Images**:
- CV_0130556.jpg
- CV_0219250.jpg
- CV_0135936.jpg

**Characteristics**:
- **Lighting**: Fixed (single setup)
- **Boards**: Single board style
- **Pieces**: Single piece set
- **Camera**: Fixed overhead angle
- **FEN Source**: Real Lichess game positions
- **Render Time**: Already done (0 seconds)
- **File Size**: ~varies
- **Resolution**: 1280x1280
- **Count**: 200K train, 20K test

**Pros**:
- Ready immediately (no rendering time)
- Proven baseline (VALUE paper results)
- Large scale (220K images)
- Real game positions

**Cons**:
- No visual diversity (single board/piece style)
- Fixed camera angle (may not match robot view)
- Can't customize for specific setup

---

### 3. ChessR Original (Couldn't Test - Missing Dependencies)

**Would have provided**:
- Random piece placement (unrealistic positions)
- High visual diversity
- Reference for comparing our hybrid implementation

**Issues**:
- Requires `progress` module (needs user site-packages fix)
- Same Blender scene as our hybrid approach
- Would show if our implementation matches original ChessR rendering

---

## Key Findings

### HDRI Lighting is Critical

**Without HDRI** (empty hdri folder):
- Purple/magenta color cast
- Unrealistic appearance
- Environment Texture node has no texture loaded

**With HDRI** (3 studio HDRIs added):
- Natural lighting and colors
- Realistic shadows and reflections
- Professional appearance

**HDRIs Added**:
```
/Users/max/Code/ChessR/hdri/
├── studio_small_08_1k.hdr (1.4MB)
├── studio_small_09_1k.hdr (1.5MB)
└── photo_studio_loft_hall_1k.hdr (1.6MB)
```

Source: [Poly Haven](https://polyhaven.com) (CC0 license)

### Visual Diversity Comparison

| Feature | VALUE | ChessR Original | Hybrid (Ours) |
|---------|-------|-----------------|---------------|
| **Board Styles** | 1 | Multiple | 5 ✅ |
| **Piece Sets** | 1 | Multiple | 4 ✅ |
| **Camera Angles** | Fixed | Random 0-20° | Random 0-20° ✅ |
| **Lighting** | Fixed | Random HDRI | Random HDRI ✅ |
| **FEN Positions** | Real Lichess ✅ | Random | Real Lichess ✅ |
| **Render Speed** | N/A | ~1.5 min/img | ~1.5 min/img |

### Recommendation

**For immediate training**: Use VALUE dataset (ready now)

**For production deployment**: Generate hybrid dataset
- Start with 100 images to verify quality (~2.5 hours)
- Scale to 10K-50K for production (~1-4 weeks)
- Provides best generalization to real robot

---

## Rendering Performance

### Current Setup (Single-threaded, CPU)
- **Per image**: ~1-1.5 minutes
- **100 images**: ~2.5 hours
- **1,000 images**: ~25 hours (~1 day)
- **10,000 images**: ~10 days
- **50,000 images**: ~50 days (~7 weeks)

### Optimization Options

**1. GPU Rendering** (3-5x speedup)
- Enable CUDA/OptiX in Blender
- 10K images: ~2-3 days instead of 10 days

**2. Lower Quality** (2x speedup)
- Reduce samples: 64 → 32
- Lower resolution: current → 512x512
- 10K images: ~5 days

**3. Parallel Rendering** (Nx speedup)
- Run on multiple machines
- Cloud GPU instances (Vast.ai, RunPod)
- 10 machines: 10K images in ~1 day
- Cost: ~$30-60 for 10K images on RTX 4090s

**4. Combined** (10-20x speedup)
- GPU + parallel + lower quality
- 10K images: 12-24 hours
- 50K images: 2.5-5 days

---

## Next Steps

### Option A: Start Training Now (VALUE)
```bash
# VALUE dataset is ready - 180K train samples
# Update cosmos_rl config to use VALUE
# Train on GPU server (4-6 hours)
# Test on robot
```

### Option B: Generate Small Hybrid Dataset
```bash
# Generate 100 images for quality check (~2.5 hours)
cd /Users/max/Code/ChessR
/Applications/Blender.app/Contents/MacOS/Blender ChessR_datagen.blend -b -P /Users/max/Code/cosmos-chessbot/scripts/generate_hybrid_dataset.py -- --num-images 100

# If quality is good, scale up
# 1K images: ~25 hours
# 10K images: ~10 days (or 1 day with 10 parallel machines)
```

### Option C: Hybrid Approach
1. Start training with VALUE now
2. Generate 100 hybrid images in parallel
3. Evaluate both models on robot
4. If hybrid shows improvement, generate more and retrain

---

## Comparison Summary

**Visual Quality**:
- Hybrid: ✅ Good (with HDRI)
- VALUE: ✅ Good
- ChessR Original: ❓ Unknown (couldn't test)

**Position Realism**:
- Hybrid: ✅ Real Lichess games
- VALUE: ✅ Real Lichess games
- ChessR Original: ❌ Random placement

**Visual Diversity**:
- Hybrid: ✅ High (5 boards, 4 piece sets, varied lighting/angles)
- VALUE: ❌ Low (single board/piece/angle)
- ChessR Original: ✅ High (but unrealistic positions)

**Ready to Use**:
- Hybrid: ⏳ Needs rendering (hours to weeks)
- VALUE: ✅ Immediate (0 time)
- ChessR Original: ⏳ Needs rendering

**Robot Generalization** (predicted):
- Hybrid: ⭐⭐⭐⭐⭐ Excellent
- VALUE: ⭐⭐⭐ Good (if robot setup matches)
- ChessR Original: ⭐⭐ Poor (unrealistic positions)

---

## Recommendation

**Best Strategy**: Hybrid approach (Option C)

1. **This week**: Train VALUE baseline (quick validation)
2. **This week**: Generate 100 hybrid images (verify quality)
3. **Next week**: Test both models on robot
4. **If hybrid better**: Generate 10K-50K hybrid (parallel rendering)
5. **Final**: Train on hybrid dataset for competition

This gives you:
- Quick baseline to validate pipeline
- Quality verification before large-scale generation
- Empirical comparison to guide final decision
- Time to set up parallel rendering if needed
