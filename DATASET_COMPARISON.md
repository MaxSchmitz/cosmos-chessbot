# Chess FEN Detection Dataset Comparison

## Summary

We have three approaches for generating chess board training data. Here's a detailed comparison to help choose the right approach.

## Approaches

### 1. VALUE Dataset (Pre-rendered, Ready to Use)

**Source:** Zenodo download (18.5GB, 220K images)

**Pros:**
- ✅ Ready immediately (already downloaded)
- ✅ Real Lichess game positions (realistic)
- ✅ Large scale (200K train, 20K test)
- ✅ Proven baseline (VALUE paper benchmarks)
- ✅ No rendering time needed

**Cons:**
- ❌ Single board style (no variation)
- ❌ Single camera angle (overhead, fixed)
- ❌ Single lighting setup
- ❌ May not match robot camera perspective
- ❌ Won't generalize to different boards/pieces

**When to Use:**
- Quick baseline to test Cosmos-RL pipeline
- Proof that FEN detection works
- Initial model before robot testing

**Training Time:** 4-6 hours (H100, 180K samples)

**Expected Accuracy:** 85-92% (on VALUE test set, may be lower on real robot images)

---

### 2. ChessR Only (Blender Rendering)

**Source:** Blender rendering with randomization

**Pros:**
- ✅ High visual diversity (boards, pieces, lighting, angles)
- ✅ Camera angle variations (egocentric possible)
- ✅ Multiple board styles
- ✅ Multiple piece sets
- ✅ Custom HDRI lighting
- ✅ Matches robot perspective better

**Cons:**
- ❌ Random piece placement (unrealistic positions)
- ❌ Rendering time (10-20s per image)
- ❌ May learn incorrect chess patterns
- ❌ Fewer realistic game states

**When to Use:**
- If you need specific camera angles
- If you have custom boards/pieces
- If robot uses non-standard viewpoint

**Training Time:** 4-6 hours (H100) + rendering time

**Expected Accuracy:** 70-85% (lower due to unrealistic positions)

**Rendering Time:**
- 10K images: ~28-56 hours
- 50K images: ~6-12 days
- 100K images: ~12-24 days

---

### 3. Hybrid (ChessR + VALUE, Recommended)

**Source:** Blender rendering with VALUE's FEN positions

**Pros:**
- ✅ Real Lichess positions (realistic game states)
- ✅ High visual diversity (boards, pieces, lighting, angles)
- ✅ Best generalization to real robot
- ✅ Customizable to your setup
- ✅ Egocentric camera angles
- ✅ Robust to different environments

**Cons:**
- ❌ Requires rendering time
- ❌ More complex setup
- ❌ Need both ChessR and VALUE code

**When to Use:**
- **Production deployment on real robot**
- Need to match specific camera setup
- Want maximum performance
- Have time for rendering (or GPU cluster)

**Training Time:** 4-6 hours (H100) + rendering time

**Expected Accuracy:** 90-95% (best of both approaches)

**Rendering Time:** Same as ChessR (10-20s per image)

---

## Detailed Comparison Table

| Feature | VALUE | ChessR | Hybrid |
|---------|-------|--------|--------|
| **Positions** | Real Lichess ✅ | Random ❌ | Real Lichess ✅ |
| **Board Variety** | Single ❌ | Multiple ✅ | Multiple ✅ |
| **Piece Variety** | Single ❌ | Multiple ✅ | Multiple ✅ |
| **Camera Angles** | Fixed ❌ | Variable ✅ | Variable ✅ |
| **Lighting** | Single ❌ | Variable ✅ | Variable ✅ |
| **Egocentric View** | No ❌ | Yes ✅ | Yes ✅ |
| **Ready to Use** | Yes ✅ | No (render) ❌ | No (render) ❌ |
| **Rendering Time** | 0 ✅ | High ❌ | High ❌ |
| **Expected Accuracy** | 85-92% | 70-85% | 90-95% ✅ |
| **Robot Generalization** | Poor ❌ | Good ✅ | Excellent ✅ |
| **Setup Complexity** | Low ✅ | Medium | High ❌ |

---

## Recommendation by Use Case

### Scenario 1: Quick Proof of Concept
**Use VALUE Dataset**

You want to:
- Verify Cosmos-RL training works
- Test FEN detection capability
- Get results in <1 day

**Steps:**
1. Use already-converted VALUE data
2. Train Cosmos-RL (4-6 hours)
3. Evaluate on VALUE test set

**Caveat:** May not work well on real robot due to visual domain gap.

---

### Scenario 2: Robot Deployment (Recommended)
**Use Hybrid Dataset**

You want to:
- Deploy on real SO-101 robot
- Handle different boards/pieces
- Robust to lighting changes
- Match egocentric camera view

**Steps:**
1. Generate 10K hybrid images (test, ~2 days)
2. Verify quality on robot camera
3. If good, scale to 50-100K (~1-2 weeks)
4. Train Cosmos-RL (4-6 hours)
5. Deploy and test

**This is the best approach for Cosmos Cookoff submission.**

---

### Scenario 3: Research / Ablation Study
**Use All Three**

Compare:
- VALUE baseline (realistic positions, low diversity)
- ChessR (high diversity, random positions)
- Hybrid (best of both)

Document performance differences and generalization.

---

## Dataset Size Recommendations

### For Testing (Proof of Concept)
- **10,000 images** (any approach)
- Rendering time: ~2 days (hybrid/ChessR)
- Training time: ~1 hour
- Good for: Pipeline validation, initial testing

### For Development
- **50,000 images** (hybrid recommended)
- Rendering time: ~6-12 days (can parallelize)
- Training time: ~3-4 hours
- Good for: Robot deployment, iterative testing

### For Competition / Production
- **100,000+ images** (hybrid recommended)
- Rendering time: ~12-24 days (use GPU cluster)
- Training time: ~6-8 hours
- Good for: Maximum performance, edge case handling, Cosmos Cookoff

---

## Rendering Optimization Tips

If using ChessR or Hybrid:

### 1. Reduce Render Quality (Faster)
- Lower samples: 64 instead of 128 (2x speedup)
- Lower resolution: 512x512 instead of 1280x1280 (Cosmos resizes anyway)
- Disable motion blur (small speedup)

### 2. Use GPU Rendering
- Enable CUDA/OptiX in Blender
- 3-5x faster on RTX GPUs
- 10,000 images: ~6-12 hours instead of 2 days

### 3. Parallel Rendering (Best)
- Split dataset across multiple machines
- Use cloud GPU instances (Vast.ai, RunPod)
- 10 machines: 10x speedup
- 100K images in ~1-2 days instead of 2-3 weeks

**Example (Vast.ai):**
```bash
# Rent 10x RTX 4090 instances (~$0.30/hr each)
# Total cost for 100K images: ~$30-60
# Rendering time: ~10-24 hours
```

---

## Migration Path

**Week 1: Start with VALUE (immediate)**
- Train baseline model
- Test on robot
- Measure accuracy drop on real camera

**Week 2: Generate hybrid dataset (10K)**
- Render small hybrid dataset
- Compare to VALUE baseline
- Verify improvement on robot camera

**Week 3: Scale up (50-100K)**
- If hybrid works well, scale to production size
- Use GPU cluster or cloud rendering
- Final training for competition

**Week 4: Deploy and iterate**
- Integrate best model into orchestrator
- Test full manipulation pipeline
- Record submission video

---

## File Locations

### VALUE Dataset
- Images: `/Users/max/Code/cosmos-chessbot/data/data/train/`, `test/`
- Annotations: `/Users/max/Code/cosmos-chessbot/data/value_llava/chess_fen_train.json`
- Config: `scripts/cosmos_rl/chess_sft_config.toml` (current setup)

### Hybrid Dataset
- Generator: `scripts/generate_hybrid_dataset.py`
- Guide: `HYBRID_DATASET.md`
- Output: `data/hybrid_dataset/` (after rendering)

### ChessR
- Code: `/Users/max/Code/ChessR/src/`
- Blender: `/Users/max/Code/ChessR/ChessR_datagen.blend`
- Guide: ChessR README

---

## Next Steps

### If Choosing VALUE (Quick Start)
1. ✅ Already done - dataset converted
2. Set up GPU server (Task #15)
3. Train model (Task #17)
4. Test on robot

### If Choosing Hybrid (Recommended)
1. Test hybrid generator with 100 images
2. Verify quality manually
3. Generate 10K images (proof of concept)
4. Train and test on robot
5. If successful, scale to 50-100K
6. Final training and deployment

---

## Questions to Answer Before Choosing

1. **Do you have rendering time?**
   - Yes → Hybrid
   - No → VALUE

2. **Will the robot use different boards/pieces?**
   - Yes → Hybrid or ChessR
   - No (same board always) → VALUE might be OK

3. **What's the camera angle?**
   - Overhead, fixed → VALUE
   - Egocentric, angled → Hybrid

4. **What's the deadline?**
   - <1 week → VALUE
   - 2-4 weeks → Hybrid
   - Research timeline → All three (comparison)

5. **What's the accuracy target?**
   - 85% → VALUE
   - 90%+ → Hybrid

---

## Conclusion

**For Cosmos Cookoff robot deployment: Use Hybrid Dataset**

The hybrid approach provides the best balance of realistic chess positions and visual robustness, ensuring the model works reliably on the real SO-101 robot across different boards, lighting conditions, and camera angles.

Start with 10K images to validate, then scale to 50-100K for production deployment.
