# Next Steps: Fine-tuning Cosmos-Reason2 for Chess FEN Detection

## Summary of Progress

We've thoroughly explored FEN detection and identified the winning strategy: **Fine-tuning Cosmos-Reason2**.

### What We've Tested (All Failed on Egocentric Images)

| Approach | Accuracy | Conclusion |
|----------|----------|------------|
| Base Cosmos-Reason2-8B | 0/4 (0%) | Not trained on chess |
| Claude Sonnet 4.5 | 0/4 (0%) | Spatial reasoning limitation |
| GPT-4o | 0/4 (0%) | Same issues |
| GPT-5.2 | 1/4 (25%) | Best LLM, still poor |
| LiveChess2FEN (fixed) | 0/4 (0%) | Wrong domain |
| Fenify-3D | 0/4 (0%) | Wrong dataset |

### The Solution: Domain-Specific Fine-tuning

Following the Cosmos Cookbook post-training recipe, we can fine-tune Cosmos-Reason2 on **chess-specific data** to achieve 70-90% accuracy.

## Current Status

✅ **Completed:**
1. Tested all alternative approaches
2. Created fine-tuning plan (COSMOS_FEN_FINETUNING_PLAN.md)
3. Created dataset generation script
4. Generated minimal dataset (4 samples) from existing images
5. Set up dataset format (JSONL with conversations)

⏳ **Next:**
1. Collect 50-100+ chess board images
2. Set up H100 GPU training environment
3. Adapt training script from recipe
4. Run fine-tuning (2-4 hours)
5. Evaluate and iterate

## Immediate Action Items

### 1. Data Collection (This Week)

**Option A: Use Robot Camera** (Recommended)
- Set up SO-101 robot with egocentric camera
- Manually arrange 50-100 different chess positions
- Capture images from consistent angle
- Label with ground truth FEN
- **Advantage:** Real data matching deployment scenario

**Option B: Generate Synthetic Data**
- Install Cairo: `brew install cairo`
- Use chess.svg to render positions
- Add perspective transforms (egocentric angles)
- Generate 100+ varied positions
- **Advantage:** Fast, controlled conditions

**Option C: Hybrid** (Best)
- 30-50 real images from robot
- 50-70 synthetic with transforms
- Mix for robust training

### 2. Training Environment Setup

**Requirements:**
- H100 GPU (80GB VRAM) or A100
- CUDA 12.1+
- PyTorch 2.0+
- Transformers, PEFT, Datasets libraries

**Cloud Options:**
- RunPod: ~$2/hour for H100
- Lambda Labs: ~$1.50/hour for A100
- Vast.ai: Variable pricing

### 3. Training Script

Adapt from recipe:
```bash
# Clone Cosmos Cookbook
git clone https://github.com/NVIDIA-Cosmos/cosmos-cookbook.git

# Adapt training script
cp cosmos-cookbook/recipes/post_training/reason2/intelligent-transportation/train.py \
   scripts/finetune_cosmos_chess.py

# Modify for chess dataset and FEN task
```

### 4. Expected Results

Based on recipe results:
- **Training time:** 2-4 hours on H100
- **Data needed:** 50-100 labeled examples minimum
- **Expected accuracy:** 70-90% (vs 0% base model)
- **Cost:** ~$4-8 for training

## Timeline Estimate

### Week 1: Data Collection
- **Mon-Wed:** Collect 50 real images + 50 synthetic
- **Thu-Fri:** Create JSONL dataset, verify labels
- **Output:** `chess_fen_train.jsonl` (80 samples), `chess_fen_val.jsonl` (10), `chess_fen_test.jsonl` (10)

### Week 2: Training & Evaluation
- **Mon:** Set up H100 environment, test dependencies
- **Tue:** Adapt training script, configure hyperparameters
- **Wed:** Run fine-tuning (2-4 hours), monitor loss
- **Thu:** Evaluate on test set, analyze errors
- **Fri:** Iterate if needed (adjust data, retrain)

### Week 3: Integration & Demo
- **Mon-Tue:** Integrate fine-tuned model into orchestrator
- **Wed-Thu:** Test end-to-end on robot
- **Fri:** Record demo video, prepare submission

## Files Created

```
cosmos-chessbot/
├── COSMOS_FEN_FINETUNING_PLAN.md     # Detailed fine-tuning plan
├── NEXT_STEPS.md                      # This file
├── data/
│   └── chess_fen_minimal.jsonl        # Minimal dataset (4 samples)
├── scripts/
│   └── generate_chess_dataset.py      # Dataset generation
└── models/
    └── cosmos-reason2-chess-fen/      # (Future) Fine-tuned model
```

## Key Resources

1. **Recipe:** https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason2/intelligent-transportation/post_training.html
2. **Cosmos-Reason2:** https://huggingface.co/nvidia/Cosmos-Reason2-8B
3. **PEFT/LoRA:** https://github.com/huggingface/peft
4. **Transformers:** https://huggingface.co/docs/transformers

## Why This Will Work

1. ✅ **Proven approach** - Recipe shows dramatic accuracy improvement
2. ✅ **Domain-specific** - Teach model chess visual patterns
3. ✅ **Aligned with judging criteria** - "Compelling application of Cosmos Reason"
4. ✅ **Practical** - 2-4 hours training, 50-100 samples
5. ✅ **Flexible** - Can iterate and improve

## Alternative: If No H100 Access

If H100/A100 GPU access is not available:

**Option 1: Use Smaller Model**
- Fine-tune Cosmos-Reason2-2B instead (less VRAM)
- Can run on RTX 4090 or similar

**Option 2: Manual FEN Input**
- Use Cosmos-Reason2 for game flow reasoning only
- Manual FEN entry or QR code pieces
- Focus demo on AI reasoning, not vision

**Option 3: Hybrid Approach**
- Use GPT-5.2 (25% accuracy) with error correction
- Cosmos-Reason2 for reasoning + validation

## Recommendation

**Go with fine-tuning Cosmos-Reason2!** This is:
- The most technically sound approach
- Aligned with competition goals
- Achievable in 2-3 weeks
- Will create a compelling demo

Start with data collection this week, and we can have a working system in 2-3 weeks.
