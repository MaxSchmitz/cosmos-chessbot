# Cosmos-Reason2 Chess FEN Training Guide

Quick guide for fine-tuning Cosmos-Reason2 on the GPU server.

## Prerequisites

You should have already:
- ✅ Cloned the repository on GPU server
- ✅ Transferred dataset (1,943 images + JSONL files)
- ✅ GPU server with H100 or A100

## Setup (One-time)

### 1. Install uv (if not installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Sync dependencies

```bash
cd ~/cosmos-chessbot
uv sync
```

This will install all required packages:
- PyTorch with CUDA
- Transformers
- PEFT (for LoRA)
- Datasets
- Accelerate
- bitsandbytes
- wandb

## Verify Setup

Before training, verify everything is working:

```bash
uv run python scripts/verify_training_setup.py
```

This will check:
- ✅ GPU is available
- ✅ Dataset files exist and are valid
- ✅ Cosmos-Reason2-8B model loads
- ✅ Sample processing works

**If all checks pass, proceed to training!**

## Training

### Quick Start (Default Settings)

```bash
uv run python scripts/train_cosmos_chess.py
```

**Default configuration:**
- Model: `nvidia/Cosmos-Reason2-8B`
- Epochs: 3
- Batch size: 4 per device
- Gradient accumulation: 4 steps (effective batch size = 16)
- Learning rate: 2e-5
- LoRA: r=16, alpha=32, dropout=0.05
- Precision: BF16
- Output: `checkpoints/cosmos-chess-fen/`

**Expected time:** 2-4 hours on H100

### Custom Configuration

```bash
uv run python scripts/train_cosmos_chess.py \
  --num-epochs 5 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --lora-r 32 \
  --output-dir checkpoints/my-chess-model
```

### Monitor Training

The script will log:
- Training loss every 50 steps
- Evaluation loss every 500 steps
- Checkpoints saved every 500 steps

**Example output:**
```
Step 100/1163: loss=2.456
Step 200/1163: loss=1.234
Step 500/1163: loss=0.678 | Eval loss=0.712
Checkpoint saved: checkpoints/cosmos-chess-fen/checkpoint-500
```

### With Weights & Biases Logging

```bash
# Login to wandb (one-time)
uv run wandb login

# Train with wandb
uv run python scripts/train_cosmos_chess.py --wandb
```

View training curves at: https://wandb.ai/

## After Training

### 1. Check Final Model

The fine-tuned model is saved at:
```
checkpoints/cosmos-chess-fen/final/
├── adapter_config.json
├── adapter_model.safetensors
├── preprocessor_config.json
└── ...
```

### 2. Test the Model

```bash
# Test on a single image
uv run python scripts/test_finetuned_model.py \
  --model-path checkpoints/cosmos-chess-fen/final \
  --image data/chessboards/0.jpg
```

### 3. Evaluate on Test Set

```bash
uv run python scripts/evaluate_model.py \
  --model-path checkpoints/cosmos-chess-fen/final \
  --test-data data/chess_fen_test.jsonl
```

### 4. Deploy to Server

Copy the fine-tuned model back to your local machine:

```bash
# On local machine
rsync -avz --progress \
  user@gpu-server:~/cosmos-chessbot/checkpoints/cosmos-chess-fen/final/ \
  ./checkpoints/cosmos-chess-fen/final/
```

## Training Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-data` | `data/chess_fen_train.jsonl` | Training JSONL file |
| `--val-data` | `data/chess_fen_val.jsonl` | Validation JSONL file |
| `--model-name` | `nvidia/Cosmos-Reason2-8B` | Base model |
| `--output-dir` | `checkpoints/cosmos-chess-fen` | Output directory |
| `--num-epochs` | `3` | Training epochs |
| `--batch-size` | `4` | Batch size per device |
| `--gradient-accumulation-steps` | `4` | Gradient accumulation |
| `--learning-rate` | `2e-5` | Learning rate |
| `--lora-r` | `16` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha |
| `--lora-dropout` | `0.05` | LoRA dropout |
| `--warmup-steps` | `100` | Warmup steps |
| `--save-steps` | `500` | Save every N steps |
| `--eval-steps` | `500` | Eval every N steps |
| `--gradient-checkpointing` | `True` | Enable gradient checkpointing |
| `--bf16` | `True` | Use BF16 precision |
| `--wandb` | `False` | Enable W&B logging |

## Troubleshooting

### Out of Memory (OOM)

If you get OOM errors, try:

```bash
# Reduce batch size
uv run python scripts/train_cosmos_chess.py --batch-size 2

# Or increase gradient accumulation
uv run python scripts/train_cosmos_chess.py \
  --batch-size 2 \
  --gradient-accumulation-steps 8
```

### Slow Training

If training is slow:
- Check GPU utilization: `nvidia-smi -l 1`
- Ensure batch size is optimized for your GPU
- Consider using FP16 instead of BF16: `--fp16 --no-bf16`

### CUDA Version Mismatch

If PyTorch CUDA version doesn't match:

```bash
# Check CUDA version
nvcc --version

# Install matching PyTorch (example for CUDA 12.1)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Dataset Path Issues

If images aren't found, verify paths in JSONL files:

```bash
# Check first entry
head -1 data/chess_fen_train.jsonl | python -m json.tool

# If paths are wrong, regenerate JSONL on server
uv run python scripts/convert_chessboard_dataset.py
```

## Expected Results

Based on the Cosmos Cookbook recipe:

| Metric | Value |
|--------|-------|
| Training loss (final) | ~0.3-0.5 |
| Validation loss (final) | ~0.4-0.6 |
| Training time (H100) | 2-4 hours |
| Training time (A100) | 4-8 hours |
| Expected accuracy | 70-90% |

## Next Steps

After successful training:

1. Integrate model into orchestrator
2. Test on real robot images
3. Evaluate end-to-end performance
4. Record demo video

See `DATASET_READY.md` for full project roadmap.
