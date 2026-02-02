# Cosmos-RL Training for Chess FEN Detection

Training scripts adapted from [Cosmos Cookbook intelligent-transportation recipe](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason2/intelligent-transportation/post_training.html).

## Files

- `chess_sft_config.toml` - Training configuration
- `chess_sft.py` - Training script
- `README.md` - This file

## Setup on GPU Server

### 1. Clone Cosmos-Reason2 Repository

```bash
cd ~/
git clone https://github.com/nvidia-cosmos/cosmos-reason2.git
cd cosmos-reason2/examples/cosmos_rl
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install uv
uv pip install -e .
```

Verify installation:
```bash
python -c "import cosmos_rl; print('cosmos-rl installed successfully')"
```

### 4. Clone Chess Project

```bash
cd ~/
git clone https://github.com/MaxSchmitz/cosmos-chessbot.git
cd cosmos-chessbot
```

### 5. Update Config Paths

Edit `scripts/cosmos_rl/chess_sft_config.toml`:

```toml
[custom.dataset]
annotation_path = "/root/cosmos-chessbot/data/value_llava/chess_fen_train.json"
```

Make paths absolute for your server.

## Training

### Start Training

```bash
# Activate cosmos-rl environment
cd ~/cosmos-reason2/examples/cosmos_rl
source .venv/bin/activate

# Run training
cosmos-rl --config ~/cosmos-chessbot/scripts/cosmos_rl/chess_sft_config.toml \
           ~/cosmos-chessbot/scripts/cosmos_rl/chess_sft.py
```

### Monitor Training

```bash
# Watch logs
tail -f ~/cosmos-reason2/examples/cosmos_rl/outputs/chess_fen/training.log

# Check GPU usage
nvidia-smi -l 5

# Weights & Biases (if enabled)
# Visit https://wandb.ai/
```

### Training Progress

Expected for 180K samples on single H100:
- **Steps per epoch**: ~22,500 (180K / batch_size 8)
- **Total steps (3 epochs)**: ~67,500
- **Time per epoch**: ~1.5-2 hours
- **Total training time**: 4-6 hours
- **Checkpoints**: Saved every 500 steps

## After Training

### Model Location

```bash
~/cosmos-reason2/examples/cosmos_rl/outputs/chess_fen/checkpoints/
```

### Evaluate Model

```bash
# Copy model back to chess project
cd ~/cosmos-chessbot

# Evaluate on test set (19,967 samples)
# TODO: Create evaluation script for cosmos-rl model
```

### Expected Results

Based on Cosmos Cookbook with 32x more data:
- **Accuracy**: 90-95% (vs cookbook's 92% with 32x less data)
- **Training loss**: < 0.5
- **Validation loss**: < 0.7

## Troubleshooting

### Out of Memory

Reduce batch size in config:
```toml
[train]
train_batch_per_replica = 4  # Reduce from 8
```

### Dataset Not Found

Check paths are absolute:
```bash
ls -la /root/cosmos-chessbot/data/value_llava/chess_fen_train.json
```

### Import Errors

Verify cosmos-rl environment:
```bash
cd ~/cosmos-reason2/examples/cosmos_rl
source .venv/bin/activate
which python  # Should show .venv/bin/python
```

## Configuration Options

### Batch Size

Adjust based on GPU memory:
- H100 80GB: `train_batch_per_replica = 8-16`
- A100 40GB: `train_batch_per_replica = 4-8`
- A100 80GB: `train_batch_per_replica = 16`

### Learning Rate

Current: `5e-5` (higher than cookbook for faster convergence)
- Increase to `1e-4` for even faster training (may reduce accuracy)
- Decrease to `2e-5` for more stable training (cookbook default)

### Validation Frequency

Current: Every 500 steps
- Increase to `1000` for faster training (less overhead)
- Decrease to `200` for more frequent monitoring

### Multi-GPU Training

For 8x GPUs, update config:
```toml
[policy.parallelism]
dp_shard_size = 8  # Use 8 GPUs
```

## Next Steps

1. Train model (4-6 hours)
2. Evaluate on test set
3. Integrate into orchestrator
4. Test on real robot
5. Submit to Cosmos Cookoff!
