# Cosmos-RL Training Setup for Chess FEN Detection

Following the Cosmos Cookbook intelligent-transportation example to train Cosmos-Reason2 for chess FEN detection.

## Overview

This guide adapts the [intelligent-transportation recipe](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason2/intelligent-transportation/post_training.html) for chess FEN detection.

**Why Cosmos-RL instead of HuggingFace Transformers:**
- ✅ Optimized for Cosmos models
- ✅ Achieved 92% accuracy after 1 epoch in cookbook
- ✅ Proper vision token configuration
- ✅ Better training dynamics

## Step 1: Download VALUE Dataset

```bash
# Download from Zenodo (18.5GB)
wget https://zenodo.org/record/10607059/files/dataset.zip
unzip dataset.zip -d ~/VALUE-Dataset

# Verify structure
ls ~/VALUE-Dataset/
# Should see: images/ data/
```

## Step 2: Convert VALUE to Llava Format

```bash
cd ~/cosmos-chessbot

# Convert full dataset
uv run python scripts/convert_value_to_llava.py \
  --value-root ~/VALUE-Dataset \
  --output-dir data/value_llava

# Or test with small subset first
uv run python scripts/convert_value_to_llava.py \
  --value-root ~/VALUE-Dataset \
  --output-dir data/value_llava_test \
  --max-samples 1000
```

This creates:
- `data/value_llava/chess_fen_train.json`
- `data/value_llava/chess_fen_val.json`
- `data/value_llava/chess_fen_test.json`

## Step 3: Set Up Cosmos-Reason2 Training Environment

```bash
# Clone Cosmos-Reason2 repo
cd ~/
git clone https://github.com/nvidia-cosmos/cosmos-reason2.git
cd cosmos-reason2/examples/cosmos_rl

# Create virtual environment (separate from our project)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install uv
uv pip install -e .
```

## Step 4: Create Chess Training Configuration

Create `~/cosmos-chessbot/scripts/cosmos_rl/chess_sft_config.toml`:

```toml
[custom.dataset]
annotation_path = "/path/to/cosmos-chessbot/data/value_llava/chess_fen_train.json"
media_path = ""  # Empty because paths are absolute in JSON
system_prompt = "You are a helpful chess analysis assistant."

[custom.vision]
# Use single frame (images not videos)
nframes = 1

[train]
optm_lr = 5e-5  # Higher LR for faster learning
output_dir = "outputs/chess_fen"
optm_warmup_steps = 0  # No warmup for small datasets
optm_decay_type = "cosine"
optm_weight_decay = 0.01
train_batch_per_replica = 8  # Adjust based on GPU memory
enable_validation = true
validation_freq = 200  # Validate every 200 steps
compile = false

[policy]
model_name_or_path = "nvidia/Cosmos-Reason2-8B"
model_max_length = 32768

[logging]
logger = ['console', 'wandb']
project_name = "cosmos_chessbot"
experiment_name = "chess_fen_detection"

[train.train_policy]
type = "sft"
mini_batch = 1
dataset.test_size = 0
dataloader_num_workers = 4
dataloader_prefetch_factor = 4

[train.ckpt]
enable_checkpoint = true
save_freq = 200  # Save every 200 steps

[policy.parallelism]
tp_size = 1
cp_size = 1
dp_shard_size = 1  # Single GPU
pp_size = 1
```

## Step 5: Create Training Script

Create `~/cosmos-chessbot/scripts/cosmos_rl/chess_sft.py`:

```python
#!/usr/bin/env python3
"""Chess FEN detection training adapted from intelligent-transportation example."""

import argparse
import json
import os
from pathlib import Path

import cosmos_rl.launcher.worker_entry
import cosmos_rl.policy.config
import pydantic
import torch.utils.data
from cosmos_reason2_utils.text import create_conversation
from cosmos_reason2_utils.vision import VisionConfig


class CustomDatasetConfig(pydantic.BaseModel):
    annotation_path: str = pydantic.Field()
    media_path: str = pydantic.Field(default="")
    system_prompt: str = pydantic.Field(default="")


class CustomConfig(pydantic.BaseModel):
    dataset: CustomDatasetConfig = pydantic.Field()
    vision: VisionConfig = pydantic.Field(default=VisionConfig(nframes=1))


class ChessFENDataset(torch.utils.data.Dataset):
    """Dataset for chess FEN detection."""

    def __init__(self, config: cosmos_rl.policy.config.Config, custom_config: CustomConfig):
        self.annotation = json.load(open(custom_config.dataset.annotation_path))
        self.media_path = custom_config.dataset.media_path
        self.system_prompt = custom_config.dataset.system_prompt
        self.config = config
        self.custom_config = custom_config
        self.vision_kwargs = custom_config.vision.model_dump(exclude_none=True)

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx: int) -> list[dict]:
        sample = self.annotation[idx]

        # Extract from Llava format
        user_prompt = sample["conversations"][0]["value"]
        response = sample["conversations"][1]["value"]
        image_path = sample["image"]

        # Remove <image> tag
        import re
        user_prompt = re.sub(r"(\n)?</?image>(\n)?", "", user_prompt)

        # If media_path is set, join it
        if self.media_path != "":
            image_path = os.path.join(self.media_path, image_path)

        conversations = create_conversation(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response=response,
            images=[image_path],
            videos=None,
            vision_kwargs=self.vision_kwargs,
        )

        return conversations


def main(args):
    # Entry point for cosmos-rl training
    cosmos_rl.launcher.worker_entry.main(args)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
```

## Step 6: Train on GPU Server

```bash
# On GPU server
cd ~/cosmos-chessbot
git pull origin main  # Get latest scripts

# Activate cosmos-rl environment
cd ~/cosmos-reason2/examples/cosmos_rl
source .venv/bin/activate

# Run training
cosmos-rl --config ~/cosmos-chessbot/scripts/cosmos_rl/chess_sft_config.toml \
           ~/cosmos-chessbot/scripts/cosmos_rl/chess_sft.py
```

## Expected Results

Based on Cosmos Cookbook intelligent-transportation recipe:

| Metric | Cookbook (Traffic) | Expected (Chess) |
|--------|-------------------|------------------|
| Dataset size | 5,600 samples | 160,000+ samples (VALUE) |
| Accuracy after 1 epoch | 92% | 85-95% |
| Training time (8x A100) | 1h 16m | 3-5 hours |
| Final accuracy (3 epochs) | 93.65% | 90-95% |

**With VALUE dataset (160K+ samples) we should achieve BETTER results than cookbook!**

## Monitoring Training

```bash
# Watch training logs
tail -f outputs/chess_fen/training.log

# Check wandb (if enabled)
# Visit https://wandb.ai/

# Check checkpoints
ls -lh outputs/chess_fen/checkpoints/
```

## After Training

```bash
# Evaluate on test set
cosmos-rl --config eval_config.toml evaluate.py

# Or use our evaluation script
cd ~/cosmos-chessbot
uv run python scripts/evaluate_model.py \
  --model-path ~/cosmos-reason2/examples/cosmos_rl/outputs/chess_fen/final
```

## Troubleshooting

### OOM Error
Reduce batch size in config:
```toml
[train]
train_batch_per_replica = 4  # Reduce from 8
```

### Dataset Path Issues
Make sure paths in JSON are absolute:
```python
# In convert_value_to_llava.py, we use:
str(image_path.absolute())
```

### Cosmos-RL Import Errors
Ensure you're in the cosmos-rl virtual environment:
```bash
cd ~/cosmos-reason2/examples/cosmos_rl
source .venv/bin/activate
which python  # Should show .venv/bin/python
```

## Next Steps After Successful Training

1. **Integrate into orchestrator**
   - Update `src/cosmos_chessbot/vision/cosmos_fen_detector.py`
   - Load fine-tuned model instead of base model

2. **Deploy with NIM**
   - Quantize to FP8 for faster inference
   - Deploy with NVIDIA NIM container

3. **Test on robot**
   - Run end-to-end test with real chess board
   - Verify FEN accuracy on robot camera images

4. **Competition submission**
   - Record demo video
   - Document approach
   - Submit to Cosmos Cookoff
