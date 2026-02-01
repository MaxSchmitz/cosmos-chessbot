# Fine-tuning Cosmos-Reason2 for Chess FEN Detection

## Overview

Adapt the Cosmos Cookbook intelligent transportation recipe for chess board FEN detection.

**Original Recipe:** Traffic scene understanding (road attributes, pedestrian situations)
**Our Adaptation:** Chess board understanding (piece positions, FEN notation)

## Why Fine-tuning?

The base Cosmos-Reason2 model failed (0% accuracy) because:
- Not trained on chess-specific visual patterns
- Doesn't understand chess piece types and positions
- Lacks domain-specific vocabulary (FEN notation)

**Solution:** Fine-tune on labeled chess board images to teach:
- Visual patterns: piece shapes, colors, board squares
- Spatial reasoning: piece positions on 64 squares
- Domain vocabulary: FEN notation format

## Dataset Preparation

### Current Data
- 4 test images with FEN labels (in filenames)
- Egocentric camera angles
- Various game positions

### Required Data (Minimum for Fine-tuning)
According to the recipe, we need:
- **50-100 labeled examples** minimum
- **200-500 examples** for good performance
- Diverse positions, angles, lighting conditions

### Data Collection Strategy

**Option 1: Generate Synthetic Data**
```python
# Use chess.Board to generate positions
# Render with python-chess SVG → convert to images
# Add perspective transforms for egocentric angles
# Label with ground truth FEN
```

**Option 2: Collect Real Data**
```python
# Use SO-101 robot camera
# Manually arrange 100+ different positions
# Capture from egocentric angle
# Label with ground truth FEN
```

**Option 3: Hybrid Approach**
- 50 real images from robot camera
- 100 synthetic images with transforms
- Mix for robust training

### Data Format

Following the recipe, create dataset in JSONL format:

```json
{
  "image": "path/to/chess_board_001.jpg",
  "conversations": [
    {
      "role": "user",
      "content": "What is the FEN position of this chess board?"
    },
    {
      "role": "assistant",
      "content": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e3 0 1"
    }
  ]
}
```

## Fine-tuning Configuration

### Model
- **Base:** `nvidia/Cosmos-Reason2-8B`
- **Task:** Vision-language instruction following
- **Method:** Supervised Fine-Tuning (SFT)

### Hyperparameters (adapted from recipe)
```yaml
# Training
learning_rate: 2e-5
batch_size: 4
num_epochs: 3-5
warmup_steps: 100
gradient_accumulation_steps: 4

# LoRA (efficient fine-tuning)
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, v_proj, k_proj, o_proj]

# Hardware
device: H100 GPU (80GB VRAM)
mixed_precision: bf16
```

### Training Script Structure

```python
# Fine-tune Cosmos-Reason2 for chess FEN detection
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# 1. Load base model
model = AutoModelForVision2Seq.from_pretrained(
    "nvidia/Cosmos-Reason2-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("nvidia/Cosmos-Reason2-8B")

# 2. Configure LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 3. Load chess FEN dataset
dataset = load_dataset("json", data_files="data/chess_fen_train.jsonl")

# 4. Training loop
# ... (following recipe structure)

# 5. Save fine-tuned model
model.save_pretrained("models/cosmos-reason2-chess-fen")
```

## Prompt Engineering for Chess

### Training Prompts (vary for robustness)

```python
PROMPTS = [
    "What is the FEN position of this chess board?",
    "Convert this chess board to FEN notation.",
    "Output the FEN string for this position.",
    "Analyze this board and return the FEN.",
    "What is the current position in FEN format?",
]
```

### System Prompt

```
You are a chess position analyzer. When given an image of a chess board,
you output the position in FEN (Forsyth-Edwards Notation) format.

FEN format: piece_placement active_color castling en_passant halfmove fullmove
- Uppercase = White pieces (K Q R B N P)
- Lowercase = Black pieces (k q r b n p)
- Numbers = consecutive empty squares
- / = separates ranks (8 to 1, top to bottom)

Always output just the FEN string with no additional explanation.
```

## Implementation Steps

### Phase 1: Data Preparation (Week 1)
- [ ] Generate 100 synthetic chess positions
- [ ] Render with perspective transforms (egocentric angles)
- [ ] Create JSONL dataset with FEN labels
- [ ] Split: 80% train, 10% validation, 10% test

### Phase 2: Environment Setup (Week 1)
- [ ] Set up training environment on H100 GPU
- [ ] Install dependencies (transformers, peft, datasets)
- [ ] Download Cosmos-Reason2-8B base model
- [ ] Verify GPU memory and batch size

### Phase 3: Fine-tuning (Week 2)
- [ ] Implement training script (adapt from recipe)
- [ ] Configure LoRA for efficient fine-tuning
- [ ] Run training for 3-5 epochs
- [ ] Monitor loss and validation accuracy

### Phase 4: Evaluation (Week 2)
- [ ] Test on held-out test set
- [ ] Compare to base model (0% accuracy)
- [ ] Test on our 4 original egocentric images
- [ ] Measure accuracy improvement

### Phase 5: Integration (Week 3)
- [ ] Update RemoteCosmosPerception to use fine-tuned model
- [ ] Test end-to-end in orchestrator
- [ ] Demonstrate on real robot

## Expected Results

Based on the recipe's results:
- **Base model:** 0% accuracy (current)
- **After fine-tuning:** 70-90% accuracy (target)
- **Training time:** 2-4 hours on H100
- **Inference:** Same speed as base model

## File Structure

```
cosmos-chessbot/
├── data/
│   ├── chess_fen_train.jsonl     # Training data
│   ├── chess_fen_val.jsonl       # Validation data
│   ├── chess_fen_test.jsonl      # Test data
│   └── synthetic_boards/          # Generated images
├── scripts/
│   ├── generate_chess_dataset.py  # Create synthetic data
│   ├── finetune_cosmos_chess.py   # Training script
│   └── evaluate_finetuned.py      # Evaluation script
├── models/
│   └── cosmos-reason2-chess-fen/  # Fine-tuned model
└── notebooks/
    └── chess_finetuning.ipynb     # Interactive training

```

## Advantages Over Other Approaches

| Approach | Accuracy | Training Required | Domain-Specific |
|----------|----------|-------------------|-----------------|
| Base Cosmos-Reason2 | 0% | ❌ No | ❌ No |
| LLM Vision (GPT-5.2) | 25% | ❌ No | ❌ No |
| LiveChess2FEN | 0% | ✅ Pre-trained | ⚠️ Different domain |
| Fenify-3D | 0%* | ✅ Pre-trained | ⚠️ Different data |
| **Fine-tuned Cosmos** | **70-90%** (expected) | ✅ **Chess-specific** | ✅ **Our data** |

*Note: Failed on our specific egocentric images

## Next Steps

1. **Generate synthetic dataset** (50-100 images minimum)
2. **Set up training environment** (H100 access)
3. **Adapt training script** from recipe
4. **Run fine-tuning** (2-4 hours)
5. **Evaluate and iterate**

## Resources

- Recipe: https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason2/intelligent-transportation/post_training.html
- Cosmos-Reason2: https://huggingface.co/nvidia/Cosmos-Reason2-8B
- PEFT/LoRA: https://github.com/huggingface/peft
