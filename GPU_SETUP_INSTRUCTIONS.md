# GPU Instance Setup Instructions

## 1. Clone Repository on GPU Server

```bash
# SSH into your GPU instance
ssh user@your-gpu-instance

# Clone the repository
git clone https://github.com/MaxSchmitz/cosmos-chessbot.git
cd cosmos-chessbot

# Verify you have the latest code
git log -1
# Should show: "Add Cosmos-Reason2 fine-tuning pipeline for chess FEN detection"
```

## 2. Upload Dataset to GPU Server

The 447MB chessboards dataset was excluded from git. Upload it separately:

### Option A: Using rsync (Recommended)

```bash
# From your local machine
rsync -avz --progress \
  /Users/max/Code/cosmos-chessbot/data/chessboards/ \
  user@your-gpu-instance:~/cosmos-chessbot/data/chessboards/
```

### Option B: Using scp

```bash
# From your local machine
scp -r /Users/max/Code/cosmos-chessbot/data/chessboards \
  user@your-gpu-instance:~/cosmos-chessbot/data/
```

### Option C: Download from Cloud Storage

If you've uploaded to cloud storage:

```bash
# On GPU instance
cd ~/cosmos-chessbot/data
wget https://your-storage-url/chessboards.tar.gz
tar -xzf chessboards.tar.gz
```

## 3. Verify Dataset

```bash
# On GPU instance
cd ~/cosmos-chessbot

# Check dataset files
ls -lh data/*.jsonl
# Should see: chess_fen_train.jsonl (1554 samples)
#             chess_fen_val.jsonl (194 samples)
#             chess_fen_test.jsonl (195 samples)

# Count images
ls data/chessboards/*.jpg | wc -l
# Should show: 1943

# Check sample entry
head -1 data/chess_fen_train.jsonl | python3 -m json.tool
```

## 4. Install Dependencies

```bash
# On GPU instance
cd ~/cosmos-chessbot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install transformers and training dependencies
pip install transformers>=4.36.0
pip install peft>=0.7.0
pip install datasets>=2.14.0
pip install accelerate>=0.25.0
pip install bitsandbytes>=0.41.0
pip install wandb  # Optional: for experiment tracking

# Install other project dependencies
pip install -e .
```

## 5. Verify GPU

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA in PyTorch
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

## 6. Test Dataset Loading

```bash
# On GPU instance
python3 -c "
import json
from pathlib import Path

# Load sample
with open('data/chess_fen_train.jsonl') as f:
    sample = json.loads(f.readline())

print('Image path:', sample['image'])
print('User prompt:', sample['conversations'][0]['content'])
print('FEN answer:', sample['conversations'][1]['content'])

# Verify image exists
image_path = Path(sample['image'])
print(f'Image exists: {image_path.exists()}')
print(f'Image size: {image_path.stat().st_size / 1024:.1f}KB')
"
```

## 7. Download Cosmos-Reason2 Base Model

```bash
# On GPU instance
python3 -c "
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch

print('Downloading Cosmos-Reason2-8B...')
model = AutoModelForVision2Seq.from_pretrained(
    'nvidia/Cosmos-Reason2-8B',
    torch_dtype=torch.bfloat16,
    device_map='auto',
    cache_dir='./models/cache'
)
processor = AutoProcessor.from_pretrained(
    'nvidia/Cosmos-Reason2-8B',
    cache_dir='./models/cache'
)
print('✅ Model downloaded successfully!')
"
```

## 8. Ready to Train!

Once all steps are complete, you're ready to:

1. Adapt the training script from Cosmos Cookbook
2. Run fine-tuning (2-4 hours)
3. Evaluate on test set

## Quick Verification Checklist

- [ ] Repository cloned
- [ ] Latest code pulled (eb856f6 commit)
- [ ] Dataset uploaded (1943 images)
- [ ] Dependencies installed
- [ ] GPU detected
- [ ] Dataset loads correctly
- [ ] Base model downloaded

## Troubleshooting

### Dataset paths incorrect
If image paths in JSONL don't match server paths, regenerate JSONL on server:

```bash
cd ~/cosmos-chessbot
python scripts/convert_chessboard_dataset.py
```

### Out of memory during model download
Use smaller precision or model sharding:

```python
model = AutoModelForVision2Seq.from_pretrained(
    'nvidia/Cosmos-Reason2-8B',
    torch_dtype=torch.float16,  # Use FP16 instead of BF16
    device_map='auto',
    max_memory={0: '40GB'}  # Adjust to your GPU
)
```

### CUDA version mismatch
Install PyTorch for your specific CUDA version:

```bash
# Check CUDA version
nvcc --version

# Install matching PyTorch
# For CUDA 11.8: cu118
# For CUDA 12.1: cu121
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Next: Training Script

See `COSMOS_FEN_FINETUNING_PLAN.md` for training configuration and script adaptation from the Cosmos Cookbook recipe.
