# Model Files

This directory contains the model files needed for FEN detection.

## Required Models

### 1. Ultimate V2 Board Segmentation (ONNX)

**Purpose:** Fast chess board detection and segmentation
**Speed:** ~15ms on CPU
**Size:** 2.09MB

**Download:**
```bash
# Download from Hugging Face
wget https://huggingface.co/yamero999/ultimate-v2-chess-onnx/resolve/main/ultimate_v2_breakthrough_accurate.onnx

# Or use huggingface-cli
huggingface-cli download yamero999/ultimate-v2-chess-onnx ultimate_v2_breakthrough_accurate.onnx --local-dir .
```

**Expected file:**
- `models/ultimate_v2_breakthrough_accurate.onnx`

### 2. YOLO Chess Piece Detection

**Purpose:** Detect and classify chess pieces on the board
**Classes:** 12 (white/black × pawn/rook/knight/bishop/queen/king)

**Download:**
```bash
# Download from Hugging Face
wget https://huggingface.co/dopaul/chessboard-detector/resolve/main/best.pt -O chess_piece_yolo.pt

# Or use huggingface-cli
huggingface-cli download dopaul/chessboard-detector best.pt --local-dir .
mv best.pt chess_piece_yolo.pt
```

**Expected file:**
- `models/chess_piece_yolo.pt`

## Quick Setup

```bash
cd models/

# Download Ultimate V2 ONNX
wget https://huggingface.co/yamero999/ultimate-v2-chess-onnx/resolve/main/ultimate_v2_breakthrough_accurate.onnx

# Download YOLO piece detector
wget https://huggingface.co/dopaul/chessboard-detector/resolve/main/best.pt -O chess_piece_yolo.pt

# Verify files
ls -lh
```

Expected output:
```
ultimate_v2_breakthrough_accurate.onnx  (2.1MB)
chess_piece_yolo.pt                     (varies)
```

## Alternative: Using Hugging Face CLI

```bash
pip install huggingface_hub[cli]

# Download both models
huggingface-cli download yamero999/ultimate-v2-chess-onnx ultimate_v2_breakthrough_accurate.onnx --local-dir models/
huggingface-cli download dopaul/chessboard-detector best.pt --local-dir models/
mv models/best.pt models/chess_piece_yolo.pt
```

## Model Info

### Ultimate V2 ONNX
- **Model Card:** https://huggingface.co/yamero999/ultimate-v2-chess-onnx
- **License:** Apache 2.0
- **Input:** 256x256 RGB image
- **Output:** 256x256 segmentation mask

### YOLO Piece Detector
- **Model Card:** https://huggingface.co/dopaul/chessboard-detector
- **Framework:** Ultralytics YOLOv8/v11
- **Input:** Variable size image
- **Output:** Bounding boxes + class labels

## Troubleshooting

**Models not found error:**
```
FileNotFoundError: models/ultimate_v2_breakthrough_accurate.onnx not found
```
Solution: Download the models using commands above

**Import error:**
```
ImportError: ultralytics not found
```
Solution: `pip install ultralytics`

**ONNX error:**
```
ImportError: onnxruntime not found
```
Solution: `pip install onnxruntime`
