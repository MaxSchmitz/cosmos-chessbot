# YOLO26-DINO-MLP FEN Detection Implementation

State-of-the-art chess FEN detection using YOLO26 for piece detection + DINO-MLP for classification.

**Target Accuracy:** 99%+ (based on Stanford CS231N paper methodology)

## Overview

This implementation combines:
1. **YOLO26** - Latest Ultralytics model (Jan 2026) with 43% faster inference, end-to-end NMS-free architecture
2. **DINO** - Self-supervised vision transformer for feature extraction
3. **MLP** - Lightweight classifier trained on DINO features

### Why YOLO26?

**Released January 14, 2026** with breakthrough improvements:
- ✅ **43% faster CPU inference** than YOLO11
- ✅ **End-to-end NMS-free** (no post-processing needed)
- ✅ **Better small object detection** via ProgLoss + STAL (perfect for chess pieces!)
- ✅ **MuSGD optimizer** for stable training
- ✅ **Simplified architecture** for edge deployment

### Why This Approach?

| Method | Accuracy | Speed | Strengths |
|--------|----------|-------|-----------|
| **YOLO26-DINO-MLP** | 99%+ | Fastest | YOLO26 speed + DINO accuracy + end-to-end |
| YOLO26-only | 95-98% | Very Fast | Good for real-time, may misclassify similar pieces |
| YOLO11-DINO-MLP | 99%+ | Fast | Previous gen (43% slower inference) |
| Fenify-3D | 85-95% | Medium | Works on real images, struggles with synthetic |
| LLM-based | Variable | Slow | Flexible but inconsistent |

## Implementation Status

✅ **Completed:**
- [x] Dataset conversion (Llava JSON → YOLO26 format)
- [x] Bounding box generation (programmatic from rendering)
- [x] YOLO11 training script
- [x] DINO-MLP classifier implementation
- [x] Integrated FEN detector
- [x] Benchmarking framework

📊 **Dataset:**
- **1,818 images** with 42,846 piece annotations
- **Train:** 1,635 images (38,336 pieces)
- **Val:** 183 images (4,510 pieces)
- **Format:** YOLO detection format with 12 piece classes

## Quick Start

### 1. Install Dependencies

```bash
pip3 install ultralytics transformers torch torchvision pillow opencv-python python-chess tqdm
```

### 2. Dataset Preparation (COMPLETED)

The dataset has already been converted to YOLO format:

```
data/yolo26_chess/
├── images/
│   ├── train/ (1,635 images)
│   └── val/   (183 images)
├── labels/
│   ├── train/ (38,336 piece annotations)
│   └── val/   (4,510 piece annotations)
└── chess_dataset.yaml
```

### 3. Train YOLO26 Detector

```bash
# Train YOLO26 nano (fast, good accuracy, 43% faster inference)
python3 scripts/train_yolo26_pieces.py --model yolo26n.pt --epochs 100 --batch 16 --device mps

# Or train YOLO26 medium (slower training, higher accuracy)
python3 scripts/train_yolo26_pieces.py --model yolo26m.pt --epochs 100 --batch 8 --device mps
```

**YOLO26 advantages:**
- ✅ End-to-end NMS-free (faster, simpler)
- ✅ Better small object detection (ProgLoss + STAL)
- ✅ 43% faster inference than YOLO11
- ✅ MuSGD optimizer for better convergence

**Expected results:**
- **mAP50:** >95%
- **mAP50-95:** >87% (better than YOLO11 due to improved small object detection)
- **Training time:** 2-4 hours on Apple Silicon GPU

**Output:** `runs/detect/yolo26_chess/weights/best.pt`

### 4. Train DINO-MLP Classifier

```bash
# Extract piece crops from YOLO detections
python3 scripts/train_dino_mlp_classifier.py --extract-crops --yolo-weights runs/detect/yolo26_chess/weights/best.pt

# Train MLP classifier (only 10 epochs needed!)
python3 scripts/train_dino_mlp_classifier.py --train --epochs 10 --batch-size 32 --device mps

# Or combined
python3 scripts/train_dino_mlp_classifier.py --extract-crops --train
```

**Expected results:**
- **Classification accuracy:** >99.5%
- **Training time:** <30 minutes
- **Piece crops:** ~40,000 training samples

**Output:** `models/dino_mlp/dino_mlp_best.pth`

### 5. Inference

```python
from cosmos_chessbot.vision.yolo_dino_detector import YOLODINOFenDetector
import cv2

# Initialize detector
detector = YOLODINOFenDetector(
    yolo_weights='runs/detect/yolo26_chess/weights/best.pt',
    mlp_weights='models/dino_mlp/dino_mlp_best.pth',
    device='mps'
)

# Detect FEN from image
image = cv2.imread('test_board.jpg')
fen = detector.detect_fen(image)
print(f"FEN: {fen}")

# Get detailed results
result = detector.detect_fen_with_metadata(image)
print(f"FEN: {result['fen']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Pieces detected: {len(result['pieces'])}")
```

### 6. Benchmark Performance

```bash
# Benchmark on validation set
python3 scripts/benchmark_fen_detectors.py \
    --test-dir data/chess_with_bboxes \
    --yolo-weights runs/detect/yolo26_chess/weights/best.pt \
    --mlp-weights models/dino_mlp/dino_mlp_best.pth \
    --num-samples 100
```

**Metrics tracked:**
- Per-square accuracy
- Per-piece classification accuracy
- Full board exact match rate
- Inference speed (FPS)

## Architecture Details

### YOLO26 Detector

**Model:** YOLO26n (nano) or YOLO26m (medium)

**Key improvements over YOLO11:**
- End-to-end NMS-free architecture (no post-processing)
- ProgLoss + STAL for better small object detection
- MuSGD optimizer (inspired by Moonshot AI's Kimi K2)
- 43% faster CPU inference

**Training configuration:**
- **Input size:** 640×640
- **Batch size:** 16 (nano) or 8 (medium)
- **Epochs:** 100
- **Optimizer:** MuSGD (YOLO26's improved optimizer)
- **Augmentations:** Mosaic, flip, perspective, color jitter

**Classes (12 total):**
```
0: white_pawn    6: black_pawn
1: white_knight  7: black_knight
2: white_bishop  8: black_bishop
3: white_rook    9: black_rook
4: white_queen   10: black_queen
5: white_king    11: black_king
```

### DINO-MLP Classifier

**Architecture:**
```
Input Image (cropped piece)
    ↓
DINO ViT-S/16 (frozen, pretrained)
    ↓
[CLS] token (384-dim features)
    ↓
Linear(384 → 256) + ReLU + Dropout(0.1)
    ↓
Linear(256 → 128) + ReLU + Dropout(0.1)
    ↓
Linear(128 → 12)
    ↓
Softmax → 12 class probabilities
```

**Training:**
- **Optimizer:** AdamW (lr=0.001, weight_decay=0.005)
- **Loss:** CrossEntropyLoss
- **Epochs:** 10 (from Stanford paper - more is unnecessary!)
- **Batch size:** 32

**Why DINO?**
- Self-supervised pretraining (ImageNet)
- Excellent generalization to new visual styles
- No manual feature engineering
- Only MLP needs training (fast!)

### Inference Pipeline

```
1. Load image
    ↓
2. YOLO26 detects piece bounding boxes (end-to-end, no NMS)
    ↓
3. For each detected piece:
   a. Crop piece image
   b. Extract DINO features (frozen)
   c. Classify with MLP
   d. Map pixel position → chess square
    ↓
4. Place pieces on chess.Board
    ↓
5. Generate FEN string
```

## Dataset Generation

### Option 1: Use Existing Dataset with Estimated Bboxes

Already completed! The existing 1,818 images have been processed with estimated bounding boxes.

```bash
# Add bounding boxes to existing data
python3 scripts/add_bboxes_to_existing_data.py \
    --input data/chess \
    --output data/chess_with_bboxes

# Convert to YOLO format
python3 scripts/convert_llava_to_yolo26_v2.py \
    --input data/chess_with_bboxes \
    --output data/yolo26_chess
```

### Option 2: Generate New Dataset with Ground Truth Bboxes (Recommended)

For maximum accuracy, regenerate the dataset with the updated rendering script that saves ground truth bounding boxes during rendering:

```bash
# Render new dataset with bounding boxes
cd /Users/max/Code/ChessR
blender ChessR_datagen.blend -b -P /Users/max/Code/cosmos-chessbot/scripts/generate_hybrid_dataset.py -- \
    --num-images 5000 \
    --output-dir /Users/max/Code/cosmos-chessbot/data/chess_new

# Convert to YOLO format
cd /Users/max/Code/cosmos-chessbot
python3 scripts/convert_llava_to_yolo26_v2.py \
    --input data/chess_new \
    --output data/yolo26_chess_new
```

**Updated rendering script includes:**
- Exact 3D bounding box calculation in Blender
- Camera projection to 2D screen space
- Normalized YOLO-format coordinates
- Saved in annotations.json

## Performance Targets

Based on Stanford CS231N paper results:

| Metric | Target | Notes |
|--------|--------|-------|
| YOLO26 mAP50-95 | >87% | Piece detection (better than YOLO11) |
| DINO-MLP accuracy | >99.5% | Piece classification |
| Per-square accuracy | >96% | Board reconstruction |
| Full board FEN match | >87% | Complete position |
| Inference time | <60ms | 43% faster than YOLO11 (real-time capable) |

## Files Created

### Scripts
- `scripts/convert_llava_to_yolo26_v2.py` - Dataset conversion (uses pre-computed bboxes)
- `scripts/add_bboxes_to_existing_data.py` - Add bboxes to existing dataset
- `scripts/train_yolo26_pieces.py` - YOLO26 training
- `scripts/train_dino_mlp_classifier.py` - DINO-MLP training
- `scripts/benchmark_fen_detectors.py` - Performance benchmarking

### Source Code
- `src/cosmos_chessbot/vision/yolo_dino_detector.py` - Main detector class

### Configuration
- `data/yolo26_chess/chess_dataset.yaml` - YOLO dataset config

### Updated Scripts
- `scripts/generate_hybrid_dataset.py` - Now saves bounding boxes during rendering

## Next Steps

1. **Train YOLO26** (2-4 hours)
   ```bash
   python3 scripts/train_yolo26_pieces.py
   ```

2. **Train DINO-MLP** (<30 min)
   ```bash
   python3 scripts/train_dino_mlp_classifier.py --extract-crops --train
   ```

3. **Benchmark** (5-10 min)
   ```bash
   python3 scripts/benchmark_fen_detectors.py
   ```

4. **Integration** with robot vision system
   - Add to perception pipeline
   - Test on real camera images
   - Fine-tune if needed

## Troubleshooting

### Out of Memory (OOM) during training

**YOLO26:**
```bash
# Reduce batch size
python3 scripts/train_yolo26_pieces.py --batch 8  # or 4

# Or use smaller model (already using yolo26n.pt by default)
python3 scripts/train_yolo26_pieces.py --model yolo26n.pt
```

**DINO-MLP:**
```bash
# Reduce batch size
python3 scripts/train_dino_mlp_classifier.py --train --batch-size 16
```

### Slow training on CPU

```bash
# Force CPU (if MPS/CUDA not working)
python3 scripts/train_yolo26_pieces.py --device cpu

# Or use smaller image size
python3 scripts/train_yolo26_pieces.py --imgsz 416  # instead of 640
```

### Low accuracy

1. **More training data:** Generate 5,000-10,000 images instead of 1,818
2. **Longer training:** Increase epochs (YOLO: 150-200, DINO-MLP: 15-20)
3. **Larger model:** Use `yolo26m.pt` or `yolo26l.pt`
4. **Better bounding boxes:** Re-render with ground truth bboxes

### Resume interrupted training

**YOLO26:**
```bash
python3 scripts/train_yolo26_pieces.py --resume runs/detect/yolo26_chess/weights/last.pt
```

**DINO-MLP:**
Training is fast (<30 min), just restart from scratch.

## References

- **Stanford CS231N paper:** "Chess Position Recognition using Deep Learning" (99.83% accuracy with YOLOv8)
- **YOLO26:** Ultralytics YOLO26 (January 14, 2026) - https://docs.ultralytics.com/models/yolo26/
- **DINO:** "Emerging Properties in Self-Supervised Vision Transformers" (Facebook AI, 2021)
- **Fenify-3D:** ChessVision.ai open-source FEN detector

## License

Part of cosmos-chessbot project.
