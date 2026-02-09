# Model Files

This directory contains model weights for the YOLO26-DINO-MLP FEN detection pipeline.

## Current Models

### 1. YOLO26 Piece Detection

Detects chess piece bounding boxes (12 classes: white/black x pawn/rook/knight/bishop/queen/king).

**Weights:** `../runs/detect/yolo26_chess/weights/best.pt`
**Training script:** `scripts/training/train_yolo26_pieces.py`

### 2. YOLO26 Corner Pose Detection

Detects the 4 board corners as keypoints for homography computation.

**Weights:** `../runs/pose/board_corners/weights/best.pt`
**Training script:** `scripts/training/train_yolo26_corners.py`

### 3. DINO-MLP Piece Classifier

Re-classifies detected pieces using DINO ViT-S/16 features + 3-layer MLP for 99%+ accuracy.

**Weights:** `models/dino_mlp/dino_mlp_best.pth`
**Training script:** `scripts/training/train_dino_mlp_classifier.py`

## Evaluation

```bash
# Evaluate FEN detection accuracy on ChessReD2k test set
python scripts/evaluation/evaluate_fen_accuracy.py --visualize

# Benchmark YOLO-only vs YOLO-DINO
python scripts/evaluation/benchmark_fen_detectors.py
```
