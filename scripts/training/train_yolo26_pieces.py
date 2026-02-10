#!/usr/bin/env python3
"""
Train YOLO26 for Chess Piece Detection

Trains YOLO26 (latest Ultralytics model) on chess piece detection task.
YOLO26 improvements over YOLOv8:
- End-to-end NMS-free design
- Better small object detection with ProgLoss + STAL
- 43% faster CPU inference

Usage:
    # Train from scratch with pretrained weights
    python scripts/train_yolo26_pieces.py

    # Resume from checkpoint
    python scripts/train_yolo26_pieces.py --resume runs/detect/yolo26_chess/weights/last.pt

    # Train with different model size
    python scripts/train_yolo26_pieces.py --model yolo26m.pt  # Medium (more accurate)
"""

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not found")
    print("Install with: pip3 install ultralytics")
    exit(1)


def train_yolo26(
    data_yaml: Path,
    model_name: str = 'yolo26n.pt',  # YOLO26n - latest with 43% faster inference
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = 'mps',  # MPS for Apple Silicon, or 'cuda' for NVIDIA GPU
    resume: str = None,
    project: str = 'runs/detect',
    name: str = 'yolo26_chess',
):
    """
    Train YOLO26 model on chess piece detection.

    YOLO26 advantages (released Jan 14, 2026):
    - 43% faster CPU inference than YOLO11
    - End-to-end NMS-free architecture
    - Better small object detection (ProgLoss + STAL)
    - MuSGD optimizer for stable training

    Args:
        data_yaml: Path to dataset YAML configuration
        model_name: Model variant (yolo26n.pt, yolo26s.pt, yolo26m.pt, etc.)
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        device: Device to use ('mps', 'cuda', or 'cpu')
        resume: Path to checkpoint to resume from
        project: Project directory for outputs
        name: Experiment name
    """
    print("=" * 60)
    print("YOLO26 Chess Piece Detection Training")
    print("=" * 60)
    print(f"Dataset: {data_yaml}")
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {device}")
    print(f"Resume: {resume or 'No'}")
    print("=" * 60)
    print()

    # Load model
    if resume:
        print(f"Resuming from checkpoint: {resume}")
        model = YOLO(resume)
    else:
        print(f"Loading pretrained {model_name}")
        model = YOLO(model_name)

    # Train
    print("\nStarting training...")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        resume=bool(resume),
        optimizer='auto',  # MuSGD for YOLO26 (>10k iterations); auto sets lr and momentum
        weight_decay=0.0005,
        warmup_epochs=3.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        patience=20,
        save=True,
        save_period=10,
        project=project,
        name=name,
        exist_ok=True,
        verbose=True,
        seed=42,
        close_mosaic=10,
        amp=True,
        val=True,
        plots=True
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best weights: {project}/{name}/weights/best.pt")
    print(f"Last weights: {project}/{name}/weights/last.pt")
    print(f"Results: {project}/{name}")
    print()
    print("Metrics:")
    print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print()
    print("Next steps:")
    print(f"1. Validate: python scripts/validate_yolo26.py")
    print(f"2. Export: model = YOLO('{project}/{name}/weights/best.pt'); model.export(format='onnx')")
    print(f"3. Train DINO-MLP: python scripts/train_dino_mlp_classifier.py")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26 for chess piece detection")

    parser.add_argument(
        '--data',
        type=Path,
        default=Path('data/yolo26_chess/chess_dataset.yaml'),
        help='Path to dataset YAML'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolo26n.pt',
        help='Model variant (yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26l.pt, yolo26x.pt)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (default: 16, reduce if OOM)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        help='Device to use: mps (Apple Silicon), cuda (NVIDIA GPU), or cpu (default: mps)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from (e.g., runs/detect/yolo26_chess/weights/last.pt)'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='runs/detect',
        help='Project directory (default: runs/detect)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='yolo26_chess',
        help='Experiment name (default: yolo26_chess)'
    )

    args = parser.parse_args()

    # Validate data file exists
    if not args.data.exists():
        print(f"Error: Dataset YAML not found: {args.data}")
        print("Run: python scripts/convert_llava_to_yolo26_v2.py first")
        exit(1)

    # Train
    train_yolo26(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        resume=args.resume,
        project=args.project,
        name=args.name,
    )


if __name__ == '__main__':
    main()
