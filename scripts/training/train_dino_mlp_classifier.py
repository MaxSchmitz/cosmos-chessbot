#!/usr/bin/env python3
"""
Train DINO-MLP Chess Piece Classifier

Uses pretrained DINO (self-supervised ViT) for feature extraction + lightweight MLP for classification.
Achieves 99.8%+ accuracy with only 10 epochs (from Stanford CS231N paper).

Usage:
    # Step 1: Extract piece crops from YOLO detections
    python scripts/train_dino_mlp_classifier.py --extract-crops

    # Step 2: Train MLP classifier
    python scripts/train_dino_mlp_classifier.py --train

    # Combined
    python scripts/train_dino_mlp_classifier.py --extract-crops --train
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

try:
    from transformers import ViTImageProcessor, ViTModel
    from ultralytics import YOLO
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip3 install transformers ultralytics pillow torch torchvision")
    exit(1)


# Class mapping
CLASS_NAMES = [
    'white_pawn', 'white_knight', 'white_bishop', 'white_rook', 'white_queen', 'white_king',
    'black_pawn', 'black_knight', 'black_bishop', 'black_rook', 'black_queen', 'black_king'
]


class DINOChessPieceClassifier(nn.Module):
    """DINO + MLP classifier for chess pieces (from Stanford CS231N paper)."""

    def __init__(self, num_classes=12, freeze_dino=True):
        super().__init__()

        # Pretrained DINO ViT-S/16 (smaller and faster than ViT-S/8)
        self.dino = ViTModel.from_pretrained('facebook/dino-vits16')
        self.dino_feature_dim = self.dino.config.hidden_size  # 384 for ViT-S

        # Freeze DINO weights
        if freeze_dino:
            for param in self.dino.parameters():
                param.requires_grad = False

        # 3-layer MLP classifier (from Stanford paper)
        self.classifier = nn.Sequential(
            nn.Linear(self.dino_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, pixel_values):
        # Extract DINO features
        with torch.no_grad():
            outputs = self.dino(pixel_values=pixel_values)
            features = outputs.last_hidden_state[:, 0]  # CLS token

        # Classify
        logits = self.classifier(features)
        return logits


class ChessPieceDataset(Dataset):
    """Dataset of cropped chess piece images with labels."""

    def __init__(self, crops_dir: Path, split: str = 'train'):
        self.crops_dir = crops_dir / split
        self.split = split

        # Load metadata
        metadata_path = self.crops_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        print(f"Loaded {len(self.metadata)} {split} samples")

        # DINO preprocessing
        self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        image_path = self.crops_dir / item['filename']
        label = item['class_id']

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors='pt')
        pixel_values = inputs['pixel_values'].squeeze(0)

        return pixel_values, label


def extract_piece_crops(
    yolo_weights: Path,
    dataset_dir: Path,
    output_dir: Path,
    device: str = 'mps'
):
    """
    Extract piece crops from images using YOLO detections.

    Args:
        yolo_weights: Path to trained YOLO weights
        dataset_dir: Path to YOLO dataset directory
        output_dir: Output directory for cropped pieces
        device: Device to use
    """
    print("=" * 60)
    print("Extracting piece crops from YOLO detections")
    print("=" * 60)

    # Load YOLO model
    print(f"Loading YOLO model from {yolo_weights}")
    model = YOLO(str(yolo_weights))

    # Process train and val splits
    for split in ['train', 'val']:
        print(f"\nProcessing {split} set...")

        images_dir = dataset_dir / 'images' / split
        labels_dir = dataset_dir / 'labels' / split
        output_split_dir = output_dir / split
        output_split_dir.mkdir(parents=True, exist_ok=True)

        if not images_dir.exists():
            print(f"Skipping {split} - directory not found")
            continue

        image_files = sorted(images_dir.glob('*.jpg'))
        metadata = []
        crop_count = 0

        for image_path in tqdm(image_files, desc=f"Processing {split}"):
            # Read ground truth labels
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue

            # Load image
            image = Image.open(image_path).convert('RGB')
            img_width, img_height = image.size

            # Parse labels
            with open(label_path) as f:
                labels = [line.strip().split() for line in f if line.strip()]

            # Crop each piece
            for label_data in labels:
                class_id = int(label_data[0])
                x_center = float(label_data[1])
                y_center = float(label_data[2])
                width = float(label_data[3])
                height = float(label_data[4])

                # Convert to pixel coordinates
                x1 = int((x_center - width / 2) * img_width)
                y1 = int((y_center - height / 2) * img_height)
                x2 = int((x_center + width / 2) * img_width)
                y2 = int((y_center + height / 2) * img_height)

                # Ensure valid crop
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_width, x2)
                y2 = min(img_height, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                # Crop piece
                piece_crop = image.crop((x1, y1, x2, y2))

                # Save crop
                crop_filename = f"{image_path.stem}_piece_{crop_count:04d}.jpg"
                crop_path = output_split_dir / crop_filename
                piece_crop.save(crop_path)

                # Add to metadata
                metadata.append({
                    'filename': crop_filename,
                    'class_id': class_id,
                    'class_name': CLASS_NAMES[class_id],
                    'source_image': image_path.name,
                })

                crop_count += 1

        # Save metadata
        metadata_path = output_split_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  Extracted {crop_count} pieces to {output_split_dir}")

    print("\nCrop extraction complete!")


def train_classifier(
    crops_dir: Path,
    output_dir: Path,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
    device: str = 'mps'
):
    """
    Train DINO-MLP classifier on piece crops.

    Args:
        crops_dir: Directory with cropped pieces
        output_dir: Output directory for model weights
        epochs: Number of training epochs (10 is enough per Stanford paper)
        batch_size: Batch size
        lr: Learning rate
        device: Device to use
    """
    print("=" * 60)
    print("Training DINO-MLP Classifier")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_dataset = ChessPieceDataset(crops_dir, split='train')
    val_dataset = ChessPieceDataset(crops_dir, split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Create model
    print(f"\nLoading DINO-MLP model...")
    model = DINOChessPieceClassifier(num_classes=12, freeze_dino=True)
    model = model.to(device)

    # Optimizer and loss (from Stanford paper)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=0.005)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for pixel_values, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for pixel_values, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                pixel_values = pixel_values.to(device)
                labels = labels.to(device)

                outputs = model(pixel_values)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total

        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights_path = output_dir / 'dino_mlp_best.pth'
            torch.save(model.state_dict(), best_weights_path)
            print(f"  Saved best model: {best_weights_path}")

    # Save final model
    final_weights_path = output_dir / 'dino_mlp_final.pth'
    torch.save(model.state_dict(), final_weights_path)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best weights: {output_dir / 'dino_mlp_best.pth'}")
    print(f"Final weights: {final_weights_path}")
    print()
    print("Next steps:")
    print("  python scripts/yolo26_dino_detector.py  # Create inference pipeline")


def main():
    parser = argparse.ArgumentParser(description="Train DINO-MLP chess piece classifier")

    parser.add_argument(
        '--extract-crops',
        action='store_true',
        help='Extract piece crops from YOLO detections'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train DINO-MLP classifier'
    )
    parser.add_argument(
        '--yolo-weights',
        type=Path,
        default=Path('runs/detect/yolo26_chess/weights/best.pt'),
        help='Path to trained YOLO weights'
    )
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=Path('data/yolo26_chess'),
        help='Path to YOLO dataset directory'
    )
    parser.add_argument(
        '--crops-dir',
        type=Path,
        default=Path('data/piece_crops'),
        help='Directory for piece crops'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('models/dino_mlp'),
        help='Output directory for trained model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        help='Device to use (mps, cuda, or cpu)'
    )

    args = parser.parse_args()

    # Extract crops if requested
    if args.extract_crops:
        if not args.yolo_weights.exists():
            print(f"Error: YOLO weights not found: {args.yolo_weights}")
            print("Train YOLO first: python scripts/train_yolo26_pieces.py")
            exit(1)

        extract_piece_crops(
            yolo_weights=args.yolo_weights,
            dataset_dir=args.dataset_dir,
            output_dir=args.crops_dir,
            device=args.device
        )

    # Train classifier if requested
    if args.train:
        if not args.crops_dir.exists():
            print(f"Error: Crops directory not found: {args.crops_dir}")
            print("Extract crops first: python scripts/train_dino_mlp_classifier.py --extract-crops")
            exit(1)

        train_classifier(
            crops_dir=args.crops_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device
        )

    if not args.extract_crops and not args.train:
        parser.print_help()


if __name__ == '__main__':
    main()
