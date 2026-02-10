#!/usr/bin/env python3
"""
Visualize YOLO Training Data Annotations

Displays training images with bounding boxes and class labels to verify
annotation quality before training.

Usage:
    # Visualize random samples (interactive)
    python scripts/visualize_yolo_annotations.py --num-samples 10

    # Save to output directory
    python scripts/visualize_yolo_annotations.py --num-samples 20 --output-dir annotation_check

    # Check specific image
    python scripts/visualize_yolo_annotations.py --image data/yolo26_chess/images/train/chess_0000001.jpg

    # Check validation set
    python scripts/visualize_yolo_annotations.py --split val --num-samples 10 --output-dir val_check
"""

import argparse
import random
from pathlib import Path
import cv2
import numpy as np

# Chess piece class names and colors
CLASS_NAMES = [
    'white_pawn', 'white_knight', 'white_bishop', 'white_rook', 'white_queen', 'white_king',
    'black_pawn', 'black_knight', 'black_bishop', 'black_rook', 'black_queen', 'black_king'
]

# Colors for visualization (BGR format)
COLORS = {
    # White pieces - blue shades
    0: (255, 200, 100),  # white_pawn
    1: (255, 150, 100),  # white_knight
    2: (255, 100, 100),  # white_bishop
    3: (200, 100, 255),  # white_rook
    4: (150, 100, 255),  # white_queen
    5: (100, 100, 255),  # white_king
    # Black pieces - red shades
    6: (100, 200, 255),  # black_pawn
    7: (100, 150, 255),  # black_knight
    8: (100, 100, 255),  # black_bishop
    9: (255, 100, 200),  # black_rook
    10: (255, 100, 150), # black_queen
    11: (255, 100, 100), # black_king
}


def load_yolo_annotations(label_path: Path):
    """Load YOLO format annotations from a label file."""
    annotations = []

    if not label_path.exists():
        return annotations

    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            annotations.append({
                'class_id': class_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })

    return annotations


def draw_annotations(image: np.ndarray, annotations: list):
    """Draw bounding boxes and labels on image."""
    h, w = image.shape[:2]
    vis_image = image.copy()

    # Draw each annotation
    for ann in annotations:
        class_id = ann['class_id']

        # Convert normalized coordinates to pixels
        x_center = ann['x_center'] * w
        y_center = ann['y_center'] * h
        box_width = ann['width'] * w
        box_height = ann['height'] * h

        # Calculate corners
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Get color
        color = COLORS.get(class_id, (0, 255, 0))

        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # Draw center point
        cv2.circle(vis_image, (int(x_center), int(y_center)), 3, color, -1)

        # Draw label with background
        label = CLASS_NAMES[class_id]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1

        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # Draw background rectangle
        cv2.rectangle(vis_image,
                     (x1, y1 - text_height - 8),
                     (x1 + text_width + 4, y1),
                     color, -1)

        # Draw text
        cv2.putText(vis_image, label, (x1 + 2, y1 - 4),
                   font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Add statistics overlay
    stats_text = f"Pieces: {len(annotations)}"
    cv2.putText(vis_image, stats_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    return vis_image


def visualize_dataset(
    dataset_dir: Path,
    split: str = 'train',
    num_samples: int = 10,
    output_dir: Path = None,
    specific_image: Path = None
):
    """
    Visualize YOLO dataset annotations.

    Args:
        dataset_dir: Path to YOLO dataset (contains images/ and labels/)
        split: Dataset split ('train' or 'val')
        num_samples: Number of random samples to visualize
        output_dir: If provided, save images instead of displaying
        specific_image: Visualize a specific image instead of random samples
    """
    print("=" * 60)
    print("YOLO Annotation Visualizer")
    print("=" * 60)

    if specific_image:
        # Visualize specific image
        image_paths = [specific_image]
        print(f"Visualizing: {specific_image}")
    else:
        # Get image paths
        images_dir = dataset_dir / 'images' / split
        if not images_dir.exists():
            print(f"Error: Images directory not found: {images_dir}")
            return

        image_paths = sorted(images_dir.glob('*.jpg'))
        if not image_paths:
            print(f"Error: No images found in {images_dir}")
            return

        # Random sample
        if len(image_paths) > num_samples:
            image_paths = random.sample(image_paths, num_samples)

        print(f"Dataset: {dataset_dir}")
        print(f"Split: {split}")
        print(f"Samples: {len(image_paths)}")

    print("=" * 60)
    print()

    # Create output directory if needed
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving to: {output_dir}\n")

    # Process each image
    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing {image_path.name}...")

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  Error: Could not load image")
            continue

        # Get label path
        if specific_image:
            # Try to find label file
            label_path = image_path.parent.parent / 'labels' / split / f"{image_path.stem}.txt"
            if not label_path.exists():
                # Try without split
                label_path = image_path.parent.parent / 'labels' / f"{image_path.stem}.txt"
        else:
            label_path = dataset_dir / 'labels' / split / f"{image_path.stem}.txt"

        # Load annotations
        annotations = load_yolo_annotations(label_path)
        print(f"  Annotations: {len(annotations)}")

        if not annotations:
            print(f"  Warning: No annotations found!")

        # Count pieces by type
        piece_counts = {}
        for ann in annotations:
            class_name = CLASS_NAMES[ann['class_id']]
            piece_counts[class_name] = piece_counts.get(class_name, 0) + 1

        if piece_counts:
            print(f"  Piece counts:")
            for piece_name, count in sorted(piece_counts.items()):
                print(f"    {piece_name}: {count}")

        # Draw annotations
        vis_image = draw_annotations(image, annotations)

        # Save or display
        if output_dir:
            output_path = output_dir / image_path.name
            cv2.imwrite(str(output_path), vis_image)
            print(f"  Saved to: {output_path}")
        else:
            # Display with window
            window_name = f"Annotations - {image_path.name} (Press any key for next, ESC to quit)"
            cv2.imshow(window_name, vis_image)

            key = cv2.waitKey(0)
            cv2.destroyAllWindows()

            # ESC key to quit
            if key == 27:
                print("\nVisualization stopped by user")
                break

        print()

    print("=" * 60)
    print("Visualization complete!")
    print("=" * 60)

    # Print legend
    print("\nColor Legend:")
    print("  White pieces: Blue shades")
    print("  Black pieces: Red shades")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO training data annotations"
    )

    parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=Path('data/yolo26_chess'),
        help='Path to YOLO dataset directory'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val'],
        help='Dataset split to visualize'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of random samples to visualize (default: 10)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Save visualizations to this directory instead of displaying'
    )
    parser.add_argument(
        '--image',
        type=Path,
        default=None,
        help='Visualize a specific image instead of random samples'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Validate dataset directory
    if not args.image and not args.dataset_dir.exists():
        print(f"Error: Dataset directory not found: {args.dataset_dir}")
        print("Have you converted the dataset to YOLO format?")
        print("Run: python3 scripts/convert_llava_to_yolo26_v2.py")
        exit(1)

    # Visualize
    visualize_dataset(
        dataset_dir=args.dataset_dir,
        split=args.split,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        specific_image=args.image
    )


if __name__ == '__main__':
    main()
