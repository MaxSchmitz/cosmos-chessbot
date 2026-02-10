#!/usr/bin/env python3
"""
Convert Llava Chess Dataset to YOLO26 Format

Converts the hybrid chess dataset from Llava JSON format to YOLO26 detection format:
- Parses FEN strings from annotations.json
- Calculates bounding boxes for each piece
- Generates YOLO format labels (class_id x_center y_center width height)
- Splits into train/val (90/10)

Usage:
    python scripts/convert_llava_to_yolo26.py --input data/chess --output data/yolo26_chess
"""

import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import random

try:
    import chess
    from PIL import Image
    import numpy as np
    import cv2
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install python-chess pillow numpy opencv-python")
    exit(1)


# FEN piece to YOLO class mapping (12 classes)
FEN_TO_CLASS = {
    'P': 0,  # white_pawn
    'N': 1,  # white_knight
    'B': 2,  # white_bishop
    'R': 3,  # white_rook
    'Q': 4,  # white_queen
    'K': 5,  # white_king
    'p': 6,  # black_pawn
    'n': 7,  # black_knight
    'b': 8,  # black_bishop
    'r': 9,  # black_rook
    'q': 10, # black_queen
    'k': 11, # black_king
}

CLASS_NAMES = [
    'white_pawn', 'white_knight', 'white_bishop', 'white_rook', 'white_queen', 'white_king',
    'black_pawn', 'black_knight', 'black_bishop', 'black_rook', 'black_queen', 'black_king'
]


def detect_board_corners(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect chessboard corners using Hough line detection.

    Returns:
        board_corners: 4x2 array of corner coordinates [top-left, top-right, bottom-right, bottom-left]
        board_mask: Binary mask of the detected board region
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to find board edges
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    # Use Hough line detection to find dominant lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                           minLineLength=min(w, h) * 0.2,
                           maxLineGap=20)

    # Fallback to central region if board detection fails
    board_corners = None

    if lines is not None and len(lines) > 10:
        # Cluster lines by angle to find horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            # Horizontal lines (angle near 0 or 180)
            if abs(angle) < 20 or abs(angle - 180) < 20:
                horizontal_lines.append((min(y1, y2), max(y1, y2)))

            # Vertical lines (angle near 90 or -90)
            elif abs(abs(angle) - 90) < 20:
                vertical_lines.append((min(x1, x2), max(x1, x2)))

        # Find bounding box from line clusters
        if horizontal_lines and vertical_lines:
            h_lines = sorted(horizontal_lines, key=lambda x: x[0])
            v_lines = sorted(vertical_lines, key=lambda x: x[0])

            # Use median lines to avoid outliers
            top_y = np.median([y[0] for y in h_lines[:len(h_lines)//3]])
            bottom_y = np.median([y[1] for y in h_lines[-len(h_lines)//3:]])
            left_x = np.median([x[0] for x in v_lines[:len(v_lines)//3]])
            right_x = np.median([x[1] for x in v_lines[-len(v_lines)//3:]])

            # Validate detection (board should be reasonably sized)
            board_width = right_x - left_x
            board_height = bottom_y - top_y

            if (board_width > w * 0.3 and board_height > h * 0.3 and
                board_width < w * 0.95 and board_height < h * 0.95):
                board_corners = np.array([
                    [left_x, top_y],
                    [right_x, top_y],
                    [right_x, bottom_y],
                    [left_x, bottom_y]
                ], dtype=np.float32)

    # Fallback: Use largest centered square region
    if board_corners is None:
        # Assume board is centered and takes up 70-80% of image
        board_size = int(min(w, h) * 0.75)
        center_x, center_y = w // 2, h // 2
        half_size = board_size // 2

        board_corners = np.array([
            [center_x - half_size, center_y - half_size],
            [center_x + half_size, center_y - half_size],
            [center_x + half_size, center_y + half_size],
            [center_x - half_size, center_y + half_size]
        ], dtype=np.float32)

    # Order corners consistently
    board_corners = order_corners(board_corners)

    # Create mask
    board_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(board_mask, [board_corners.astype(np.int32)], 255)

    return board_corners, board_mask


def order_corners(corners: np.ndarray) -> np.ndarray:
    """Order corners as [top-left, top-right, bottom-right, bottom-left]."""
    # Sort by y-coordinate (top vs bottom)
    sorted_y = corners[np.argsort(corners[:, 1])]
    top_two = sorted_y[:2]
    bottom_two = sorted_y[2:]

    # Sort top two by x-coordinate (left vs right)
    top_left, top_right = top_two[np.argsort(top_two[:, 0])]

    # Sort bottom two by x-coordinate
    bottom_left, bottom_right = bottom_two[np.argsort(bottom_two[:, 0])]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def calculate_piece_bbox(
    square: chess.Square,
    board_corners: np.ndarray,
    image_shape: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    """
    Calculate normalized bounding box for a piece at a given square.

    Args:
        square: Chess square (0-63, where a1=0, h8=63)
        board_corners: 4x2 array of board corners [TL, TR, BR, BL]
        image_shape: (height, width) of image

    Returns:
        (x_center, y_center, width, height) normalized to [0, 1]
    """
    height, width = image_shape[:2]

    # Get rank and file (0-7)
    rank = chess.square_rank(square)  # 0 (rank 1) to 7 (rank 8)
    file = chess.square_file(square)  # 0 (a-file) to 7 (h-file)

    # Define 8x8 grid in perspective space
    # Use homography to map board square to image pixels

    # Board corners in board space (standard square)
    src_corners = np.array([
        [0, 0],      # top-left (a8 corner)
        [8, 0],      # top-right (h8 corner)
        [8, 8],      # bottom-right (h1 corner)
        [0, 8]       # bottom-left (a1 corner)
    ], dtype=np.float32)

    # Compute homography matrix
    H, _ = cv2.findHomography(src_corners, board_corners)

    # Square corners in board space (rank is flipped: rank 7 = visual top)
    # FEN rank 8 is at the top visually, rank 1 at bottom
    visual_rank = 7 - rank  # Flip so rank 7 (8th rank) maps to y=0
    square_corners = np.array([
        [file, visual_rank],          # top-left
        [file + 1, visual_rank],      # top-right
        [file + 1, visual_rank + 1],  # bottom-right
        [file, visual_rank + 1]       # bottom-left
    ], dtype=np.float32).reshape(-1, 1, 2)

    # Transform to image space
    square_corners_img = cv2.perspectiveTransform(square_corners, H)
    square_corners_img = square_corners_img.reshape(4, 2)

    # Calculate bounding box
    x_min = np.min(square_corners_img[:, 0])
    x_max = np.max(square_corners_img[:, 0])
    y_min = np.min(square_corners_img[:, 1])
    y_max = np.max(square_corners_img[:, 1])

    # Add padding for piece (pieces are taller than squares)
    square_height = y_max - y_min
    y_min -= square_height * 0.3  # Extend upward for tall pieces
    y_max += square_height * 0.05  # Small extension downward

    # Small horizontal padding
    square_width = x_max - x_min
    x_min -= square_width * 0.05
    x_max += square_width * 0.05

    # Clip to image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)

    # Convert to YOLO format (normalized x_center, y_center, width, height)
    x_center = (x_min + x_max) / 2 / width
    y_center = (y_min + y_max) / 2 / height
    bbox_width = (x_max - x_min) / width
    bbox_height = (y_max - y_min) / height

    return x_center, y_center, bbox_width, bbox_height


def fen_to_yolo_labels(
    fen: str,
    image_path: Path,
    debug: bool = False
) -> List[Tuple[int, float, float, float, float]]:
    """
    Convert FEN position to YOLO format labels.

    Args:
        fen: FEN string (e.g., "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        image_path: Path to corresponding image
        debug: If True, save debug visualization

    Returns:
        List of (class_id, x_center, y_center, width, height) tuples
    """
    # Parse FEN
    board = chess.Board(fen)

    # Load image and detect board
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    board_corners, board_mask = detect_board_corners(image)

    # Generate labels for each piece
    labels = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # Get piece symbol (e.g., 'P', 'n', 'K')
            piece_symbol = piece.symbol()
            class_id = FEN_TO_CLASS[piece_symbol]

            # Calculate bounding box
            x_center, y_center, width, height = calculate_piece_bbox(
                square, board_corners, image.shape
            )

            labels.append((class_id, x_center, y_center, width, height))

    # Debug visualization
    if debug:
        debug_image = image.copy()

        # Draw board corners
        for i in range(4):
            pt1 = tuple(board_corners[i].astype(int))
            pt2 = tuple(board_corners[(i + 1) % 4].astype(int))
            cv2.line(debug_image, pt1, pt2, (0, 255, 0), 2)

        # Draw bounding boxes
        h, w = image.shape[:2]
        for class_id, xc, yc, bw, bh in labels:
            x1 = int((xc - bw/2) * w)
            y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)

            # Color by piece type
            color = (255, 0, 0) if class_id < 6 else (0, 0, 255)  # Blue for white, red for black
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(debug_image, CLASS_NAMES[class_id], (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Save debug image
        debug_path = image_path.parent / 'debug_labels' / image_path.name
        debug_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(debug_path), debug_image)

    return labels


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    train_split: float = 0.9,
    seed: int = 42,
    debug_samples: int = 5
):
    """
    Convert Llava dataset to YOLO26 format.

    Args:
        input_dir: Directory containing annotations.json and images
        output_dir: Output directory for YOLO dataset
        train_split: Fraction of data for training (default 0.9)
        seed: Random seed for reproducibility
        debug_samples: Number of samples to visualize for debugging
    """
    random.seed(seed)

    # Load annotations
    annotations_path = input_dir / 'annotations.json'
    print(f"Loading annotations from {annotations_path}")
    with open(annotations_path) as f:
        annotations = json.load(f)

    print(f"Found {len(annotations)} annotated images")

    # Shuffle and split
    random.shuffle(annotations)
    split_idx = int(len(annotations) * train_split)
    train_data = annotations[:split_idx]
    val_data = annotations[split_idx:]

    print(f"Split: {len(train_data)} train, {len(val_data)} val")

    # Create output directories
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Process datasets
    for split_name, split_data in [('train', train_data), ('val', val_data)]:
        print(f"\nProcessing {split_name} set...")

        for i, item in enumerate(split_data):
            # Extract FEN from conversation
            fen = None
            for conv in item['conversations']:
                if conv['from'] == 'gpt':
                    fen = conv['value']
                    break

            if fen is None:
                print(f"Warning: No FEN found for {item['id']}")
                continue

            # Get image path
            image_path = Path(item['image'])
            if not image_path.is_absolute():
                image_path = input_dir / image_path.name

            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue

            # Generate YOLO labels
            debug = (i < debug_samples)  # Debug first N samples
            try:
                labels = fen_to_yolo_labels(fen, image_path, debug=debug)
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
                continue

            # Copy image
            output_image_path = output_dir / 'images' / split_name / image_path.name
            shutil.copy(image_path, output_image_path)

            # Write labels
            output_label_path = output_dir / 'labels' / split_name / f"{image_path.stem}.txt"
            with open(output_label_path, 'w') as f:
                for class_id, xc, yc, w, h in labels:
                    f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(split_data)} images")

        print(f"Completed {split_name}: {len(split_data)} images")

    # Create dataset YAML
    yaml_content = f"""# YOLO26 Chess Piece Detection Dataset
path: {output_dir.absolute()}
train: images/train
val: images/val

# Classes (12 piece types)
names:
  0: white_pawn
  1: white_knight
  2: white_bishop
  3: white_rook
  4: white_queen
  5: white_king
  6: black_pawn
  7: black_knight
  8: black_bishop
  9: black_rook
  10: black_queen
  11: black_king
"""

    yaml_path = output_dir / 'chess_dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\nDataset YAML saved to {yaml_path}")
    print(f"\nConversion complete!")
    print(f"  Train: {len(train_data)} images")
    print(f"  Val: {len(val_data)} images")
    print(f"  Debug visualizations saved to {output_dir / 'images' / 'train' / 'debug_labels'}")


def main():
    parser = argparse.ArgumentParser(description="Convert Llava chess dataset to YOLO26 format")
    parser.add_argument('--input', type=Path, default=Path('data/chess'),
                       help='Input directory with annotations.json and images')
    parser.add_argument('--output', type=Path, default=Path('data/yolo26_chess'),
                       help='Output directory for YOLO dataset')
    parser.add_argument('--train-split', type=float, default=0.9,
                       help='Fraction of data for training (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--debug-samples', type=int, default=5,
                       help='Number of samples to visualize for debugging')

    args = parser.parse_args()

    convert_dataset(
        input_dir=args.input,
        output_dir=args.output,
        train_split=args.train_split,
        seed=args.seed,
        debug_samples=args.debug_samples
    )


if __name__ == '__main__':
    main()
