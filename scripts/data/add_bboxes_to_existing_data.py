#!/usr/bin/env python3
"""
Add Bounding Boxes to Existing Dataset

Post-processes an existing Llava chess dataset to add bounding box annotations
by parsing FEN strings and estimating piece positions.

Usage:
    python scripts/add_bboxes_to_existing_data.py --input data/chess --output data/chess_with_bboxes
"""

import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import chess
    from PIL import Image
    import numpy as np
    import cv2
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip3 install python-chess pillow numpy opencv-python")
    exit(1)


# FEN piece to piece_type mapping
FEN_TO_PIECE_TYPE = {
    'P': 'pawn_w', 'N': 'knight_w', 'B': 'bishop_w',
    'R': 'rook_w', 'Q': 'queen_w', 'K': 'king_w',
    'p': 'pawn_b', 'n': 'knight_b', 'b': 'bishop_b',
    'r': 'rook_b', 'q': 'queen_b', 'k': 'king_b',
}


def estimate_board_region(image_path: Path) -> Tuple[float, float, float, float]:
    """
    Estimate the board region in an image as normalized coordinates.

    Returns:
        (x_min, y_min, x_max, y_max) normalized to [0, 1]
    """
    # Simple heuristic: assume board is centered and takes up 75% of image
    # For more accuracy, could use edge detection, but this is good enough
    margin = 0.125  # 12.5% margin on each side
    return (margin, margin, 1 - margin, 1 - margin)


def fen_to_bounding_boxes(
    fen: str,
    image_path: Path
) -> List[Dict]:
    """
    Convert FEN string to bounding box annotations.

    Args:
        fen: FEN string
        image_path: Path to image (used for board region estimation)

    Returns:
        List of bounding box dictionaries
    """
    board = chess.Board(fen)

    # Estimate board region
    board_x_min, board_y_min, board_x_max, board_y_max = estimate_board_region(image_path)
    board_width = board_x_max - board_x_min
    board_height = board_y_max - board_y_min

    bboxes = []

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        # Get rank and file (0-7)
        rank = chess.square_rank(square)  # 0 (rank 1) to 7 (rank 8)
        file = chess.square_file(square)  # 0 (a-file) to 7 (h-file)

        # Map to board region (rank 7 is top, rank 0 is bottom visually)
        visual_rank = 7 - rank
        square_width = board_width / 8
        square_height = board_height / 8

        # Calculate square center
        square_x = board_x_min + (file + 0.5) * square_width
        square_y = board_y_min + (visual_rank + 0.5) * square_height

        # Extend vertically for tall pieces (pieces are taller than squares)
        piece_width = square_width * 1.1  # 10% padding
        piece_height = square_height * 1.4  # 40% taller for piece height

        # Calculate bounding box
        x_center = square_x
        y_center = square_y - square_height * 0.15  # Shift up slightly
        width = piece_width
        height = piece_height

        # Clamp to [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = min(width, 1)
        height = min(height, 1)

        # Get square name and piece type
        square_name = chess.square_name(square)
        piece_type = FEN_TO_PIECE_TYPE[piece.symbol()]

        bboxes.append({
            "square": square_name,
            "piece_type": piece_type,
            "bbox": {
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "x_min": x_center - width / 2,
                "y_min": y_center - height / 2,
                "x_max": x_center + width / 2,
                "y_max": y_center + height / 2,
            }
        })

    return bboxes


def add_bboxes_to_dataset(
    input_dir: Path,
    output_dir: Path,
):
    """
    Add bounding boxes to an existing Llava dataset.

    Args:
        input_dir: Directory with annotations.json and images
        output_dir: Output directory for updated dataset
    """
    # Load annotations
    annotations_path = input_dir / 'annotations.json'
    print(f"Loading annotations from {annotations_path}")

    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    with open(annotations_path) as f:
        annotations = json.load(f)

    print(f"Found {len(annotations)} annotated images")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each annotation
    updated_annotations = []

    for i, item in enumerate(annotations):
        # Extract FEN
        fen = None
        for conv in item['conversations']:
            if conv['from'] == 'gpt':
                fen = conv['value']
                break

        if fen is None:
            print(f"Warning: No FEN found for {item['id']}")
            updated_annotations.append(item)
            continue

        # Get image path
        image_path = Path(item['image'])
        if not image_path.is_absolute():
            image_path = input_dir / image_path.name

        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            updated_annotations.append(item)
            continue

        # Generate bounding boxes
        try:
            bboxes = fen_to_bounding_boxes(fen, image_path)
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            updated_annotations.append(item)
            continue

        # Copy image to output directory
        output_image_path = output_dir / image_path.name
        if not output_image_path.exists():
            shutil.copy(image_path, output_image_path)

        # Update annotation with bounding boxes
        updated_item = item.copy()
        updated_item['image'] = f"data/{output_dir.name}/{output_image_path.name}"
        updated_item['bounding_boxes'] = bboxes

        updated_annotations.append(updated_item)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(annotations)} images")

    # Save updated annotations
    output_annotations_path = output_dir / 'annotations.json'
    with open(output_annotations_path, 'w') as f:
        json.dump(updated_annotations, f, indent=2)

    print(f"\n{'='*60}")
    print("Bounding box addition complete!")
    print(f"{'='*60}")
    print(f"Images: {output_dir}")
    print(f"Annotations: {output_annotations_path}")
    print(f"Total samples: {len(updated_annotations)}")
    print(f"\nNext step:")
    print(f"  python3 scripts/convert_llava_to_yolo26_v2.py --input {output_dir} --output data/yolo26_chess")


def main():
    parser = argparse.ArgumentParser(
        description="Add bounding boxes to existing Llava chess dataset"
    )
    parser.add_argument('--input', type=Path, default=Path('data/chess'),
                       help='Input directory with annotations.json and images')
    parser.add_argument('--output', type=Path, default=Path('data/chess_with_bboxes'),
                       help='Output directory for updated dataset')

    args = parser.parse_args()

    add_bboxes_to_dataset(
        input_dir=args.input,
        output_dir=args.output,
    )


if __name__ == '__main__':
    main()
