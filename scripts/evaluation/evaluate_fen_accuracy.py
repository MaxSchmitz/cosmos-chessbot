#!/usr/bin/env python3
"""
Evaluate FEN Detection Accuracy on Test Set

Tests YOLO-DINO FEN detector on test images and generates visualizations
of failure cases for debugging.

Usage:
    # Evaluate on ChessReD2k test set
    python scripts/evaluate_fen_accuracy.py

    # Generate visualizations for worst cases
    python scripts/evaluate_fen_accuracy.py --visualize --num-viz 30
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import chess
from collections import defaultdict

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from cosmos_chessbot.vision.yolo_dino_detector import YOLODINOFenDetector


def load_chessred2k_annotations(annotations_path: Path) -> Dict[str, str]:
    """
    Load ground truth FENs from ChessReD2k COCO-format annotations.

    Args:
        annotations_path: Path to annotations.json

    Returns:
        Dict mapping image filename to FEN string
    """
    print(f"Loading annotations from {annotations_path}...")
    with open(annotations_path) as f:
        data = json.load(f)

    # Category ID to FEN piece symbol
    CATEGORY_TO_PIECE = {
        0: 'P', 1: 'R', 2: 'N', 3: 'B', 4: 'Q', 5: 'K',  # White
        6: 'p', 7: 'r', 8: 'n', 9: 'b', 10: 'q', 11: 'k',  # Black
        12: '.'  # Empty
    }

    # Build mapping: image_id -> list of pieces
    pieces_by_image = defaultdict(list)
    for ann in data['annotations']['pieces']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        square = ann['chessboard_position']

        if category_id in CATEGORY_TO_PIECE:
            piece_symbol = CATEGORY_TO_PIECE[category_id]
            if piece_symbol != '.':  # Skip empty squares
                pieces_by_image[image_id].append({
                    'square': square,
                    'piece': piece_symbol
                })

    # Convert pieces to FEN for each image
    fen_by_image_id = {}
    for image_id, pieces in pieces_by_image.items():
        # Build 8x8 board
        board_array = np.full((8, 8), '.', dtype=object)

        for piece_data in pieces:
            square_name = piece_data['square']
            piece = piece_data['piece']

            # Parse square (e.g., "a8" -> file=0, rank=7)
            file_idx = ord(square_name[0]) - ord('a')  # a=0, b=1, ..., h=7
            rank_idx = int(square_name[1]) - 1  # 1=0, 2=1, ..., 8=7

            # Place piece (FEN starts from rank 8, so invert)
            board_array[7 - rank_idx, file_idx] = piece

        # Convert board array to FEN
        fen_ranks = []
        for rank in range(8):  # Start from rank 8 (index 0)
            fen_rank = ""
            empty_count = 0

            for file in range(8):
                piece = board_array[rank, file]
                if piece == '.':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_rank += str(empty_count)
                        empty_count = 0
                    fen_rank += piece

            # Add remaining empty squares
            if empty_count > 0:
                fen_rank += str(empty_count)

            fen_ranks.append(fen_rank)

        # Join ranks with /
        fen_position = '/'.join(fen_ranks)
        fen_by_image_id[image_id] = fen_position

    # Build mapping: filename -> FEN
    fen_by_filename = {}
    for img in data['images']:
        img_id = img['id']
        if img_id in fen_by_image_id:
            fen_by_filename[img['file_name']] = fen_by_image_id[img_id]

    print(f"Loaded {len(fen_by_filename)} ground truth FENs")
    return fen_by_filename


def fen_to_board_array(fen: str) -> np.ndarray:
    """Convert FEN to 8x8 array of piece symbols."""
    board = chess.Board(fen)
    board_array = np.zeros((8, 8), dtype=object)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        board_array[7 - rank, file] = piece.symbol() if piece else '.'

    return board_array


def compare_fens(pred_fen: str, gt_fen: str) -> Dict:
    """
    Compare predicted FEN with ground truth.

    Returns:
        Dict with per-square accuracy and exact match
    """
    pred_board = fen_to_board_array(pred_fen)
    gt_board = fen_to_board_array(gt_fen)

    # Per-square accuracy
    correct_squares = np.sum(pred_board == gt_board)
    total_squares = 64
    square_accuracy = correct_squares / total_squares

    # Exact FEN match
    exact_match = (pred_fen == gt_fen)

    # Find mismatches for visualization
    mismatches = []
    for rank in range(8):
        for file in range(8):
            if pred_board[rank, file] != gt_board[rank, file]:
                square_name = chess.square_name(chess.square(file, 7 - rank))
                mismatches.append({
                    'square': square_name,
                    'predicted': str(pred_board[rank, file]),
                    'expected': str(gt_board[rank, file])
                })

    return {
        'square_accuracy': square_accuracy,
        'exact_match': exact_match,
        'correct_squares': int(correct_squares),
        'mismatches': mismatches
    }


def draw_fen_visualization(
    image: np.ndarray,
    corners: Optional[np.ndarray],
    pieces: List[Dict],
    pred_fen: str,
    gt_fen: str,
    accuracy: float,
    mismatches: List[Dict]
) -> np.ndarray:
    """Draw annotated visualization showing detections and errors."""
    vis = image.copy()
    h, w = vis.shape[:2]

    # --- Board outline & corners ---
    if corners is not None:
        pts = corners.astype(np.int32)
        order = [0, 1, 2, 3, 0]
        for i in range(len(order) - 1):
            cv2.line(vis, tuple(pts[order[i]]), tuple(pts[order[i + 1]]),
                    (255, 255, 255), 2)

        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]
        labels = ['TL', 'TR', 'BR', 'BL']
        for pt, col, lbl in zip(pts, colors, labels):
            cv2.circle(vis, tuple(pt), 8, col, -1)
            cv2.circle(vis, tuple(pt), 10, (255, 255, 255), 2)

    # --- Piece bounding boxes ---
    mismatch_squares = {m['square'] for m in mismatches}

    for det in pieces:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        square = det['square']
        piece = det['piece']
        conf = det['confidence']

        # Color code: green for correct, red for error
        if square in mismatch_squares:
            color = (0, 0, 255)  # Red for errors
            thickness = 3
        else:
            color = (0, 200, 255) if piece.isupper() else (100, 100, 255)
            thickness = 2

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        label = f"{det['class']} {conf:.2f} -> {square}"
        cv2.putText(vis, label, (x1, y1 - 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # --- Text overlay ---
    y_offset = 20
    text_color = (0, 255, 0) if accuracy == 1.0 else (0, 165, 255)

    cv2.putText(vis, f"Accuracy: {accuracy:.1%} ({len(mismatches)} errors)",
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

    y_offset += 25
    cv2.putText(vis, f"GT:   {gt_fen}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    y_offset += 20
    cv2.putText(vis, f"Pred: {pred_fen}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # List mismatches
    if mismatches:
        y_offset += 25
        cv2.putText(vis, "Errors:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        for i, mm in enumerate(mismatches[:5]):  # Show first 5 errors
            y_offset += 20
            error_text = f"  {mm['square']}: {mm['predicted']} != {mm['expected']}"
            cv2.putText(vis, error_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    return vis


def evaluate_detector(
    detector: YOLODINOFenDetector,
    test_images: List[Path],
    ground_truth: Dict[str, str],
    output_dir: Optional[Path] = None,
    visualize: bool = False,
    num_visualizations: int = 30
) -> Dict:
    """
    Evaluate FEN detector on test images.

    Args:
        detector: FEN detector instance
        test_images: List of image paths
        ground_truth: Dict mapping filename to FEN
        output_dir: Optional output directory for visualizations
        visualize: Whether to generate visualizations
        num_visualizations: Number of worst cases to visualize

    Returns:
        Dict with evaluation metrics and per-image results
    """
    print("\n" + "=" * 60)
    print("Evaluating FEN Detection")
    print("=" * 60)

    results = []

    for i, image_path in enumerate(test_images):
        # Load image
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"Warning: Could not load {image_path}")
            continue

        # Get ground truth
        gt_fen = ground_truth.get(image_path.name)
        if gt_fen is None:
            print(f"Warning: No ground truth for {image_path.name}")
            continue

        # Convert BGR to RGB for detector
        image_rgb = image_bgr[:, :, ::-1]

        # Run detection
        try:
            corners = detector._detect_corners(image_rgb)
            result = detector.detect_fen_with_metadata(image_rgb)
            pred_fen = result['fen']
            pieces = result['pieces']
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            continue

        # Compare
        comparison = compare_fens(pred_fen, gt_fen)

        results.append({
            'image': image_path.name,
            'image_path': str(image_path),
            'pred_fen': pred_fen,
            'gt_fen': gt_fen,
            'accuracy': comparison['square_accuracy'],
            'exact_match': comparison['exact_match'],
            'mismatches': comparison['mismatches'],
            'corners': corners.tolist() if corners is not None else None,
            'pieces': pieces
        })

        # Progress
        if (i + 1) % 20 == 0 or i == len(test_images) - 1:
            avg_acc = np.mean([r['accuracy'] for r in results])
            print(f"  [{i+1}/{len(test_images)}] Avg accuracy: {avg_acc:.2%}")

    # Aggregate metrics
    accuracies = [r['accuracy'] for r in results]
    exact_matches = sum(1 for r in results if r['exact_match'])

    metrics = {
        'num_images': len(results),
        'mean_accuracy': float(np.mean(accuracies)),
        'median_accuracy': float(np.median(accuracies)),
        'min_accuracy': float(np.min(accuracies)),
        'max_accuracy': float(np.max(accuracies)),
        'exact_match_rate': exact_matches / len(results),
        'exact_matches': exact_matches,
        'images_95plus': sum(1 for acc in accuracies if acc >= 0.95),
        'images_99plus': sum(1 for acc in accuracies if acc >= 0.99),
    }

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Images evaluated:    {metrics['num_images']}")
    print(f"Mean square accuracy: {metrics['mean_accuracy']:.4f} ({metrics['mean_accuracy']*100:.2f}%)")
    print(f"Exact FEN matches:    {exact_matches}/{len(results)} ({metrics['exact_match_rate']:.1%})")
    print(f"Min accuracy:         {metrics['min_accuracy']:.4f}")
    print(f"Max accuracy:         {metrics['max_accuracy']:.4f}")
    print(f"Median accuracy:      {metrics['median_accuracy']:.4f}")
    print(f"Images >= 95% acc:    {metrics['images_95plus']}/{len(results)} ({metrics['images_95plus']/len(results):.1%})")
    print(f"Images >= 99% acc:    {metrics['images_99plus']}/{len(results)} ({metrics['images_99plus']/len(results):.1%})")
    print("=" * 60)

    # Generate visualizations for worst cases
    if visualize and output_dir:
        print(f"\nGenerating visualizations for {num_visualizations} worst cases...")
        viz_dir = output_dir / 'failure_visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Sort by accuracy (worst first)
        sorted_results = sorted(results, key=lambda x: x['accuracy'])

        for i, result in enumerate(sorted_results[:num_visualizations]):
            image_bgr = cv2.imread(result['image_path'])
            image_rgb = image_bgr[:, :, ::-1]

            corners = np.array(result['corners']) if result['corners'] else None

            vis = draw_fen_visualization(
                image_bgr,
                corners,
                result['pieces'],
                result['pred_fen'],
                result['gt_fen'],
                result['accuracy'],
                result['mismatches']
            )

            acc_pct = int(result['accuracy'] * 100)
            out_path = viz_dir / f"{i+1:02d}_acc{acc_pct:02d}_{result['image']}"
            cv2.imwrite(str(out_path), vis)

        print(f"Failure visualizations saved to {viz_dir}")

        # Generate visualizations for best cases (perfect accuracy)
        print(f"\nGenerating visualizations for {num_visualizations} best cases (100% accuracy)...")
        success_dir = output_dir / 'success_visualizations'
        success_dir.mkdir(parents=True, exist_ok=True)

        # Get perfect accuracy cases
        perfect_results = [r for r in results if r['accuracy'] == 1.0]
        print(f"Found {len(perfect_results)} perfect detections")

        # Take first N perfect cases
        for i, result in enumerate(perfect_results[:num_visualizations]):
            image_bgr = cv2.imread(result['image_path'])
            image_rgb = image_bgr[:, :, ::-1]

            corners = np.array(result['corners']) if result['corners'] else None

            vis = draw_fen_visualization(
                image_bgr,
                corners,
                result['pieces'],
                result['pred_fen'],
                result['gt_fen'],
                result['accuracy'],
                result['mismatches']
            )

            out_path = success_dir / f"{i+1:02d}_perfect_{result['image']}"
            cv2.imwrite(str(out_path), vis)

        print(f"Success visualizations saved to {success_dir}")

    return {
        'metrics': metrics,
        'per_image_results': results
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate FEN detection accuracy")

    parser.add_argument(
        '--test-dir',
        type=Path,
        default=Path('data/chessred2k_pose/images/test'),
        help='Directory with test images'
    )
    parser.add_argument(
        '--annotations',
        type=Path,
        default=Path('data/chessred2k/annotations.json'),
        help='Path to annotations.json with ground truth FENs'
    )
    parser.add_argument(
        '--piece-weights',
        type=Path,
        default=Path('runs/detect/runs/detect/yolo26_chess/weights/best.pt'),
        help='Path to piece detection weights'
    )
    parser.add_argument(
        '--corner-weights',
        type=Path,
        default=Path('runs/pose/runs/pose/board_corners/weights/best.pt'),
        help='Path to corner detection weights'
    )
    parser.add_argument(
        '--corners',
        type=Path,
        default=None,
        help='Path to static calibrated corners JSON (optional)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.10,
        help='YOLO confidence threshold'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/fen_evaluation'),
        help='Output directory for results and visualizations'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations of failure cases'
    )
    parser.add_argument(
        '--num-viz',
        type=int,
        default=30,
        help='Number of worst cases to visualize (default: 30)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use (cpu, mps, cuda)'
    )

    args = parser.parse_args()

    # Validate paths
    if not args.test_dir.exists():
        print(f"Error: Test directory not found: {args.test_dir}")
        sys.exit(1)

    if not args.annotations.exists():
        print(f"Error: Annotations file not found: {args.annotations}")
        sys.exit(1)

    if not args.piece_weights.exists():
        print(f"Error: Piece weights not found: {args.piece_weights}")
        sys.exit(1)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FEN Detection Accuracy Evaluation")
    print("=" * 60)
    print(f"Test dir:       {args.test_dir}")
    print(f"Annotations:    {args.annotations}")
    print(f"Piece weights:  {args.piece_weights}")
    print(f"Corner weights: {args.corner_weights}")
    print(f"Device:         {args.device}")
    print(f"Output:         {args.output}")
    print("=" * 60)

    # Load ground truth
    ground_truth = load_chessred2k_annotations(args.annotations)

    # Get test images
    test_images = sorted(args.test_dir.glob('*.jpg'))
    # Filter to only images with ground truth
    test_images = [img for img in test_images if img.name in ground_truth]
    print(f"Found {len(test_images)} test images with ground truth")

    if not test_images:
        print("Error: No test images found")
        sys.exit(1)

    # Load detector
    use_static_corners = args.corners is not None and args.corners.exists()

    print("\nLoading FEN detector...")
    detector = YOLODINOFenDetector(
        yolo_weights=str(args.piece_weights),
        corner_weights=None if use_static_corners else str(args.corner_weights),
        mlp_weights=None,
        device=args.device,
        conf_threshold=args.conf,
        use_dino=False,
        static_corners=str(args.corners) if use_static_corners else None,
    )
    print("Detector loaded!")

    # Evaluate
    results = evaluate_detector(
        detector,
        test_images,
        ground_truth,
        output_dir=args.output,
        visualize=args.visualize,
        num_visualizations=args.num_viz
    )

    # Save results
    results_file = args.output / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
