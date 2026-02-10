#!/usr/bin/env python3
"""
Benchmark FEN Detection Methods

Compares different FEN detection approaches:
- YOLO26-DINO-MLP (this implementation)
- YOLO26-only (no DINO re-classification)
- Fenify-3D (if available)
- Ground truth (from dataset annotations)

Metrics:
- Per-piece classification accuracy
- Per-square accuracy
- Full board FEN accuracy
- Inference time
- Confidence scores

Usage:
    # Benchmark on validation set
    python scripts/benchmark_fen_detectors.py

    # Benchmark on specific directory
    python scripts/benchmark_fen_detectors.py --test-dir data/test_images

    # Compare specific models
    python scripts/benchmark_fen_detectors.py --yolo-weights runs/detect/yolo26_chess/weights/best.pt
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import cv2
import chess
from collections import defaultdict

try:
    from cosmos_chessbot.vision.yolo_dino_detector import YOLODINOFenDetector
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from cosmos_chessbot.vision.yolo_dino_detector import YOLODINOFenDetector


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
        Dict with per-square accuracy, piece accuracy, and exact match
    """
    pred_board = fen_to_board_array(pred_fen)
    gt_board = fen_to_board_array(gt_fen)

    # Per-square accuracy
    correct_squares = np.sum(pred_board == gt_board)
    total_squares = 64
    square_accuracy = correct_squares / total_squares

    # Per-piece accuracy (only occupied squares)
    gt_pieces = gt_board[gt_board != '.']
    pred_pieces = pred_board[gt_board != '.']  # Check predictions at GT piece locations

    if len(gt_pieces) > 0:
        piece_accuracy = np.sum(pred_pieces == gt_pieces) / len(gt_pieces)
    else:
        piece_accuracy = 1.0

    # Exact FEN match
    exact_match = (pred_fen == gt_fen)

    return {
        'square_accuracy': square_accuracy,
        'piece_accuracy': piece_accuracy,
        'exact_match': exact_match,
        'correct_squares': int(correct_squares),
        'total_pieces': len(gt_pieces)
    }


def benchmark_detector(
    detector: YOLODINOFenDetector,
    test_images: List[Path],
    ground_truth: Dict[str, str],
    detector_name: str = "Detector"
) -> Dict:
    """
    Benchmark a FEN detector on test images.

    Args:
        detector: FEN detector instance
        test_images: List of image paths
        ground_truth: Dict mapping image filename to FEN
        detector_name: Name for logging

    Returns:
        Dict with aggregate metrics
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {detector_name}")
    print(f"{'='*60}")

    results = []
    inference_times = []

    for image_path in test_images:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load {image_path}")
            continue

        # Get ground truth
        gt_fen = ground_truth.get(image_path.name)
        if gt_fen is None:
            print(f"Warning: No ground truth for {image_path.name}")
            continue

        # Inference
        start_time = time.time()
        try:
            pred_fen = detector.detect_fen(image, verbose=False)
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            continue
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        # Compare
        comparison = compare_fens(pred_fen, gt_fen)
        comparison['image'] = image_path.name
        comparison['pred_fen'] = pred_fen
        comparison['gt_fen'] = gt_fen
        comparison['inference_time'] = inference_time
        results.append(comparison)

        # Log progress
        if len(results) % 10 == 0:
            avg_accuracy = np.mean([r['square_accuracy'] for r in results])
            print(f"  Processed {len(results)}/{len(test_images)} - "
                  f"Avg square accuracy: {avg_accuracy:.2%}")

    # Aggregate metrics
    if not results:
        print("No results to aggregate!")
        return {}

    metrics = {
        'detector_name': detector_name,
        'num_images': len(results),
        'avg_square_accuracy': np.mean([r['square_accuracy'] for r in results]),
        'avg_piece_accuracy': np.mean([r['piece_accuracy'] for r in results]),
        'exact_match_rate': np.mean([r['exact_match'] for r in results]),
        'avg_inference_time': np.mean(inference_times),
        'median_inference_time': np.median(inference_times),
        'fps': 1.0 / np.mean(inference_times) if inference_times else 0,
        'total_correct_squares': sum(r['correct_squares'] for r in results),
        'total_squares': len(results) * 64,
    }

    # Print summary
    print(f"\n{detector_name} Results:")
    print(f"  Images processed: {metrics['num_images']}")
    print(f"  Square accuracy: {metrics['avg_square_accuracy']:.2%}")
    print(f"  Piece accuracy: {metrics['avg_piece_accuracy']:.2%}")
    print(f"  Exact match rate: {metrics['exact_match_rate']:.2%}")
    print(f"  Avg inference time: {metrics['avg_inference_time']*1000:.1f}ms")
    print(f"  Median inference time: {metrics['median_inference_time']*1000:.1f}ms")
    print(f"  FPS: {metrics['fps']:.1f}")

    return {
        'metrics': metrics,
        'per_image_results': results
    }


def load_ground_truth(dataset_dir: Path) -> Dict[str, str]:
    """
    Load ground truth FENs from dataset annotations.

    Args:
        dataset_dir: Path to dataset directory with annotations.json

    Returns:
        Dict mapping image filename to FEN string
    """
    annotations_path = dataset_dir / 'annotations.json'
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations not found: {annotations_path}")

    with open(annotations_path) as f:
        annotations = json.load(f)

    ground_truth = {}
    for item in annotations:
        # Extract FEN from conversation
        fen = None
        for conv in item['conversations']:
            if conv['from'] == 'gpt':
                fen = conv['value'].split()[0]  # Just position, not turn/castling
                break

        if fen:
            image_name = Path(item['image']).name
            ground_truth[image_name] = fen

    print(f"Loaded {len(ground_truth)} ground truth FENs")
    return ground_truth


def main():
    parser = argparse.ArgumentParser(description="Benchmark FEN detection methods")

    parser.add_argument(
        '--test-dir',
        type=Path,
        default=Path('data/chess_with_bboxes'),
        help='Directory with test images and annotations.json'
    )
    parser.add_argument(
        '--yolo-weights',
        type=Path,
        default=Path('runs/detect/yolo26_chess/weights/best.pt'),
        help='Path to YOLO weights'
    )
    parser.add_argument(
        '--mlp-weights',
        type=Path,
        default=Path('models/dino_mlp/dino_mlp_best.pth'),
        help='Path to DINO-MLP weights'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of test samples (default: 100, -1 for all)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        help='Device to use (mps, cuda, or cpu)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('benchmark_results.json'),
        help='Output file for detailed results'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FEN Detection Benchmark")
    print("=" * 60)

    # Load ground truth
    ground_truth = load_ground_truth(args.test_dir)

    # Get test images
    test_images = sorted(args.test_dir.glob('chess_*.jpg'))
    if args.num_samples > 0:
        test_images = test_images[:args.num_samples]

    print(f"Test images: {len(test_images)}")
    print()

    # Benchmark results
    all_results = {}

    # 1. YOLO26-DINO-MLP (full pipeline)
    if args.yolo_weights.exists() and args.mlp_weights.exists():
        detector_dino = YOLODINOFenDetector(
            yolo_weights=str(args.yolo_weights),
            mlp_weights=str(args.mlp_weights),
            device=args.device,
            use_dino=True
        )
        all_results['YOLO26-DINO-MLP'] = benchmark_detector(
            detector_dino,
            test_images,
            ground_truth,
            detector_name="YOLO26-DINO-MLP"
        )
    else:
        print(f"Skipping YOLO26-DINO-MLP: weights not found")

    # 2. YOLO26-only (no DINO)
    if args.yolo_weights.exists():
        detector_yolo = YOLODINOFenDetector(
            yolo_weights=str(args.yolo_weights),
            mlp_weights=None,
            device=args.device,
            use_dino=False
        )
        all_results['YOLO26-only'] = benchmark_detector(
            detector_yolo,
            test_images,
            ground_truth,
            detector_name="YOLO26-only"
        )
    else:
        print(f"Skipping YOLO26-only: weights not found")

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Method':<20} {'Square Acc':<12} {'Piece Acc':<12} {'Exact Match':<12} {'Speed (ms)'}")
    print("-" * 60)

    for method_name, result in all_results.items():
        metrics = result['metrics']
        print(f"{method_name:<20} "
              f"{metrics['avg_square_accuracy']:<12.2%} "
              f"{metrics['avg_piece_accuracy']:<12.2%} "
              f"{metrics['exact_match_rate']:<12.2%} "
              f"{metrics['avg_inference_time']*1000:.1f}")

    # Save detailed results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nDetailed results saved to {args.output}")


if __name__ == '__main__':
    main()
