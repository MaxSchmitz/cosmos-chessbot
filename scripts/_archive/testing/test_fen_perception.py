#!/usr/bin/env python3
"""Test Cosmos-Reason2 FEN perception accuracy.

This script loads test images with ground-truth FEN in filenames,
passes them to Cosmos-Reason2, and compares predictions with ground truth.

Usage:
    # Test with local model
    python scripts/test_fen_perception.py

    # Test with remote server
    python scripts/test_fen_perception.py --cosmos-server http://192.241.168.72:8080
"""

import argparse
from pathlib import Path
from PIL import Image

from cosmos_chessbot.vision import CosmosPerception, RemoteCosmosPerception


def filename_to_fen(filename: str) -> str:
    """Extract FEN from filename.

    Args:
        filename: Filename with FEN notation (e.g., "rnbqkbnr:pppppppp:8:8:4P3:8:PPPP1PPP:RNBQKBNR b KQkq e3 0 1.png")

    Returns:
        FEN string (e.g., "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    """
    # Remove extension
    stem = Path(filename).stem

    # Replace colons with slashes (FEN uses / for rank separator, but we use : in filenames since / isn't allowed)
    fen_string = stem.replace(':', '/')

    return fen_string


def normalize_fen(fen: str) -> str:
    """Normalize FEN for comparison.

    Extracts just the board position (first part of FEN),
    removing turn, castling, en passant, etc.

    Args:
        fen: Full FEN string

    Returns:
        Normalized board position
    """
    # FEN format: "position turn castling en_passant halfmove fullmove"
    # We only care about position for this test
    return fen.split()[0] if ' ' in fen else fen


def compare_fens(ground_truth: str, predicted: str) -> dict:
    """Compare ground truth and predicted FEN.

    Args:
        ground_truth: Expected FEN
        predicted: Predicted FEN from model

    Returns:
        Dict with comparison results
    """
    gt_norm = normalize_fen(ground_truth)
    pred_norm = normalize_fen(predicted)

    exact_match = (gt_norm == pred_norm)

    # Count position differences
    gt_ranks = gt_norm.split('/')
    pred_ranks = pred_norm.split('/')

    rank_matches = 0
    square_matches = 0
    total_squares = 0

    for gt_rank, pred_rank in zip(gt_ranks, pred_ranks):
        if gt_rank == pred_rank:
            rank_matches += 1

        # Expand ranks for square-by-square comparison
        gt_squares = expand_rank(gt_rank)
        pred_squares = expand_rank(pred_rank)

        for gt_sq, pred_sq in zip(gt_squares, pred_squares):
            total_squares += 1
            if gt_sq == pred_sq:
                square_matches += 1

    return {
        'exact_match': exact_match,
        'rank_accuracy': rank_matches / 8.0,
        'square_accuracy': square_matches / total_squares if total_squares > 0 else 0.0,
        'ground_truth': gt_norm,
        'predicted': pred_norm,
    }


def expand_rank(rank_str: str) -> str:
    """Expand rank notation to 8 characters.

    Args:
        rank_str: Rank string (e.g., "1B1B3R" = "_B_B___R")

    Returns:
        8-character string with _ for empty squares
    """
    result = []
    for char in rank_str:
        if char.isdigit():
            # Empty squares
            result.extend(['_'] * int(char))
        else:
            # Piece
            result.append(char)
    return ''.join(result)


def main():
    parser = argparse.ArgumentParser(
        description="Test Cosmos-Reason2 FEN perception accuracy"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory with test images (default: data/raw)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Cosmos-Reason2-2B",
        help="Cosmos model to use (default: nvidia/Cosmos-Reason2-2B)",
    )
    parser.add_argument(
        "--cosmos-server",
        type=str,
        default=None,
        help="Remote Cosmos server URL (e.g., http://192.241.168.72:8080)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Cosmos-Reason2 FEN Perception Test")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Model: {args.model if not args.cosmos_server else f'Remote: {args.cosmos_server}'}")
    print()

    # Initialize perception
    if args.cosmos_server:
        print(f"Using remote Cosmos server at {args.cosmos_server}")
        perception = RemoteCosmosPerception(server_url=args.cosmos_server)
    else:
        print(f"Using local Cosmos model: {args.model}")
        perception = CosmosPerception(model_name=args.model)

    # Find all test images with FEN in filename
    test_images = []
    for pattern in ['*.jpeg', '*.jpg', '*.png']:
        for img_path in args.data_dir.glob(pattern):
            # Skip images without FEN notation (FEN uses colons in filename)
            if img_path.stem == 'frame' or ':' not in img_path.stem:
                continue
            test_images.append(img_path)

    if not test_images:
        print(f"ERROR: No test images found in {args.data_dir}")
        print("Expected images with FEN notation in filename (e.g., 'fen-notation.jpeg')")
        return 1

    print(f"Found {len(test_images)} test images")
    print()

    # Test each image
    results = []
    for i, img_path in enumerate(test_images, 1):
        print(f"[{i}/{len(test_images)}] Testing {img_path.name}")

        # Extract ground truth FEN
        ground_truth_fen = filename_to_fen(img_path.name)
        print(f"  Ground truth: {ground_truth_fen}")

        # Load image
        image = Image.open(img_path)

        # Get prediction
        board_state = perception.perceive(image)
        predicted_fen = board_state.fen

        print(f"  Predicted:    {normalize_fen(predicted_fen)}")
        print(f"  Confidence:   {board_state.confidence:.2%}")
        print(f"  Raw response: {board_state.raw_response[:200]}...")

        # Compare
        comparison = compare_fens(ground_truth_fen, predicted_fen)
        results.append(comparison)

        if comparison['exact_match']:
            print(f"  ✓ EXACT MATCH")
        else:
            print(f"  ✗ Mismatch")
            print(f"    Rank accuracy:   {comparison['rank_accuracy']:.1%}")
            print(f"    Square accuracy: {comparison['square_accuracy']:.1%}")

        if board_state.anomalies:
            print(f"  Anomalies: {board_state.anomalies}")

        print()

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    exact_matches = sum(1 for r in results if r['exact_match'])
    avg_rank_acc = sum(r['rank_accuracy'] for r in results) / len(results)
    avg_square_acc = sum(r['square_accuracy'] for r in results) / len(results)

    print(f"Total images:        {len(test_images)}")
    print(f"Exact matches:       {exact_matches}/{len(test_images)} ({exact_matches/len(test_images):.1%})")
    print(f"Avg rank accuracy:   {avg_rank_acc:.1%}")
    print(f"Avg square accuracy: {avg_square_acc:.1%}")
    print()

    if exact_matches == len(test_images):
        print("✓ All tests PASSED! Cosmos-Reason2 perception is working correctly.")
        return 0
    else:
        print(f"⚠ {len(test_images) - exact_matches} test(s) failed.")
        print("\nFailed images:")
        for img_path, result in zip(test_images, results):
            if not result['exact_match']:
                print(f"  {img_path.name}")
                print(f"    Expected: {result['ground_truth']}")
                print(f"    Got:      {result['predicted']}")
        return 1


if __name__ == "__main__":
    exit(main())
