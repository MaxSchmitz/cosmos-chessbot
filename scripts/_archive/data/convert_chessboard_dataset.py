#!/usr/bin/env python3
"""Convert synthetic chessboard dataset to Cosmos-Reason2 fine-tuning format."""

import json
import random
from pathlib import Path
from typing import Dict, List
import chess


def piece_name_to_fen(piece_name: str) -> str:
    """Convert piece name (e.g., 'knight_b') to FEN character (e.g., 'n').

    FEN notation:
    - Uppercase = White pieces (K Q R B N P)
    - Lowercase = Black pieces (k q r b n p)
    """
    piece_map = {
        "king_w": "K",
        "queen_w": "Q",
        "rook_w": "R",
        "bishop_w": "B",
        "knight_w": "N",
        "pawn_w": "P",
        "king_b": "k",
        "queen_b": "q",
        "rook_b": "r",
        "bishop_b": "b",
        "knight_b": "n",
        "pawn_b": "p",
    }
    return piece_map.get(piece_name, "?")


def config_to_fen(config: Dict[str, str]) -> str:
    """Convert board config to FEN notation.

    Args:
        config: Dictionary mapping cells (e.g., "A3") to pieces (e.g., "knight_b")

    Returns:
        FEN string (piece placement only)
    """
    # Create empty board (8x8)
    board = [[None for _ in range(8)] for _ in range(8)]

    # Place pieces on board
    for cell, piece_name in config.items():
        # Parse cell (e.g., "A3" -> file=0, rank=2)
        file = ord(cell[0]) - ord('A')  # A=0, B=1, ..., H=7
        rank = int(cell[1]) - 1  # 1=0, 2=1, ..., 8=7

        # Convert piece name to FEN character
        piece_char = piece_name_to_fen(piece_name)
        board[rank][file] = piece_char

    # Convert board to FEN (rank 8 to rank 1, left to right)
    fen_ranks = []
    for rank_idx in range(7, -1, -1):  # 7 down to 0 (rank 8 to rank 1)
        rank = board[rank_idx]
        fen_rank = ""
        empty_count = 0

        for piece in rank:
            if piece is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_rank += str(empty_count)
                    empty_count = 0
                fen_rank += piece

        if empty_count > 0:
            fen_rank += str(empty_count)

        fen_ranks.append(fen_rank)

    # Join ranks with '/'
    fen_position = "/".join(fen_ranks)

    # Add default metadata (active color, castling, en passant, halfmove, fullmove)
    # For synthetic random positions, we use defaults
    full_fen = f"{fen_position} w - - 0 1"

    return full_fen


def convert_dataset(
    data_dir: Path,
    output_file: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """Convert chessboard dataset to JSONL format for Cosmos fine-tuning.

    Args:
        data_dir: Directory containing image/json pairs
        output_file: Output JSONL file path
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
    """
    # Get all JSON files
    json_files = sorted(data_dir.glob("*.json"))

    # Filter out config.json
    json_files = [f for f in json_files if f.name != "config.json"]

    print(f"Found {len(json_files)} annotation files")

    # Convert each annotation to dataset entry
    dataset = []
    errors = 0

    for json_file in json_files:
        try:
            # Get corresponding image
            image_id = json_file.stem
            image_file = data_dir / f"{image_id}.jpg"

            if not image_file.exists():
                print(f"Warning: Image not found for {json_file.name}")
                errors += 1
                continue

            # Load annotation
            with open(json_file) as f:
                annotation = json.load(f)

            config = annotation.get("config", {})

            # Convert to FEN
            fen = config_to_fen(config)

            # Create conversation entry
            prompts = [
                "What is the FEN position of this chess board?",
                "Convert this chess board to FEN notation.",
                "Output the FEN string for this position.",
                "Analyze this board and return the FEN.",
                "What is the current position in FEN format?",
                "Describe this chess position in FEN notation.",
                "Generate the FEN representation of this board.",
            ]

            entry = {
                "image": str(image_file.absolute()),
                "conversations": [
                    {
                        "role": "user",
                        "content": random.choice(prompts)
                    },
                    {
                        "role": "assistant",
                        "content": fen
                    }
                ],
                "metadata": {
                    "corners": annotation.get("corners", []),
                    "source": "synthetic_chessboard_dataset"
                }
            }

            dataset.append(entry)

        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            errors += 1

    print(f"\nSuccessfully converted {len(dataset)} samples")
    if errors > 0:
        print(f"Encountered {errors} errors")

    # Shuffle dataset
    random.shuffle(dataset)

    # Split into train/val/test
    total = len(dataset)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": dataset[:train_end],
        "val": dataset[train_end:val_end],
        "test": dataset[val_end:],
    }

    # Save splits
    for split_name, split_data in splits.items():
        split_file = output_file.parent / f"chess_fen_{split_name}.jsonl"
        with open(split_file, 'w') as f:
            for entry in split_data:
                f.write(json.dumps(entry) + '\n')

        print(f"Saved {len(split_data)} {split_name} samples to {split_file}")

    return splits


def main():
    """Convert chessboard dataset to Cosmos fine-tuning format."""
    print("=" * 80)
    print("Chessboard Dataset Conversion for Cosmos-Reason2 Fine-tuning")
    print("=" * 80)
    print()

    # Paths
    data_dir = Path(__file__).parent.parent / "data" / "chessboards"
    output_dir = Path(__file__).parent.parent / "data"

    if not data_dir.exists():
        print(f"Error: Dataset directory not found: {data_dir}")
        return

    # Convert dataset
    splits = convert_dataset(
        data_dir=data_dir,
        output_file=output_dir / "chess_fen.jsonl",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )

    print()
    print("=" * 80)
    print("Dataset conversion complete!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  Train: {len(splits['train'])} samples (80%)")
    print(f"  Val:   {len(splits['val'])} samples (10%)")
    print(f"  Test:  {len(splits['test'])} samples (10%)")
    print(f"  Total: {sum(len(s) for s in splits.values())} samples")
    print()
    print("Dataset characteristics:")
    print("  - Images: 1280x1280 JPEG (synthetic, angled views)")
    print("  - Labels: FEN notation for each position")
    print("  - Format: JSONL with conversations")
    print()
    print("Next steps:")
    print("  1. Review sample images and FEN labels")
    print("  2. Set up H100 GPU training environment")
    print("  3. Adapt training script from Cosmos Cookbook")
    print("  4. Run fine-tuning: python scripts/finetune_cosmos_chess.py")
    print()
    print("Expected results:")
    print("  - Training time: 2-4 hours on H100")
    print("  - Accuracy: 70-90% (from 0% base model)")
    print("  - Cost: ~$5-10 for training")


if __name__ == "__main__":
    main()
