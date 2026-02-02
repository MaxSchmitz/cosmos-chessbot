#!/usr/bin/env python3
"""Convert VALUE dataset to Llava format for Cosmos-Reason2 training.

VALUE dataset structure:
- images/: Rendered chess board images
- data/fen_all.json: FEN annotations
- data/labels_all.json: Other labels

Llava format (used by Cosmos Cookbook):
[
    {
        "id": "unique_id",
        "image": "path/to/image.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\nWhat is the FEN position?"},
            {"from": "gpt", "value": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}
        ]
    },
    ...
]
"""

import json
import random
from pathlib import Path
from typing import List, Dict
import argparse


def load_value_dataset(value_root: Path) -> List[Dict]:
    """Load VALUE dataset annotations.

    Args:
        value_root: Root directory of VALUE dataset

    Returns:
        List of samples with image paths and FEN annotations
    """
    # Load FEN annotations
    fen_file = value_root / "data" / "fen_all.json"

    if not fen_file.exists():
        raise FileNotFoundError(f"FEN file not found: {fen_file}")

    print(f"Loading FEN annotations from {fen_file}...")
    with open(fen_file) as f:
        fen_data = json.load(f)

    # VALUE format: either dict with image_id -> FEN, or list
    # We'll handle both cases
    samples = []

    if isinstance(fen_data, dict):
        for image_id, fen in fen_data.items():
            image_path = value_root / "images" / f"{image_id}.jpg"
            if image_path.exists():
                samples.append({
                    "id": image_id,
                    "image": str(image_path.absolute()),
                    "fen": fen
                })
    elif isinstance(fen_data, list):
        for idx, entry in enumerate(fen_data):
            # Handle different possible formats
            if isinstance(entry, dict):
                image_id = entry.get("id", str(idx))
                fen = entry.get("fen", "")
            else:
                image_id = str(idx)
                fen = str(entry)

            # Try different image naming conventions
            for ext in [".jpg", ".png", ".jpeg"]:
                image_path = value_root / "images" / f"{image_id}{ext}"
                if image_path.exists():
                    samples.append({
                        "id": image_id,
                        "image": str(image_path.absolute()),
                        "fen": fen
                    })
                    break

    print(f"Found {len(samples)} samples with valid images")
    return samples


def convert_to_llava_format(samples: List[Dict], system_prompt: str = "") -> List[Dict]:
    """Convert VALUE samples to Llava conversation format.

    Args:
        samples: List of VALUE samples
        system_prompt: Optional system prompt

    Returns:
        List of samples in Llava format
    """
    llava_samples = []

    # Diverse prompts for variety
    prompts = [
        "What is the FEN position of this chess board?",
        "Convert this chess board to FEN notation.",
        "Output the FEN string for this position.",
        "Analyze this board and return the FEN.",
        "What is the current position in FEN format?",
        "Describe this chess position in FEN notation.",
        "Generate the FEN representation of this board.",
        "Provide the FEN encoding for this chess position.",
    ]

    for sample in samples:
        # Create conversation
        user_prompt = random.choice(prompts)

        conversation = [
            {
                "from": "human",
                "value": f"<image>\n{user_prompt}"
            },
            {
                "from": "gpt",
                "value": sample["fen"]
            }
        ]

        llava_sample = {
            "id": sample["id"],
            "image": sample["image"],
            "conversations": conversation
        }

        llava_samples.append(llava_sample)

    return llava_samples


def split_dataset(samples: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Split dataset into train/val/test sets.

    Args:
        samples: List of samples
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set

    Returns:
        Dictionary with train, val, test splits
    """
    # Shuffle
    random.shuffle(samples)

    total = len(samples)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert VALUE dataset to Llava format for Cosmos-Reason2 training"
    )
    parser.add_argument(
        "--value-root",
        type=Path,
        required=True,
        help="Root directory of VALUE dataset (contains images/ and data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/value_llava"),
        help="Output directory for Llava format files",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of data for training",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ratio of data for validation",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for testing)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("VALUE Dataset to Llava Format Conversion")
    print("=" * 80)
    print(f"VALUE root: {args.value_root}")
    print(f"Output dir: {args.output_dir}")
    print()

    # Load VALUE dataset
    samples = load_value_dataset(args.value_root)

    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"Using first {len(samples)} samples")

    # Convert to Llava format
    print("\nConverting to Llava format...")
    llava_samples = convert_to_llava_format(samples)

    # Split dataset
    print("\nSplitting dataset...")
    splits = split_dataset(llava_samples, args.train_ratio, args.val_ratio)

    # Save splits
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in splits.items():
        output_file = args.output_dir / f"chess_fen_{split_name}.json"

        with open(output_file, "w") as f:
            json.dump(split_data, f, indent=2)

        print(f"Saved {len(split_data)} {split_name} samples to {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("Conversion Complete!")
    print("=" * 80)
    print(f"Train: {len(splits['train'])} samples ({args.train_ratio:.0%})")
    print(f"Val:   {len(splits['val'])} samples ({args.val_ratio:.0%})")
    print(f"Test:  {len(splits['test'])} samples ({1-args.train_ratio-args.val_ratio:.0%})")
    print(f"Total: {sum(len(s) for s in splits.values())} samples")
    print()
    print("Next steps:")
    print("1. Verify sample quality:")
    print(f"   cat {args.output_dir}/chess_fen_train.json | head -30")
    print("2. Set up Cosmos-RL training environment")
    print("3. Adapt intelligent-transportation training script for chess")
    print()


if __name__ == "__main__":
    main()
