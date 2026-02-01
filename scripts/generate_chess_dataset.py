#!/usr/bin/env python3
"""Generate chess board dataset for fine-tuning Cosmos-Reason2.

Note: For now, this creates a dataset structure using our existing test images.
For production, you'll want to generate or collect 100+ diverse images.
"""

import json
import random
from pathlib import Path
from typing import List, Dict
import shutil


def filename_to_fen(filename: str) -> str:
    """Extract FEN from filename."""
    fen = filename.replace(".png", "").replace(".jpg", "")
    fen = fen.replace(":", "/")
    return fen


def generate_dataset_from_existing() -> List[Dict]:
    """Create dataset from existing test images."""
    dataset = []

    # Get existing test images
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    test_images = sorted(data_dir.glob("*.png"))
    test_images = [img for img in test_images if ":" in img.stem]

    print(f"Found {len(test_images)} existing images with FEN labels")

    prompts = [
        "What is the FEN position of this chess board?",
        "Convert this chess board to FEN notation.",
        "Output the FEN string for this position.",
        "Analyze this board and return the FEN.",
        "What is the current position in FEN format?",
    ]

    for img_path in test_images:
        fen = filename_to_fen(img_path.stem)

        entry = {
            "image": str(img_path),
            "conversations": [
                {
                    "role": "user",
                    "content": random.choice(prompts)
                },
                {
                    "role": "assistant",
                    "content": fen
                }
            ]
        }

        dataset.append(entry)

    return dataset


def main():
    """Generate dataset structure for fine-tuning."""
    print("=" * 80)
    print("Chess Board Dataset for Cosmos-Reason2 Fine-tuning")
    print("=" * 80)
    print()

    print("NOTE: This creates a minimal dataset from existing images.")
    print("For production fine-tuning, you'll need 50-100+ diverse images.\n")

    # Generate dataset from existing images
    dataset = generate_dataset_from_existing()

    if not dataset:
        print("Error: No images found with FEN labels in data/raw/")
        print("\nTo generate synthetic data, you'll need to:")
        print("  1. Install system Cairo library: brew install cairo")
        print("  2. Use chess.svg to render positions")
        print("  3. Add perspective transforms for egocentric angles")
        return

    # For now, use all data as training (too small to split)
    data_dir = Path(__file__).parent.parent / "data"
    output_file = data_dir / "chess_fen_minimal.jsonl"

    with open(output_file, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')

    print(f"Saved {len(dataset)} samples to {output_file}")
    print()
    print("=" * 80)
    print("Dataset creation complete!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  Total: {len(dataset)} samples (MINIMAL - need 50-100+ for real training)")
    print()
    print("Dataset format:")
    print("  - JSONL with image paths and FEN conversations")
    print("  - Compatible with Cosmos-Reason2 fine-tuning recipe")
    print()
    print("Next steps:")
    print("  1. Collect/generate 50-100+ diverse chess board images")
    print("  2. Label with ground truth FEN positions")
    print("  3. Set up H100 GPU access")
    print("  4. Adapt training script from Cosmos Cookbook")
    print("  5. Run fine-tuning (2-4 hours on H100)")
    print()
    print("Recipe: https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason2/intelligent-transportation/post_training.html")


if __name__ == "__main__":
    main()
