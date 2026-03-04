#!/usr/bin/env python3
"""Convert Llava JSON array format to JSONL format for training."""

import argparse
import json
from pathlib import Path


def convert_llava_to_jsonl(input_json: Path, output_jsonl: Path):
    """
    Convert Llava JSON array to JSONL format.

    Input format (Llava):
    [
      {"id": "...", "image": "path/to/img.jpg", "conversations": [...]},
      ...
    ]

    Output format (JSONL):
    {"image": "path/to/img.jpg", "conversations": [...]}
    {"image": "path/to/img.jpg", "conversations": [...]}
    ...
    """
    print(f"Loading {input_json}...")
    with open(input_json) as f:
        data = json.load(f)

    print(f"Converting {len(data)} samples to JSONL...")

    # Convert conversations format: "from"/"value" -> "role"/"content"
    converted = 0
    with open(output_jsonl, 'w') as f:
        for entry in data:
            # Transform conversation format
            conversations = []
            for msg in entry['conversations']:
                role = "user" if msg['from'] == 'human' else "assistant"
                content = msg['value']
                conversations.append({
                    "role": role,
                    "content": content
                })

            # Write JSONL entry
            jsonl_entry = {
                "image": entry['image'],
                "conversations": conversations
            }
            f.write(json.dumps(jsonl_entry) + '\n')
            converted += 1

    print(f"✓ Converted {converted} samples")
    print(f"✓ Saved to {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Llava JSON array to JSONL format"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input Llava JSON file (e.g., annotations.json)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file (e.g., chess_fen.jsonl)"
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist")
        return 1

    convert_llava_to_jsonl(args.input, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
