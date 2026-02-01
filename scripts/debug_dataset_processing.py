#!/usr/bin/env python3
"""Debug dataset processing to find issues."""

import json
import sys
import traceback
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


def test_single_sample():
    """Test processing a single sample to debug issues."""
    print("=" * 80)
    print("Debug: Testing Single Sample Processing")
    print("=" * 80)

    # Load first sample
    print("\n1. Loading sample...")
    with open("data/chess_fen_train.jsonl") as f:
        sample = json.loads(f.readline())

    print(f"   Image: {sample['image']}")
    print(f"   User: {sample['conversations'][0]['content'][:50]}...")
    print(f"   Assistant: {sample['conversations'][1]['content']}")

    # Load image
    print("\n2. Loading image...")
    image = Image.open(sample["image"]).convert("RGB")
    print(f"   ✅ Image loaded: {image.size}")

    # Load processor
    print("\n3. Loading processor...")
    processor = AutoProcessor.from_pretrained(
        "nvidia/Cosmos-Reason2-8B",
        trust_remote_code=True
    )
    print(f"   ✅ Processor loaded")

    # Test different conversation formats
    user_content = sample["conversations"][0]["content"]
    assistant_content = sample["conversations"][1]["content"]

    print("\n4. Testing conversation format...")

    # Format 1: Standard chat format
    try:
        print("\n   Format 1: Standard messages format")
        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_content}]},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_content}]},
        ]
        print(f"   Conversation: {conversation}")

        text = processor.apply_chat_template(conversation, tokenize=False)
        print(f"   ✅ Chat template applied")
        print(f"   Text (first 200 chars): {text[:200]}...")

        inputs = processor(
            text=text,
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        print(f"   ✅ Processor succeeded")
        print(f"   Input IDs shape: {inputs['input_ids'].shape}")
        print(f"   Pixel values shape: {inputs.get('pixel_values', 'N/A')}")

        # Flatten batch dimension
        inputs = {k: v.squeeze(0) if v.ndim > 1 else v for k, v in inputs.items()}
        print(f"   ✅ Batch flattened")
        print(f"   Input IDs shape after flatten: {inputs['input_ids'].shape}")

        # Set labels
        inputs["labels"] = inputs["input_ids"].clone()
        print(f"   ✅ Labels set")

        return inputs

    except Exception as e:
        print(f"   ❌ Format 1 failed: {e}")
        traceback.print_exc()

    # Format 2: Simple text only
    try:
        print("\n   Format 2: Simple text + image")
        text = f"User: {user_content}\n\nAssistant: {assistant_content}"
        print(f"   Text: {text[:100]}...")

        inputs = processor(
            text=text,
            images=[image],
            return_tensors="pt",
        )
        print(f"   ✅ Processor succeeded")
        return inputs

    except Exception as e:
        print(f"   ❌ Format 2 failed: {e}")
        traceback.print_exc()

    return None


def main():
    try:
        result = test_single_sample()

        if result is not None:
            print("\n" + "=" * 80)
            print("✅ SUCCESS: Sample processing works!")
            print("=" * 80)
            print("\nResult keys:", list(result.keys()))
            for key, value in result.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  {key}: {type(value)}")
            return 0
        else:
            print("\n" + "=" * 80)
            print("❌ FAILED: All formats failed")
            print("=" * 80)
            return 1

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ ERROR: {e}")
        print("=" * 80)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
