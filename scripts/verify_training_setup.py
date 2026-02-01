#!/usr/bin/env python3
"""Verify training setup before starting full fine-tuning."""

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


def check_gpu():
    """Check GPU availability."""
    print("=" * 80)
    print("GPU Check")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False

    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print(f"✅ GPU count: {torch.cuda.device_count()}")
    print(f"✅ GPU name: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return True


def check_dataset():
    """Check dataset files."""
    print("\n" + "=" * 80)
    print("Dataset Check")
    print("=" * 80)

    data_dir = Path("data")
    required_files = [
        "chess_fen_train.jsonl",
        "chess_fen_val.jsonl",
        "chess_fen_test.jsonl",
    ]

    all_exist = True
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            # Count lines
            with open(filepath) as f:
                count = sum(1 for _ in f)
            print(f"✅ {filename}: {count} samples")

            # Check first entry
            with open(filepath) as f:
                first = json.loads(f.readline())
                image_path = Path(first["image"])
                if image_path.exists():
                    print(f"   Image exists: {image_path.name}")
                else:
                    print(f"   ⚠️  Image not found: {image_path}")
        else:
            print(f"❌ {filename} not found")
            all_exist = False

    return all_exist


def check_model():
    """Check model loading."""
    print("\n" + "=" * 80)
    print("Model Check")
    print("=" * 80)

    model_name = "nvidia/Cosmos-Reason2-8B"
    print(f"Loading {model_name}...")

    try:
        # Load processor
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print(f"✅ Processor loaded")

        # Load model (small test)
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"✅ Model loaded")
        print(f"   Model device: {model.device}")
        print(f"   Model dtype: {model.dtype}")

        # Clean up
        del model
        del processor
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def test_sample_processing():
    """Test processing a sample."""
    print("\n" + "=" * 80)
    print("Sample Processing Test")
    print("=" * 80)

    try:
        # Load first training sample
        with open("data/chess_fen_train.jsonl") as f:
            sample = json.loads(f.readline())

        print(f"Sample image: {sample['image']}")
        print(f"User prompt: {sample['conversations'][0]['content']}")
        print(f"FEN answer: {sample['conversations'][1]['content']}")

        # Load image
        image = Image.open(sample["image"]).convert("RGB")
        print(f"✅ Image loaded: {image.size}")

        # Load processor
        processor = AutoProcessor.from_pretrained(
            "nvidia/Cosmos-Reason2-8B",
            trust_remote_code=True
        )

        # Create conversation
        user_content = sample["conversations"][0]["content"]
        assistant_content = sample["conversations"][1]["content"]

        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_content}]},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_content}]},
        ]

        # Process
        text = processor.apply_chat_template(conversation, tokenize=False)
        inputs = processor(
            text=text,
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        print(f"✅ Sample processed successfully")
        print(f"   Input IDs shape: {inputs['input_ids'].shape}")
        print(f"   Pixel values shape: {inputs['pixel_values'].shape}")

        return True

    except Exception as e:
        print(f"❌ Sample processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("\n" + "=" * 80)
    print("Cosmos-Reason2 Chess FEN Training Setup Verification")
    print("=" * 80)

    checks = [
        ("GPU", check_gpu),
        ("Dataset", check_dataset),
        ("Model", check_model),
        ("Sample Processing", test_sample_processing),
    ]

    results = {}
    for name, check_fn in checks:
        results[name] = check_fn()

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All checks passed! Ready to train.")
        print("\nStart training with:")
        print("  uv run python scripts/train_cosmos_chess.py")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix issues before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
