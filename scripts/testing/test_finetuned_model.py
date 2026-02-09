#!/usr/bin/env python3
"""Test fine-tuned Cosmos-Reason2 model on a single image."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoConfig, AutoModel, AutoProcessor


def test_model(model_path: Path, image_path: Path, prompt: str = None):
    """Test model on a single image."""
    print("=" * 80)
    print("Testing Fine-tuned Cosmos-Reason2 Chess Model")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print()

    # Load image
    print("Loading image...")
    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}")

    # Load processor and base model
    print("\nLoading model...")
    base_model_name = "nvidia/Cosmos-Reason2-8B"

    processor = AutoProcessor.from_pretrained(
        str(model_path) if (model_path / "preprocessor_config.json").exists()
        else base_model_name,
        trust_remote_code=True
    )

    # Get correct model architecture
    config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
    print(f"Loading {config.architectures[0]}...")

    # Import model class
    import importlib
    transformers_module = importlib.import_module("transformers")
    try:
        model_class = getattr(transformers_module, config.architectures[0])
    except AttributeError:
        model_class = AutoModel

    base_model = model_class.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, str(model_path))
    model = model.merge_and_unload()  # Merge LoRA weights
    model.eval()

    print(f"Model loaded on: {model.device}")

    # Prepare input
    if prompt is None:
        prompt = "What is the FEN position of this chess board?"

    print(f"\nPrompt: {prompt}")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    print("\nGenerating FEN...")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    # Decode
    generated_text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]

    # Extract assistant response
    if "assistant" in generated_text.lower():
        response = generated_text.split("assistant")[-1].strip()
    else:
        response = generated_text

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"Generated FEN: {response}")
    print("=" * 80)

    return response


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned model on an image")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to chess board image",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt (default: 'What is the FEN position of this chess board?')",
    )

    args = parser.parse_args()

    if not args.model_path.exists():
        print(f"Error: Model path not found: {args.model_path}")
        return 1

    if not args.image.exists():
        print(f"Error: Image not found: {args.image}")
        return 1

    test_model(args.model_path, args.image, args.prompt)
    return 0


if __name__ == "__main__":
    exit(main())
