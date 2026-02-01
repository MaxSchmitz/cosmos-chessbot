#!/usr/bin/env python3
"""Evaluate fine-tuned model on test set."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


def normalize_fen(fen: str) -> str:
    """Normalize FEN string for comparison (extract position part only)."""
    # FEN format: "position active_color castling en_passant halfmove fullmove"
    # We only care about position for now
    parts = fen.strip().split()
    if len(parts) > 0:
        return parts[0]  # Position part
    return fen.strip()


def compute_metrics(predictions: List[str], references: List[str]) -> Dict:
    """Compute evaluation metrics."""
    assert len(predictions) == len(references)

    exact_matches = 0
    position_matches = 0

    for pred, ref in zip(predictions, references):
        # Exact match
        if pred.strip() == ref.strip():
            exact_matches += 1
            position_matches += 1
        # Position-only match
        elif normalize_fen(pred) == normalize_fen(ref):
            position_matches += 1

    total = len(predictions)
    return {
        "exact_accuracy": exact_matches / total if total > 0 else 0,
        "position_accuracy": position_matches / total if total > 0 else 0,
        "total_samples": total,
        "exact_matches": exact_matches,
        "position_matches": position_matches,
    }


def evaluate_model(model_path: Path, test_data_path: Path, max_samples: int = None):
    """Evaluate model on test set."""
    print("=" * 80)
    print("Evaluating Fine-tuned Cosmos-Reason2 Chess Model")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Test data: {test_data_path}")
    print()

    # Load test data
    print("Loading test data...")
    with open(test_data_path) as f:
        test_data = [json.loads(line) for line in f]

    if max_samples:
        test_data = test_data[:max_samples]

    print(f"Test samples: {len(test_data)}")

    # Load model
    print("\nLoading model...")
    base_model_name = "nvidia/Cosmos-Reason2-8B"

    processor = AutoProcessor.from_pretrained(
        str(model_path) if (model_path / "preprocessor_config.json").exists()
        else base_model_name,
        trust_remote_code=True
    )

    base_model = AutoModel.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, str(model_path))
    model = model.merge_and_unload()
    model.eval()

    print(f"Model loaded on: {model.device}")

    # Evaluate
    print("\nRunning evaluation...")
    predictions = []
    references = []
    errors = []

    for i, example in enumerate(tqdm(test_data, desc="Evaluating")):
        try:
            # Load image
            image = Image.open(example["image"]).convert("RGB")

            # Get reference FEN
            ref_fen = example["conversations"][1]["content"]
            references.append(ref_fen)

            # Prepare input
            user_prompt = example["conversations"][0]["content"]
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]

            text = processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = processor(text=text, images=[image], return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate
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

            # Extract response
            if "assistant" in generated_text.lower():
                response = generated_text.split("assistant")[-1].strip()
            else:
                response = generated_text

            predictions.append(response)

            # Log errors
            if normalize_fen(response) != normalize_fen(ref_fen):
                errors.append({
                    "index": i,
                    "image": str(example["image"]),
                    "predicted": response,
                    "reference": ref_fen,
                })

        except Exception as e:
            print(f"\nError on sample {i}: {e}")
            predictions.append("")
            errors.append({
                "index": i,
                "image": str(example["image"]),
                "error": str(e),
            })

    # Compute metrics
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    metrics = compute_metrics(predictions, references)

    print(f"Total samples: {metrics['total_samples']}")
    print(f"Exact matches: {metrics['exact_matches']} ({metrics['exact_accuracy']:.1%})")
    print(f"Position matches: {metrics['position_matches']} ({metrics['position_accuracy']:.1%})")

    # Show sample errors
    if errors:
        print(f"\nErrors: {len(errors)}")
        print("\nSample errors (first 5):")
        for error in errors[:5]:
            print(f"\n  Sample {error['index']}:")
            if "error" in error:
                print(f"    Error: {error['error']}")
            else:
                print(f"    Image: {error['image']}")
                print(f"    Predicted: {error['predicted']}")
                print(f"    Reference: {error['reference']}")

    # Save results
    results_file = model_path / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "metrics": metrics,
            "errors": errors[:10],  # Save first 10 errors
        }, f, indent=2)

    print(f"\nDetailed results saved to: {results_file}")
    print("=" * 80)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path("data/chess_fen_test.jsonl"),
        help="Path to test JSONL file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)",
    )

    args = parser.parse_args()

    if not args.model_path.exists():
        print(f"Error: Model path not found: {args.model_path}")
        return 1

    if not args.test_data.exists():
        print(f"Error: Test data not found: {args.test_data}")
        return 1

    metrics = evaluate_model(args.model_path, args.test_data, args.max_samples)

    # Exit with success if accuracy > 70%
    if metrics["position_accuracy"] >= 0.70:
        print("\n✅ Model achieved target accuracy (>70%)!")
        return 0
    else:
        print(f"\n⚠️  Model accuracy below target: {metrics['position_accuracy']:.1%} < 70%")
        return 1


if __name__ == "__main__":
    exit(main())
