#!/usr/bin/env python3
"""Fine-tune Cosmos-Reason2 for chess FEN detection.

Adapted from Cosmos Cookbook intelligent transportation recipe:
https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason2/intelligent-transportation/post_training.html
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModel,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_jsonl_dataset(jsonl_path: Path) -> List[Dict]:
    """Load dataset from JSONL file."""
    data = []
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            data.append(entry)
    logger.info(f"Loaded {len(data)} samples from {jsonl_path}")
    return data


class VisionLanguageDataCollator:
    """Custom data collator that processes samples on-the-fly during training."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        """Process a batch of raw samples."""
        batch_images = []
        batch_texts = []

        for feature in features:
            # Load image
            image = Image.open(feature["image"]).convert("RGB")
            batch_images.append(image)

            # Create conversation
            user_content = feature["conversations"][0]["content"]
            assistant_content = feature["conversations"][1]["content"]

            conversation = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_content}]},
                {"role": "assistant", "content": [{"type": "text", "text": assistant_content}]},
            ]

            # Apply chat template
            text = self.processor.apply_chat_template(conversation, tokenize=False)
            batch_texts.append(text)

        # Process batch
        batch = self.processor(
            text=batch_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Set labels (same as input_ids for causal LM)
        batch["labels"] = batch["input_ids"].clone()

        return batch


def prepare_dataset(data: List[Dict]) -> Dataset:
    """Convert JSONL data to Hugging Face Dataset (no processing, just load metadata)."""
    # Create dataset from raw data - no image processing yet
    dataset = Dataset.from_list(data)
    logger.info(f"Created dataset with {len(dataset)} samples (will process during training)")
    return dataset


def setup_lora_model(model, lora_config: LoraConfig):
    """Apply LoRA to the model."""
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    # Simple loss tracking for now
    # TODO: Add FEN accuracy metric if needed
    return {}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Cosmos-Reason2 for chess FEN detection")
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("data/chess_fen_train.jsonl"),
        help="Path to training JSONL",
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=Path("data/chess_fen_val.jsonl"),
        help="Path to validation JSONL",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="nvidia/Cosmos-Reason2-8B",
        help="Base model name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/cosmos-chess-fen"),
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Warmup steps",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 training",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use BF16 training (default: True)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Cosmos-Reason2 Chess FEN Fine-tuning")
    logger.info("=" * 80)
    logger.info(f"Base model: {args.model_name}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Training data: {args.train_data}")
    logger.info(f"Validation data: {args.val_data}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    logger.info("=" * 80)

    # Load processor
    logger.info(f"\nLoading processor from {args.model_name}...")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    # Load datasets
    logger.info("\nLoading datasets...")
    train_data = load_jsonl_dataset(args.train_data)
    val_data = load_jsonl_dataset(args.val_data)

    logger.info("Preparing datasets...")
    train_dataset = prepare_dataset(train_data)
    val_dataset = prepare_dataset(val_data)

    # Create data collator for on-the-fly processing
    logger.info("Creating data collator...")
    data_collator = VisionLanguageDataCollator(processor)

    # Load model
    logger.info(f"\nLoading base model {args.model_name}...")
    model = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        device_map="auto",
        trust_remote_code=True,
    )

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Setup LoRA
    logger.info("\nSetting up LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        # Don't specify task_type for custom vision-language models
        # This avoids requiring prepare_inputs_for_generation
    )
    model = setup_lora_model(model, lora_config)

    # Training arguments
    logger.info("\nSetting up training...")
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=50,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        report_to="wandb" if args.wandb else "none",
        run_name="cosmos-chess-fen" if args.wandb else None,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    logger.info("\n" + "=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)
    trainer.train()

    # Save final model
    logger.info("\nSaving final model...")
    trainer.save_model(str(args.output_dir / "final"))
    processor.save_pretrained(str(args.output_dir / "final"))

    logger.info("\n" + "=" * 80)
    logger.info("Training complete!")
    logger.info(f"Model saved to {args.output_dir / 'final'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
