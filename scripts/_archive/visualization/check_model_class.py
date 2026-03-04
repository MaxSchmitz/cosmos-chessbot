#!/usr/bin/env python3
"""Check the correct model class for Cosmos-Reason2."""

from transformers import AutoConfig

config = AutoConfig.from_pretrained("nvidia/Cosmos-Reason2-8B", trust_remote_code=True)

print("Model config:")
print(f"  Model type: {config.model_type}")
print(f"  Architectures: {config.architectures}")

if hasattr(config, 'auto_map'):
    print(f"  Auto map: {config.auto_map}")

print("\nTry loading with AutoModelForCausalLM...")
from transformers import AutoModelForCausalLM
import torch

try:
    model = AutoModelForCausalLM.from_pretrained(
        "nvidia/Cosmos-Reason2-8B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"✅ Success! Model class: {type(model).__name__}")
    print(f"   Has lm_head: {hasattr(model, 'lm_head')}")
    print(f"   Has get_output_embeddings: {hasattr(model, 'get_output_embeddings')}")
except Exception as e:
    print(f"❌ Failed: {e}")
