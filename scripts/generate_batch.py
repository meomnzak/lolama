#!/usr/bin/env python3
"""
Batch Text Generation
=====================
Generate text for multiple prompts in parallel.

Usage:
    python scripts/generate_batch.py
"""

import sys
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_model, load_tokenizer


def main():
    # Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print()
    
    # Load model and tokenizer
    model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = load_model(model_path, device=device)
    tokenizer = load_tokenizer(model_path)
    
    # Multiple prompts of different lengths
    prompts = [
        "What is 2+2?",
        "Tell me a very short joke",
        "Hi",
    ]
    
    print("=" * 60)
    print("Batch Generation Test")
    print("=" * 60)
    print()
    
    # Tokenize prompts
    tokenized_prompts = []
    for prompt in prompts:
        if getattr(tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            if not isinstance(input_ids, torch.Tensor):
                input_ids = input_ids["input_ids"]
        else:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
        tokenized_prompts.append(input_ids.squeeze().to(device))
    
    print(f"Prompts ({len(prompts)}):")
    for i, (prompt, tokens) in enumerate(zip(prompts, tokenized_prompts)):
        print(f"  {i+1}. \"{prompt}\" ({len(tokens)} tokens)")
    print()
    
    # Generate in batch
    print("Generating...")
    results = model.generate_batch(
        tokenized_prompts,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or 0,
    )
    
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        output_text = tokenizer.decode(result, skip_special_tokens=True)
        print(f"\n[{i+1}] Prompt: \"{prompt}\"")
        print(f"    Output: {output_text}")
        print(f"    Generated {len(result) - len(tokenized_prompts[i])} tokens")


if __name__ == "__main__":
    main()
