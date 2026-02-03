#!/usr/bin/env python3
"""
Text Generation
===============
Generate text using a loaded model.

Usage:
    python scripts/generate.py "The meaning of life is"
    python scripts/generate.py --stream "Tell me a joke"
    python scripts/generate.py --model weights/tinyllama-1.1b "Hello"
"""

import sys
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_model, load_tokenizer


def main():
    # Parse args
    model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    prompt = "The meaning of life is"
    stream = False
    
    args = sys.argv[1:]
    
    if "--stream" in args:
        stream = True
        args.remove("--stream")
    
    if "--model" in args:
        idx = args.index("--model")
        model_path = args[idx + 1]
        args = args[:idx] + args[idx+2:]
    
    if args:
        prompt = " ".join(args)
    
    # Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print()
    
    # Load model
    model = load_model(model_path, device=device)
    
    # Load tokenizer
    tokenizer = load_tokenizer(model_path)
    
    # Prepare input
    print()
    print(f"Prompt: \"{prompt}\"")
    print("-" * 50)
    
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        if not isinstance(input_ids, torch.Tensor):
            input_ids = input_ids["input_ids"]
        input_ids = input_ids.to(device)
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    if stream:
        # Streaming generation - print tokens as they're generated
        print("Output: ", end="", flush=True)
        token_count = 0
        
        for token_id in model.generate_stream(
            input_ids,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        ):
            token_text = tokenizer.decode([token_id])
            print(token_text, end="", flush=True)
            token_count += 1
        
        print()
        print()
        print(f"Generated {token_count} tokens")
    else:
        # Batch generation
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Output: \"{output_text}\"")
        print()
        print(f"Generated {output_ids.shape[1] - input_ids.shape[1]} tokens")


if __name__ == "__main__":
    main()
