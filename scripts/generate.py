#!/usr/bin/env python3
"""
Text Generation
===============
Generate text using a loaded model.

Usage:
    python scripts/generate.py "The meaning of life is"
    python scripts/generate.py --stream "Tell me a joke"
    python scripts/generate.py --quantize "Hello"        # Quantize (int8 in memory, saves RAM, slower)
    python scripts/generate.py --quantize --fast "Hello" # Quantize + pre-dequantize (fp16 in memory, fast)
    python scripts/generate.py --model weights/tinyllama-1.1b "Hello"
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_model, load_tokenizer
from src.model import (
    quantize_model_int8,
    dequantize_model_for_inference,
    get_model_size_mb,
    save_quantized_model,
    load_quantized_model,
)


def get_quantized_path(model_path: str) -> Path:
    """Get the path where quantized model would be saved."""
    # Convert model name to a safe filename
    safe_name = model_path.replace("/", "_").replace("\\", "_")
    return PROJECT_ROOT / "weights" / f"{safe_name}-int8.pt"


def main() -> None:
    # Parse args
    model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    prompt = "The meaning of life is"
    stream = False
    quantize = False
    fast_mode = False
    
    args = sys.argv[1:]
    
    if "--stream" in args:
        stream = True
        args.remove("--stream")
    
    if "--quantize" in args:
        quantize = True
        args.remove("--quantize")
    
    if "--fast" in args:
        fast_mode = True
        args.remove("--fast")
    
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
    
    # Load model (with optional quantization)
    quantized_path = get_quantized_path(model_path)
    
    if quantize and quantized_path.exists():
        # Load existing quantized model
        print(f"Found saved quantized model: {quantized_path}")
        model = load_model(model_path, device="cpu")  # Load to CPU first
        quantize_model_int8(model, skip_layers=['lm_head', 'embed_tokens'])  # Convert architecture
        load_quantized_model(str(quantized_path), model)  # Load quantized weights
        model = model.to(device)
        
        if fast_mode:
            dequantize_model_for_inference(model)  # Cache fp16 weights for fast inference
        
        print(f"Model size: {get_model_size_mb(model):.1f} MB")
    elif quantize:
        # Quantize and save for next time
        model = load_model(model_path, device=device)
        size_before = get_model_size_mb(model)
        print(f"Quantizing model (before: {size_before:.1f} MB)...")
        quantize_model_int8(model, skip_layers=['lm_head', 'embed_tokens'])
        
        # Save for next time
        save_quantized_model(model, str(quantized_path))
        
        if fast_mode:
            # Pre-dequantize for fast inference
            dequantize_model_for_inference(model)
        
        print(f"Model size: {get_model_size_mb(model):.1f} MB")
        print()
    else:
        # Normal fp16 model
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
        
        # Track all generated tokens for proper decoding
        generated_tokens = []
        prev_text = ""
        
        for token_id in model.generate_stream(
            input_ids,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        ):
            generated_tokens.append(token_id)
            # Decode all tokens so far to get proper spacing
            full_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # Print only the new part
            new_text = full_text[len(prev_text):]
            print(new_text, end="", flush=True)
            prev_text = full_text
        
        print()
        print()
        print(f"Generated {len(generated_tokens)} tokens")
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
