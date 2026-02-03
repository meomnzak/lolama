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
WEIGHTS_DIR = PROJECT_ROOT / "weights"
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_model, load_tokenizer, resolve_model_source
from src.model import (
    quantize_model_int8,
    dequantize_model_for_inference,
    get_model_size_mb,
    save_quantized_model,
    load_quantized_model,
    is_quantized_model_dir,
)


def get_quantized_dir(model_path: str) -> Path:
    """Get the directory where quantized model would be saved."""
    # If it's a local path, use basename
    path = Path(model_path)
    if path.exists():
        name = path.name
    else:
        # HuggingFace model name - convert to safe dirname
        name = model_path.replace("/", "_").replace("\\", "_")
    
    return WEIGHTS_DIR / f"{name}-int8"


def get_source_dir(model_path: str) -> Path | None:
    """Get the source directory for copying tokenizer files."""
    source = resolve_model_source(model_path)
    if source["local_path"] is not None:
        return Path(source["local_path"])
    return None


def main() -> None:
    # Parse args
    model_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    prompt: str = "The meaning of life is"
    stream: bool = False
    quantize: bool = False
    fast_mode: bool = False
    
    args: list[str] = sys.argv[1:]
    
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
        idx: int = args.index("--model")
        model_path = args[idx + 1]
        args = args[:idx] + args[idx+2:]
    
    if args:
        prompt = " ".join(args)
    
    # Device
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print()
    
    # Load model (with optional quantization)
    quantized_dir: Path = get_quantized_dir(model_path)
    tokenizer_source: str = model_path  # Default: load tokenizer from original model
    
    if quantize and is_quantized_model_dir(str(quantized_dir)):
        # Load existing quantized model
        print(f"Found saved quantized model: {quantized_dir}/")
        model = load_model(model_path, device="cpu")  # Load to CPU first
        quantize_model_int8(model, skip_layers=['lm_head', 'embed_tokens'])  # Convert architecture
        load_quantized_model(str(quantized_dir), model)  # Load quantized weights
        model = model.to(device)
        
        if fast_mode:
            dequantize_model_for_inference(model)  # Cache fp16 weights for fast inference
        
        # Tokenizer is in the quantized directory now
        tokenizer_source = str(quantized_dir)
        print(f"Model size: {get_model_size_mb(model):.1f} MB")
    elif quantize:
        # Quantize and save for next time
        model = load_model(model_path, device=device)
        size_before: float = get_model_size_mb(model)
        print(f"Quantizing model (before: {size_before:.1f} MB)...")
        quantize_model_int8(model, skip_layers=['lm_head', 'embed_tokens'])
        
        # Save to directory (with tokenizer files)
        source_dir: Path | None = get_source_dir(model_path)
        print(f"\nSaving quantized model to {quantized_dir}/")
        save_quantized_model(model, str(quantized_dir), str(source_dir) if source_dir else None)
        
        if fast_mode:
            # Pre-dequantize for fast inference
            dequantize_model_for_inference(model)
        
        print(f"\nModel size: {get_model_size_mb(model):.1f} MB")
        print()
    else:
        # Normal fp16 model
        model = load_model(model_path, device=device)
    
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_source)
    
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
