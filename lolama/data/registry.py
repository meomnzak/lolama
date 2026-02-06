"""Model registry for supported models."""

from __future__ import annotations

MODEL_REGISTRY: dict[str, dict[str, str | bool]] = {
    "tinyllama": {
        "hf_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "folder": "tinyllama-1.1b",
        "trust_remote_code": False,
        "description": "TinyLlama 1.1B Chat — small, fast, instruction-tuned",
        "params": "1.1B",
        "download_size": "2.2 GB",
    },
    "open_llama_3b": {
        "hf_name": "openlm-research/open_llama_3b_v2",
        "folder": "open-llama-3b",
        "trust_remote_code": False,
        "description": "OpenLLaMA 3B v2 — compact base model",
        "params": "3B",
        "download_size": "6.4 GB",
    },
    "open_llama_7b": {
        "hf_name": "openlm-research/open_llama_7b_v2",
        "folder": "open-llama-7b",
        "trust_remote_code": False,
        "description": "OpenLLaMA 7B v2 — full-size base model",
        "params": "7B",
        "download_size": "13.5 GB",
    },
    "llama7b": {
        "hf_name": "meta-llama/Llama-2-7b-hf",
        "folder": "llama-7b",
        "trust_remote_code": False,
        "description": "LLaMA 2 7B — Meta's base model (gated, requires access)",
        "params": "7B",
        "download_size": "13.5 GB",
    },
    # Vision-Language Models (VLMs)
    "llava-1.5-7b": {
        "hf_name": "llava-hf/llava-1.5-7b-hf",
        "folder": "llava-1.5-7b",
        "trust_remote_code": False,
        "model_type": "vlm",
        "description": "LLaVA 1.5 7B — vision-language model for image understanding",
        "params": "7B",
        "download_size": "14 GB",
    },
    "llava-med": {
        "hf_name": "chaoyinshe/llava-med-v1.5-mistral-7b-hf",
        "folder": "llava-med-7b",
        "trust_remote_code": False,
        "model_type": "vlm",
        "description": "LLaVA-Med 7B — medical/radiology vision-language model (HF-compatible)",
        "params": "7B",
        "download_size": "14 GB",
    },
}
