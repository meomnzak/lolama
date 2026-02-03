"""Model registry for supported models."""

from __future__ import annotations

MODEL_REGISTRY: dict[str, dict[str, str | bool]] = {
    "tinyllama": {
        "hf_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "folder": "tinyllama-1.1b",
        "trust_remote_code": False,
    },
    "phi2": {
        "hf_name": "microsoft/phi-2",
        "folder": "phi-2",
        "trust_remote_code": True,
    },
    "llama7b": {
        "hf_name": "meta-llama/Llama-2-7b-hf",
        "folder": "llama-7b",
        "trust_remote_code": False,
    },
}
