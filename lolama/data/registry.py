"""Model registry for supported models."""

from __future__ import annotations

MODEL_REGISTRY: dict[str, dict[str, str | bool]] = {
    "tinyllama": {
        "hf_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "folder": "tinyllama-1.1b",
        "trust_remote_code": False,
    },
    "open_llama_3b": {
        "hf_name": "openlm-research/open_llama_3b_v2",
        "folder": "open-llama-3b",
        "trust_remote_code": False,
    },
    "open_llama_7b": {
        "hf_name": "openlm-research/open_llama_7b_v2",
        "folder": "open-llama-7b",
        "trust_remote_code": False,
    },
    "llama7b": {
        "hf_name": "meta-llama/Llama-2-7b-hf",
        "folder": "llama-7b",
        "trust_remote_code": False,
    },
}
