import pytest
import torch

from src.model.config import LlamaConfig
from src.model.llama import Llama
from src.utils.rope import precompute_rope_frequencies


@pytest.fixture
def tiny_config():
    return LlamaConfig(
        vocab_size=128,
        d_model=64,
        num_heads=2,
        num_kv_heads=2,
        num_layers=2,
        hidden_dim=172,
        max_seq_len=64,
    )


@pytest.fixture
def tiny_model(tiny_config):
    model = Llama(tiny_config)
    model.eval()
    return model


@pytest.fixture
def rope_freqs(tiny_config):
    head_dim = tiny_config.d_model // tiny_config.num_heads
    cos, sin = precompute_rope_frequencies(
        dim=head_dim,
        max_seq_len=tiny_config.max_seq_len,
        base=tiny_config.rope_base,
    )
    return cos, sin
