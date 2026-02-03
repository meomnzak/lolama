import pytest

from lolama.model.config import LlamaConfig


def test_valid_config(tiny_config):
    assert tiny_config.d_model == 64
    assert tiny_config.num_heads == 2
    assert tiny_config.num_kv_heads == 2


def test_gqa_config():
    config = LlamaConfig(d_model=64, num_heads=4, num_kv_heads=2)
    assert config.num_kv_heads == 2


def test_num_kv_heads_defaults_to_num_heads():
    config = LlamaConfig(d_model=64, num_heads=4, num_kv_heads=None)
    assert config.num_kv_heads == 4


def test_d_model_not_divisible_by_num_heads():
    with pytest.raises(ValueError):
        LlamaConfig(d_model=65, num_heads=4)


def test_num_heads_not_divisible_by_num_kv_heads():
    with pytest.raises(ValueError):
        LlamaConfig(d_model=64, num_heads=4, num_kv_heads=3)
