import torch

from lolama.model.layers import RMSNorm, SwiGLU, LlamaAttention, LlamaBlock


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_dtype_preserved(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64, dtype=torch.float32)
        out = norm(x)
        assert out.dtype == torch.float32

    def test_weight_init_ones(self):
        norm = RMSNorm(64)
        assert torch.allclose(norm.weight, torch.ones(64))


class TestSwiGLU:
    def test_output_shape(self, tiny_config):
        ffn = SwiGLU(tiny_config.d_model, tiny_config.hidden_dim)
        x = torch.randn(2, 8, tiny_config.d_model)
        out = ffn(x)
        assert out.shape == (2, 8, tiny_config.d_model)


class TestLlamaAttention:
    def test_output_shape(self, tiny_config, rope_freqs):
        attn = LlamaAttention(tiny_config)
        cos, sin = rope_freqs
        x = torch.randn(2, 8, tiny_config.d_model)
        out = attn(x, cos[:8], sin[:8])
        assert out.shape == (2, 8, tiny_config.d_model)

    def test_with_kv_cache(self, tiny_config, rope_freqs):
        from lolama.model.kv_cache import KVCache

        attn = LlamaAttention(tiny_config)
        cos, sin = rope_freqs
        head_dim = tiny_config.d_model // tiny_config.num_heads
        cache = KVCache(1, tiny_config.max_seq_len, tiny_config.num_kv_heads, head_dim, torch.device("cpu"), torch.float32)

        # Prefill — layer slices cos/sin internally using offset=0
        x = torch.randn(1, 4, tiny_config.d_model)
        out = attn(x, cos, sin, kv_cache=cache)
        assert out.shape == (1, 4, tiny_config.d_model)
        assert cache.seq_len == 4

        # Decode step — layer slices cos/sin internally using offset=4
        x2 = torch.randn(1, 1, tiny_config.d_model)
        out2 = attn(x2, cos, sin, kv_cache=cache)
        assert out2.shape == (1, 1, tiny_config.d_model)
        assert cache.seq_len == 5


class TestLlamaBlock:
    def test_output_shape(self, tiny_config, rope_freqs):
        block = LlamaBlock(tiny_config)
        cos, sin = rope_freqs
        x = torch.randn(2, 8, tiny_config.d_model)
        out = block(x, cos[:8], sin[:8])
        assert out.shape == (2, 8, tiny_config.d_model)

    def test_residual_nonzero(self, tiny_config, rope_freqs):
        block = LlamaBlock(tiny_config)
        cos, sin = rope_freqs
        x = torch.randn(2, 8, tiny_config.d_model)
        out = block(x, cos[:8], sin[:8])
        assert not torch.allclose(out, torch.zeros_like(out))
