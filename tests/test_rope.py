import torch

from lolama.utils.rope import precompute_rope_frequencies, apply_rope, rotate_half


class TestPrecomputeRopeFrequencies:
    def test_shapes(self):
        cos, sin = precompute_rope_frequencies(dim=32, max_seq_len=64)
        assert cos.shape == (64, 32)
        assert sin.shape == (64, 32)

    def test_values_in_range(self):
        cos, sin = precompute_rope_frequencies(dim=32, max_seq_len=64)
        assert cos.min() >= -1.0
        assert cos.max() <= 1.0
        assert sin.min() >= -1.0
        assert sin.max() <= 1.0


class TestApplyRope:
    def test_preserves_shape(self):
        cos, sin = precompute_rope_frequencies(dim=32, max_seq_len=64)
        x = torch.randn(2, 4, 8, 32)  # batch, heads, seq, head_dim
        out = apply_rope(x, cos[:8], sin[:8])
        assert out.shape == x.shape


class TestRotateHalf:
    def test_known_input(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = rotate_half(x)
        expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
        assert torch.allclose(out, expected)
