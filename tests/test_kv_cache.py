import torch

from lolama.model.kv_cache import KVCache, repeat_kv


class TestKVCache:
    def make_cache(self):
        return KVCache(
            batch_size=2, max_seq_len=32, num_kv_heads=2,
            head_dim=16, device=torch.device("cpu"), dtype=torch.float32,
        )

    def test_initial_seq_len(self):
        cache = self.make_cache()
        assert cache.seq_len == 0

    def test_update_increments_seq_len(self):
        cache = self.make_cache()
        k = torch.randn(2, 2, 5, 16)
        v = torch.randn(2, 2, 5, 16)
        cache.update(k, v)
        assert cache.seq_len == 5

    def test_update_returns_correct_shape(self):
        cache = self.make_cache()
        k = torch.randn(2, 2, 5, 16)
        v = torch.randn(2, 2, 5, 16)
        k_out, v_out = cache.update(k, v)
        assert k_out.shape == (2, 2, 5, 16)
        assert v_out.shape == (2, 2, 5, 16)

    def test_reset(self):
        cache = self.make_cache()
        k = torch.randn(2, 2, 5, 16)
        v = torch.randn(2, 2, 5, 16)
        cache.update(k, v)
        cache.reset()
        assert cache.seq_len == 0

    def test_multiple_updates(self):
        cache = self.make_cache()
        for length in [3, 4, 2]:
            k = torch.randn(2, 2, length, 16)
            v = torch.randn(2, 2, length, 16)
            cache.update(k, v)
        assert cache.seq_len == 9

        # Check returned shape on next update
        k = torch.randn(2, 2, 1, 16)
        v = torch.randn(2, 2, 1, 16)
        k_out, v_out = cache.update(k, v)
        assert k_out.shape == (2, 2, 10, 16)


class TestRepeatKV:
    def test_n_rep_1(self):
        x = torch.randn(2, 2, 8, 16)
        out = repeat_kv(x, 1)
        assert torch.equal(out, x)

    def test_n_rep_4(self):
        x = torch.randn(2, 2, 8, 16)
        out = repeat_kv(x, 4)
        assert out.shape == (2, 8, 8, 16)
