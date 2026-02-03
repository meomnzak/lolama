import torch


class TestLlamaModel:
    def test_output_shape(self, tiny_model, tiny_config):
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
        out = tiny_model(input_ids)
        assert out.shape == (1, 8, tiny_config.vocab_size)

    def test_batch(self, tiny_model, tiny_config):
        input_ids = torch.randint(0, tiny_config.vocab_size, (4, 8))
        out = tiny_model(input_ids)
        assert out.shape == (4, 8, tiny_config.vocab_size)

    def test_create_kv_caches(self, tiny_model, tiny_config):
        caches = tiny_model.create_kv_caches(batch_size=1)
        assert len(caches) == tiny_config.num_layers

    def test_count_parameters(self, tiny_model):
        params = tiny_model.count_parameters()
        assert "total" in params
        assert params["total"] > 0

    def test_forward_with_kv_cache(self, tiny_model, tiny_config):
        caches = tiny_model.create_kv_caches(batch_size=1)

        # Prefill
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 4))
        out1 = tiny_model(input_ids, kv_caches=caches)
        assert out1.shape == (1, 4, tiny_config.vocab_size)

        # Decode
        next_token = torch.randint(0, tiny_config.vocab_size, (1, 1))
        out2 = tiny_model(next_token, kv_caches=caches)
        assert out2.shape == (1, 1, tiny_config.vocab_size)
