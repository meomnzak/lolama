import torch

from src.model.sampler import Sampler


class TestSampler:
    def test_greedy(self):
        sampler = Sampler(do_sample=False)
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
        token = sampler.sample(logits)
        assert token.item() == 1  # argmax

    def test_top_k(self):
        torch.manual_seed(42)
        sampler = Sampler(temperature=1.0, top_k=2, do_sample=True)
        logits = torch.tensor([[1.0, 5.0, 3.0, 0.5]])
        # Sample many times; should only ever pick index 1 or 2 (top-2 logits)
        tokens = set()
        for _ in range(100):
            t = sampler.sample(logits).item()
            tokens.add(t)
        assert tokens.issubset({1, 2})

    def test_top_p(self):
        torch.manual_seed(42)
        sampler = Sampler(temperature=1.0, top_p=0.5, do_sample=True)
        logits = torch.tensor([[0.0, 10.0, 1.0, 0.0]])
        # With top_p=0.5 and one dominant logit, should mostly pick index 1
        tokens = [sampler.sample(logits).item() for _ in range(50)]
        assert all(t == 1 for t in tokens)

    def test_repetition_penalty(self):
        logits = torch.tensor([[2.0, 3.0, 1.0]])
        input_ids = torch.tensor([[1]])  # penalize token 1
        original = logits.clone()
        Sampler.apply_repetition_penalty(logits, input_ids, penalty=2.0)
        assert logits[0, 1] < original[0, 1]

    def test_repetition_penalty_noop(self):
        logits = torch.tensor([[2.0, 3.0, 1.0]])
        input_ids = torch.tensor([[1]])
        original = logits.clone()
        Sampler.apply_repetition_penalty(logits, input_ids, penalty=1.0)
        assert torch.equal(logits, original)
