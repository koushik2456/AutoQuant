"""Unit tests for bit allocation."""

import random

import torch.nn as nn

from autoquant.allocator import allocate_bits


class _TenLinears(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for i in range(10):
            self.add_module(f"l{i}", nn.Linear(32, 32, bias=False))


def test_allocate_bits_mock_model_random_scores() -> None:
    random.seed(0)
    model = _TenLinears()
    scores = {f"l{i}": random.random() for i in range(10)}
    assignments, expected_gb = allocate_bits(model, scores, target_size_gb=0.1)

    for _name, bits in assignments.items():
        assert bits in {4, 8, 16}

    assert expected_gb <= 0.1 * 1.05 + 1e-5
