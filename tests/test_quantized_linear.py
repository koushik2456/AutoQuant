"""Tests for DynamicQuantizedLinear."""

import torch
import torch.nn as nn

from autoquant.quantizer import DynamicQuantizedLinear


def test_dynamic_quantized_linear_grouped_forward() -> None:
    lin = nn.Linear(64, 32, bias=True)
    torch.manual_seed(0)
    lin.weight.data.normal_(0, 0.02)
    lin.bias.data.normal_(0, 0.02)
    q = DynamicQuantizedLinear(lin, bits=8, group_size=16)
    x = torch.randn(2, 64)
    y = q(x)
    assert y.shape == (2, 32)
    assert torch.isfinite(y).all()


def test_dynamic_quantized_linear_large_group_falls_back() -> None:
    lin = nn.Linear(32, 16, bias=False)
    torch.manual_seed(1)
    lin.weight.data.normal_(0, 0.02)
    q = DynamicQuantizedLinear(lin, bits=8, group_size=128)
    x = torch.randn(3, 32)
    y = q(x)
    assert y.shape == (3, 16)
    assert not bool(torch.isnan(y).any() or torch.isinf(y).any())
