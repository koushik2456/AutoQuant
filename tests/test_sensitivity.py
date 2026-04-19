"""Smoke tests for sensitivity profiling."""

import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from autoquant.sensitivity import compute_sensitivity


def test_compute_sensitivity_gpt2_smoke() -> None:
    model_name = "gpt2"
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.train()
    scores = compute_sensitivity(
        model,
        tok,
        num_samples=2,
        device="cpu",
        progress_callback=None,
    )
    assert isinstance(scores, dict)
    assert len(scores) > 0
    for _k, v in scores.items():
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0
        assert not math.isnan(v)
