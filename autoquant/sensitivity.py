"""
Sensitivity Profiler
Ranks each linear layer by how much quantization noise would hurt the model.
Uses Fisher Information (gradient^2 approximation) on calibration text samples.
"""

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import is_quantizable_weight_module

CALIBRATION_TEXT = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world of software engineering.",
    "Large language models require significant memory to run inference.",
    "Quantization reduces model size while preserving most of the accuracy.",
    "The transformer architecture relies on attention mechanisms.",
    "Machine learning systems improve with more data and careful engineering.",
    "Neural networks approximate functions by composing linear layers and nonlinearities.",
    "Efficient deployment of LLMs requires compression and hardware-aware optimization.",
]


def compute_sensitivity(
    model: nn.Module,
    tokenizer,
    num_samples: int = 100,
    device: str = "cpu",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, float]:
    """
    Compute per-layer sensitivity scores using Fisher Information approximation.

    For each nn.Linear layer, this accumulates the squared gradient of the loss
    w.r.t. the weight matrix across calibration samples. The mean squared gradient
    magnitude serves as a proxy for how sensitive that layer is to perturbation.

    Args:
        model: HuggingFace causal LM (already loaded, FP16 or FP32)
        tokenizer: Matching tokenizer
        num_samples: Number of calibration forward passes
        device: "cpu" or "cuda"

    Returns:
        Dict mapping layer name -> normalized sensitivity score [0.0, 1.0]
    """
    model.eval()
    model.to(device)

    sensitivity_accum: Dict[str, float] = {}

    for name, module in model.named_modules():
        if is_quantizable_weight_module(module):
            sensitivity_accum[name] = 0.0

    texts = (CALIBRATION_TEXT * ((num_samples // len(CALIBRATION_TEXT)) + 1))[
        :num_samples
    ]

    n_texts = len(texts)
    if progress_callback is None:
        text_iter = enumerate(tqdm(texts, desc="Computing sensitivity"))
    else:
        text_iter = enumerate(texts)

    for i, text in text_iter:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(device)

        model.zero_grad()

        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()

        for name, module in model.named_modules():
            if is_quantizable_weight_module(module) and module.weight.grad is not None:
                grad_sq = module.weight.grad.detach().pow(2).mean().item()
                sensitivity_accum[name] += grad_sq

        if progress_callback is not None:
            progress_callback(i + 1, n_texts, "sensitivity")

    max_val = max(sensitivity_accum.values()) if sensitivity_accum else 1.0
    if max_val == 0:
        max_val = 1.0

    normalized = {name: score / max_val for name, score in sensitivity_accum.items()}

    model.zero_grad()

    return normalized
