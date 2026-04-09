"""
Bit-Width Allocator
Solves the constrained optimization: maximize model accuracy (proxy: keep
high-sensitivity layers at high precision) subject to a total memory budget.

Uses a greedy knapsack approach:
  1. Start all layers at the minimum bit-width (INT4)
  2. Sort layers by sensitivity score (descending)
  3. Upgrade layers from INT4 → INT8 → FP16 as long as budget allows
"""

from collections import Counter
from typing import Dict, Tuple

import torch.nn as nn

from .utils import is_quantizable_weight_module

# Supported bit-widths and their byte sizes per parameter
BIT_WIDTH_OPTIONS = [4, 8, 16]  # bits
BYTES_PER_BIT = {4: 0.5, 8: 1.0, 16: 2.0}  # bytes per parameter


def estimate_layer_size_bytes(module: nn.Module, bits: int) -> float:
    """Return memory footprint of a layer at a given bit-width in bytes."""
    param_count = sum(p.numel() for p in module.parameters())
    return param_count * BYTES_PER_BIT[bits]


def allocate_bits(
    model: nn.Module,
    sensitivity_scores: Dict[str, float],
    target_size_gb: float,
    min_bits: int = 4,
    max_bits: int = 16,
) -> Tuple[Dict[str, int], float]:
    """
    Greedy knapsack bit allocation.

    Args:
        model: The loaded model (used to measure layer sizes)
        sensitivity_scores: Dict[layer_name -> score in [0, 1]]
        target_size_gb: Maximum allowed model size in gigabytes
        min_bits: Minimum bit-width to assign (default 4)
        max_bits: Maximum bit-width (default 16 = FP16)

    Returns:
        (bit_assignments: Dict[layer_name -> bits], expected_size_gb: float)
    """
    _ = max_bits  # reserved for future (e.g. INT2); allocation uses BIT_WIDTH_OPTIONS
    target_bytes = target_size_gb * (1024**3)

    layer_info: dict = {}
    for name, module in model.named_modules():
        if is_quantizable_weight_module(module) and name in sensitivity_scores:
            layer_info[name] = {
                "module": module,
                "sensitivity": sensitivity_scores[name],
                "size_at_bits": {
                    b: estimate_layer_size_bytes(module, b) for b in BIT_WIDTH_OPTIONS
                },
            }

    non_linear_bytes = 0.0
    non_linear_layers: Dict[str, int] = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Embedding, nn.LayerNorm)):
            for _pname, param in module.named_parameters(recurse=False):
                non_linear_bytes += param.numel() * 2.0  # FP16 = 2 bytes
            non_linear_layers[name] = 16
        elif (
            hasattr(module, "weight")
            and not is_quantizable_weight_module(module)
            and name not in layer_info
        ):
            for _pname, param in module.named_parameters(recurse=False):
                non_linear_bytes += param.numel() * 2.0
            non_linear_layers[name] = 16

    assignments = {name: min_bits for name in layer_info}
    current_bytes = non_linear_bytes + sum(
        info["size_at_bits"][min_bits] for info in layer_info.values()
    )

    sorted_layers = sorted(
        layer_info.items(), key=lambda x: x[1]["sensitivity"], reverse=True
    )

    for bits_target in [8, 16]:
        for name, info in sorted_layers:
            if assignments[name] >= bits_target:
                continue

            current_size = info["size_at_bits"][assignments[name]]
            new_size = info["size_at_bits"][bits_target]
            delta = new_size - current_size

            if current_bytes + delta <= target_bytes:
                current_bytes += delta
                assignments[name] = bits_target

    final_assignments = {**assignments, **non_linear_layers}

    expected_gb = current_bytes / (1024**3)
    return final_assignments, expected_gb


def summarize_assignments(assignments: Dict[str, int]) -> Dict[int, int]:
    """Returns a count of how many layers are at each bit-width."""
    return dict(Counter(assignments.values()))
