"""
Bit-Width Allocator
Solves the constrained optimization: maximize model accuracy (proxy: keep
high-sensitivity layers at high precision) subject to a total memory budget.

Greedy knapsack with:
  - 95% effective budget (headroom for non-quantized footprint estimation)
  - Sensitivity "cliff" lock: layers clearly below a score gap stay at INT4
  - INT4 → INT8 → INT16 tiers; INT16 only when sensitivity ≥ INT16_UPGRADE_THRESHOLD
"""

from collections import Counter
from typing import Dict, Tuple

import torch.nn as nn

from .config import INT16_UPGRADE_THRESHOLD
from .utils import is_quantizable_weight_module

# Supported bit-widths and their byte sizes per parameter
BIT_WIDTH_OPTIONS = [4, 8, 16]  # bits
BYTES_PER_BIT = {4: 0.5, 8: 1.0, 16: 2.0}  # bytes per parameter


def estimate_layer_size_bytes(module: nn.Module, bits: int) -> float:
    """Return memory footprint of a layer at a given bit-width in bytes."""
    param_count = sum(p.numel() for p in module.parameters())
    return param_count * BYTES_PER_BIT[bits]


def _sensitivity_cliff_threshold(scores: list[float]) -> float:
    """Score boundary below which layers are locked at INT4 (insensitive tail)."""
    sorted_scores = sorted(scores)
    if len(sorted_scores) < 2:
        return 0.0
    gaps = [
        sorted_scores[i + 1] - sorted_scores[i]
        for i in range(len(sorted_scores) - 1)
    ]
    median_gap = sorted(gaps)[len(gaps) // 2]
    cliff = next(
        (sorted_scores[i + 1] for i, g in enumerate(gaps) if g > 2 * median_gap),
        0.0,
    )
    return float(cliff)


def allocate_bits(
    model: nn.Module,
    sensitivity_scores: Dict[str, float],
    target_size_gb: float,
    min_bits: int = 4,
    max_bits: int = 16,
) -> Tuple[Dict[str, int], float]:
    """
    Greedy knapsack bit allocation with cliff guard, INT8-first path, and 5% budget headroom.

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
    effective_budget_bytes = target_size_gb * (1024**3) * 0.95

    layer_info: dict = {}
    for name, module in model.named_modules():
        if is_quantizable_weight_module(module) and name in sensitivity_scores:
            layer_info[name] = {
                "module": module,
                "sensitivity": float(sensitivity_scores[name]),
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

    score_list = [layer_info[n]["sensitivity"] for n in layer_info]
    cliff_threshold = _sensitivity_cliff_threshold(score_list)

    locked_int4 = {
        n for n, info in layer_info.items() if info["sensitivity"] < cliff_threshold
    }

    assignments = {name: min_bits for name in layer_info}
    current_bytes = non_linear_bytes + sum(
        info["size_at_bits"][min_bits] for info in layer_info.values()
    )

    sorted_names = sorted(
        layer_info.keys(), key=lambda n: layer_info[n]["sensitivity"], reverse=True
    )

    def _try_upgrade(name: str, target_bits: int) -> bool:
        nonlocal current_bytes
        info = layer_info[name]
        cur = assignments[name]
        if cur >= target_bits:
            return False
        current_size = info["size_at_bits"][cur]
        new_size = info["size_at_bits"][target_bits]
        delta = new_size - current_size
        if current_bytes + delta <= effective_budget_bytes:
            current_bytes += delta
            assignments[name] = target_bits
            return True
        return False

    # Phase A: INT4 → INT8 (repeat passes until budget exhausted)
    changed = True
    while changed:
        changed = False
        for name in sorted_names:
            if name in locked_int4:
                continue
            if assignments[name] != 4:
                continue
            if _try_upgrade(name, 8):
                changed = True

    # Phase B: INT8 → INT16 only for high-sensitivity layers
    changed = True
    while changed:
        changed = False
        for name in sorted_names:
            if assignments[name] != 8:
                continue
            if layer_info[name]["sensitivity"] < INT16_UPGRADE_THRESHOLD:
                continue
            if _try_upgrade(name, 16):
                changed = True

    final_assignments = {**assignments, **non_linear_layers}

    expected_gb = current_bytes / (1024**3)
    return final_assignments, expected_gb


def summarize_assignments(assignments: Dict[str, int]) -> Dict[int, int]:
    """Returns a count of how many layers are at each bit-width."""
    return dict(Counter(assignments.values()))
