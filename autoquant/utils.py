"""
Utility functions for size estimation and device detection.
"""

from typing import Any, Dict

import torch
import torch.nn as nn


def is_quantizable_weight_module(module: nn.Module) -> bool:
    """True for layers we treat as matmul-like (Linear or GPT-2 Conv1D)."""
    if isinstance(module, nn.Linear):
        return True
    try:
        from transformers.pytorch_utils import Conv1D
    except ImportError:
        return False
    return isinstance(module, Conv1D)


def as_linear_for_quantize(module: nn.Module) -> nn.Linear:
    """
    Return an nn.Linear with the same mapping as ``module`` (Linear or Conv1D).

    GPT-2 uses Conv1D with weight shape [in_features, out_features]; nn.Linear
    uses weight [out_features, in_features].
    """
    if isinstance(module, nn.Linear):
        return module
    from transformers.pytorch_utils import Conv1D

    if isinstance(module, Conv1D):
        w = module.weight.data
        in_f, out_f = w.shape[0], w.shape[1]
        lin = nn.Linear(in_f, out_f, bias=module.bias is not None).to(
            device=w.device, dtype=w.dtype
        )
        lin.weight.data.copy_(w.t())
        if module.bias is not None:
            lin.bias.data.copy_(module.bias.data)
        return lin
    raise TypeError(f"Unsupported module type: {type(module)}")


def compute_model_size_gb(model: nn.Module) -> float:
    """
    Compute the actual in-memory footprint of a model in gigabytes.
    Accounts for both parameters and registered buffers (important for
    DynamicQuantizedLinear which stores weights as buffers, not parameters).
    """
    total_bytes = 0

    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()

    for buf in model.buffers():
        total_bytes += buf.numel() * buf.element_size()

    return total_bytes / (1024**3)


def get_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def format_size(size_gb: float) -> str:
    """Human-readable size string."""
    if size_gb < 1.0:
        return f"{size_gb * 1024:.1f} MB"
    return f"{size_gb:.2f} GB"


def model_parameter_stats(model: nn.Module) -> Dict[str, Any]:
    """
    Logical parameter counts (quantization does not remove weights; it stores
    them in fewer bits). Used for UI: total vs quantizable weight tensors.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    q_layers = 0
    q_param_count = 0
    for _name, module in model.named_modules():
        if is_quantizable_weight_module(module):
            q_layers += 1
            q_param_count += sum(p.numel() for p in module.parameters())

    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "quantizable_layers": int(q_layers),
        "quantizable_parameter_count": int(q_param_count),
    }


def weighted_average_bits_for_quantizable(
    model: nn.Module, assignments: Dict[str, int]
) -> float:
    """Average bit-width weighted by parameter count (quantizable modules only)."""
    total_w = 0
    weighted = 0.0
    for name, module in model.named_modules():
        if not is_quantizable_weight_module(module):
            continue
        bits = int(assignments.get(name, 16))
        n = sum(p.numel() for p in module.parameters())
        total_w += n
        weighted += n * bits
    return (weighted / total_w) if total_w else 16.0


def gpu_info_dict() -> Dict[str, Any]:
    """
    Device summary for dashboards (prefers CUDA when available).

    Also explains common cases: CPU-only PyTorch wheel vs missing driver/GPU.
    """
    cuda_ok = torch.cuda.is_available()
    # None when this PyTorch was built as CPU-only (typical default pip install).
    cuda_in_build = getattr(torch.version, "cuda", None) is not None

    out: Dict[str, Any] = {
        "cuda_available": cuda_ok,
        "device": "cuda" if cuda_ok else "cpu",
        "device_name": None,
        "total_vram_gb": None,
        "pytorch_version": torch.__version__,
        "pytorch_built_with_cuda": cuda_in_build,
        "pytorch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_hint": "",
    }

    if cuda_ok:
        try:
            out["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            out["total_vram_gb"] = round(props.total_memory / (1024**3), 2)
            major, minor = torch.cuda.get_device_capability(0)
            out["compute_capability"] = f"{major}.{minor}"
            # Blackwell (RTX 50-series) is sm_90+ in API as (12, 0) → "12.0"
            if major >= 12:
                out["architecture_note"] = (
                    "Blackwell (sm_120) needs PyTorch CUDA 12.8+ wheels (cu128), not cu124."
                )
        except Exception:
            pass
        return out

    if not cuda_in_build:
        out["cuda_hint"] = (
            "This PyTorch build has no CUDA. Uninstall torch, then install a CUDA wheel "
            "from https://pytorch.org (choose your CUDA version), e.g. "
            "pip install torch --index-url https://download.pytorch.org/whl/cu124"
        )
    else:
        try:
            n = torch.cuda.device_count()
        except Exception:
            n = 0
        if n == 0:
            out["cuda_hint"] = (
                "PyTorch includes CUDA but no GPU was found. Update NVIDIA drivers, "
                "plug in a discrete GPU, or avoid Remote Desktop-only sessions that hide the GPU."
            )
        else:
            out["cuda_hint"] = (
                "CUDA is built in and devices exist but torch.cuda.is_available() is False. "
                "Check driver/CUDA compatibility or environment variables that disable CUDA."
            )

    return out
