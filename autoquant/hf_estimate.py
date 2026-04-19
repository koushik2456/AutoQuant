"""
Estimate Hugging Face causal LM footprint without loading full weights (meta tensors).
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM


def estimate_fp16_footprint_gb(model_name: str, trust_remote_code: bool = True) -> Dict[str, Any]:
    """
    Load the model on the meta device and sum parameter + buffer bytes as FP16 would use.

    Returns keys: ok, original_size_gb, total_parameters, error (if not ok).
    """
    out: Dict[str, Any] = {"ok": False, "model_name": model_name}
    model = None
    try:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="meta",
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
            )
            # Heavier path: real CPU init — only used when meta is unavailable.
    except Exception as e:
        out["error"] = str(e)
        return out

    total_elements = 0
    for p in model.parameters():
        total_elements += int(p.numel())
    for b in model.buffers():
        total_elements += int(b.numel())

    # FP16 = 2 bytes per element for the primary weight footprint estimate
    bytes_fp16 = total_elements * 2
    gb = bytes_fp16 / (1024**3)
    del model

    out["ok"] = True
    out["original_size_gb"] = round(gb, 4)
    out["total_parameters"] = total_elements
    return out
