"""
Lightweight FP16 vs quantized loss comparison on a short calibration string (proxy for quality).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, cast

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from autoquant.quantizer import DynamicQuantizedLinear
from autoquant.utils import as_linear_for_quantize, is_quantizable_weight_module


def _replace_module_in_model(parent_module: nn.Module, target_name: str, new_module: nn.Module) -> None:
    parts = target_name.split(".")
    current = parent_module
    for p in parts[:-1]:
        current = getattr(current, p)
    setattr(current, parts[-1], new_module)


def _causal_lm_loss(model: nn.Module, input_ids: torch.Tensor) -> float:
    """Single forward cross-entropy (next-token prediction)."""
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)
        loss = getattr(out, "loss", None)
        if loss is None:
            return float("nan")
        return float(loss.detach().cpu().item())


def compare_fp16_vs_quantized(
    quantized_dir: str,
    calibration_text: str = "The future of artificial intelligence depends on efficient models.",
    max_length: int = 128,
) -> Dict[str, Any]:
    """
    Load base FP16 (CPU) and quantized checkpoint; report loss on the same token batch.

    This is a **proxy** for quality change — not full perplexity on a dataset.
    """
    out: Dict[str, Any] = {"ok": False, "quantized_dir": quantized_dir}
    config_path = os.path.join(quantized_dir, "quantization_config.json")
    if not os.path.isfile(config_path):
        out["error"] = "quantization_config.json not found"
        return out

    with open(config_path, "r", encoding="utf-8") as f:
        qconf = json.load(f)
    base_name = qconf.get("model_name")
    if not base_name:
        out["error"] = "model_name missing in config"
        return out

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(quantized_dir)
    if tok.pad_token is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    enc = tok(
        calibration_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        out["error"] = "Calibration text tokenizes to fewer than 2 tokens"
        return out

    # FP16 reference
    fp16_model = None
    loss_fp16 = float("nan")
    try:
        fp16_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device if device == "cpu" else None,
            low_cpu_mem_usage=True,
        )
        if device == "cuda":
            fp16_model = cast(nn.Module, fp16_model).to(torch.device(device))
        fp16_model.eval()
        loss_fp16 = _causal_lm_loss(fp16_model, input_ids)
    except Exception as e:
        out["error"] = f"FP16 reference load/forward failed: {e}"
        return out
    finally:
        try:
            if fp16_model is not None:
                del fp16_model
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Quantized
    q_model = None
    loss_q = float("nan")
    try:
        q_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        assignments = qconf.get("bit_assignments", {})
        for name, bits in assignments.items():
            bits = int(bits)
            for module_name, child in list(q_model.named_modules()):
                if module_name == name and is_quantizable_weight_module(child):
                    if bits < 16:
                        lin = as_linear_for_quantize(child)
                        new_layer = DynamicQuantizedLinear(lin, bits)
                        _replace_module_in_model(q_model, module_name, new_layer)
                    break
        ckpt = os.path.join(quantized_dir, "pytorch_model.bin")
        if not os.path.isfile(ckpt):
            out["error"] = f"Missing {ckpt}"
            return out
        try:
            state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(ckpt, map_location="cpu")
        q_model.load_state_dict(state_dict)
        cast(nn.Module, q_model).to(torch.device(device))
        q_model.eval()
        loss_q = _causal_lm_loss(q_model, input_ids)
    except Exception as e:
        out["error"] = f"Quantized forward failed: {e}"
        return out
    finally:
        try:
            if q_model is not None:
                del q_model
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    delta = loss_q - loss_fp16
    rel_pct = (delta / loss_fp16 * 100.0) if loss_fp16 and loss_fp16 > 0 else None

    out["ok"] = True
    out["loss_fp16"] = round(loss_fp16, 6)
    out["loss_quantized"] = round(loss_q, 6)
    out["loss_delta"] = round(delta, 6)
    out["relative_loss_increase_percent"] = (
        round(rel_pct, 3) if rel_pct is not None else None
    )
    out["calibration_chars"] = len(calibration_text)
    out["device"] = device
    # For Chart.js dashboards (proxy for “accuracy” — full eval needs a held-out set).
    out["chart_data"] = {
        "loss_bar": {
            "labels": ["FP16 reference", "Quantized"],
            "values": [round(loss_fp16, 6), round(loss_q, 6)],
        },
        "relative_loss_increase_percent": (
            round(rel_pct, 3) if rel_pct is not None else None
        ),
    }
    return out
