"""
Lightweight FP16 vs quantized loss comparison on calibration strings (proxy for quality).
"""

from __future__ import annotations

import json
import math
import os
import statistics
from typing import Any, Dict, List, Optional, cast

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from autoquant.config import DEFAULT_GROUP_SIZE
from autoquant.quantizer import DynamicQuantizedLinear, infer_dynamic_quant_group_size
from autoquant.sensitivity import CALIBRATION_TEXT
from autoquant.utils import as_linear_for_quantize, is_quantizable_weight_module


def _replace_module_in_model(
    parent_module: nn.Module, target_name: str, new_module: nn.Module
) -> None:
    parts = target_name.split(".")
    current = parent_module
    for p in parts[:-1]:
        current = getattr(current, p)
    setattr(current, parts[-1], new_module)


def _causal_lm_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> float:
    """Single forward cross-entropy (next-token prediction)."""
    with torch.no_grad():
        labels = input_ids.clone()
        labels = labels.masked_fill(attention_mask.to(labels.device) == 0, -100)
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = getattr(out, "loss", None)
        if loss is None:
            return float("nan")
        return float(loss.detach().cpu().item())


def compare_fp16_vs_quantized(
    quantized_dir: str,
    calibration_text: str = "The future of artificial intelligence depends on efficient models.",
    max_length: int = 128,
    num_eval_strings: int = 5,
) -> Dict[str, Any]:
    """
    Load base FP16 (CPU) and quantized checkpoint; report loss statistics.

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

    if not qconf.get("schema_version"):
        print(
            "[WARN] quantization_config.json has no schema_version; "
            "checkpoint may be from an older AutoQuant build."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(quantized_dir)
    if tok.pad_token is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    n_eval = max(1, min(int(num_eval_strings), 32))
    base_strings: List[str] = [calibration_text]
    extra = (CALIBRATION_TEXT * ((n_eval // len(CALIBRATION_TEXT)) + 1))[: max(0, n_eval - 1)]
    eval_strings = base_strings + list(extra)

    def _encode(text: str) -> Optional[Dict[str, torch.Tensor]]:
        enc = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        ids = enc["input_ids"].to(device)
        if ids.shape[1] < 2:
            return None
        if "attention_mask" in enc and enc["attention_mask"] is not None:
            attn = enc["attention_mask"].to(device)
        else:
            attn = torch.ones_like(ids, dtype=torch.long, device=device)
        return {"input_ids": ids, "attention_mask": attn}

    input_batches: List[Dict[str, torch.Tensor]] = []
    per_meta: List[Dict[str, Any]] = []
    for s in eval_strings:
        batch = _encode(s)
        if batch is None:
            continue
        input_batches.append(batch)
        per_meta.append({"text": s})

    if not input_batches:
        out["error"] = "No calibration string tokenizes to at least 2 tokens"
        return out

    losses_fp16: List[float] = []
    losses_q: List[float] = []

    fp16_model = None
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
        for batch in input_batches:
            lf = _causal_lm_loss(
                fp16_model,
                batch["input_ids"],
                batch["attention_mask"],
            )
            if not math.isfinite(lf):
                out["error"] = "FP16 loss is non-finite for one calibration string"
                return out
            losses_fp16.append(lf)
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

    q_model = None
    repetition_rate = float("nan")
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    try:
        q_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        assignments = qconf.get("bit_assignments", {})
        ckpt = os.path.join(quantized_dir, "pytorch_model.bin")
        if not os.path.isfile(ckpt):
            out["error"] = f"Missing {ckpt}"
            return out
        try:
            state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(ckpt, map_location="cpu")

        cfg_gs = int(qconf.get("dynamic_quant_group_size", DEFAULT_GROUP_SIZE))

        for name, bits in assignments.items():
            bits = int(bits)
            for module_name, child in list(q_model.named_modules()):
                if module_name == name and is_quantizable_weight_module(child):
                    if bits < 16:
                        lin = as_linear_for_quantize(child)
                        gs = infer_dynamic_quant_group_size(
                            module_name, state_dict, lin.in_features, cfg_gs
                        )
                        new_layer = DynamicQuantizedLinear(lin, bits, group_size=gs)
                        _replace_module_in_model(q_model, module_name, new_layer)
                    break

        inc = q_model.load_state_dict(state_dict, strict=False)
        missing_keys = list(inc.missing_keys)
        unexpected_keys = list(inc.unexpected_keys)
        if missing_keys:
            print(f"[WARN] Missing keys in state_dict: {missing_keys[:10]}")
        if unexpected_keys:
            print(f"[WARN] Unexpected keys in state_dict: {unexpected_keys[:10]}")

        cast(nn.Module, q_model).to(torch.device(device))
        q_model.eval()

        for batch in input_batches:
            lq = _causal_lm_loss(
                q_model,
                batch["input_ids"],
                batch["attention_mask"],
            )
            if not math.isfinite(lq):
                out["error"] = "Quantized loss is non-finite for one calibration string"
                return out
            losses_q.append(lq)

        first = input_batches[0]
        inp = first["input_ids"]
        attn = first["attention_mask"]
        max_gen_len = int(inp.shape[1]) + 50
        with torch.no_grad():
            gen = cast(Any, q_model).generate(
                inp,
                attention_mask=attn,
                max_length=max_gen_len,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        tokens = gen[0].tolist()
        if len(tokens) >= 2:
            bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            repetition_rate = 1.0 - len(set(bigrams)) / max(len(bigrams), 1)
        else:
            repetition_rate = 0.0

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

    mean_fp16 = float(statistics.mean(losses_fp16))
    mean_q = float(statistics.mean(losses_q))
    std_fp16 = (
        float(statistics.pstdev(losses_fp16))
        if len(losses_fp16) > 1
        else 0.0
    )
    std_q = float(statistics.pstdev(losses_q)) if len(losses_q) > 1 else 0.0

    for i, row in enumerate(per_meta):
        row["loss_fp16"] = round(losses_fp16[i], 6)
        row["loss_quantized"] = round(losses_q[i], 6)

    delta = mean_q - mean_fp16
    rel_pct = (delta / mean_fp16 * 100.0) if mean_fp16 and mean_fp16 > 0 else None

    out["ok"] = True
    out["loss_fp16"] = round(mean_fp16, 6)
    out["loss_quantized"] = round(mean_q, 6)
    out["fp16_std"] = round(std_fp16, 6)
    out["quant_std"] = round(std_q, 6)
    out["per_string_losses"] = per_meta
    out["loss_delta"] = round(delta, 6)
    out["relative_loss_increase_percent"] = (
        round(rel_pct, 3) if rel_pct is not None else None
    )
    out["calibration_chars"] = len(calibration_text)
    out["device"] = device
    out["repetition_rate"] = (
        round(repetition_rate, 6) if math.isfinite(repetition_rate) else None
    )
    out["state_dict_warnings"] = {
        "missing_keys": missing_keys[:10],
        "unexpected_keys": unexpected_keys[:10],
    }
    out["chart_data"] = {
        "loss_bar": {
            "labels": ["FP16 reference", "Quantized"],
            "values": [round(mean_fp16, 6), round(mean_q, 6)],
            "error_plus": [round(std_fp16, 6), round(std_q, 6)],
            "error_minus": [round(std_fp16, 6), round(std_q, 6)],
        },
        "relative_loss_increase_percent": (
            round(rel_pct, 3) if rel_pct is not None else None
        ),
    }
    return out


def compute_quality_metrics(
    quantized_dir: str,
    calibration_text: str = "The future of artificial intelligence depends on efficient models.",
    max_length: int = 128,
    num_eval_strings: int = 5,
) -> Dict[str, Any]:
    """Alias for :func:`compare_fp16_vs_quantized` (multi-string metrics API)."""
    return compare_fp16_vs_quantized(
        quantized_dir,
        calibration_text=calibration_text,
        max_length=max_length,
        num_eval_strings=num_eval_strings,
    )
