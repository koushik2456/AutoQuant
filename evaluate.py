"""
Quick evaluation of structurally quantized model.

CLI: python evaluate.py <folder>
Programmatic: run_quick_eval(path, prompt=..., max_new_tokens=...) -> dict
"""

import autoquant  # noqa: F401 — configure UTF-8 stdout on Windows before other prints

import json
import os
import sys
import time
from typing import Any, Dict, Optional, cast

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from autoquant.quantizer import DynamicQuantizedLinear
from autoquant.utils import as_linear_for_quantize, is_quantizable_weight_module


def _generate_kwargs(
    tokenizer: Any,
    max_new_tokens: int,
    do_sample: bool,
) -> Dict[str, Any]:
    """Stable defaults: greedy is the default — sampling uses mild temperature/top_p."""
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = eos
    kw: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": pad,
        "eos_token_id": eos,
        # Reduces degenerate repetition loops (common with damaged or over-quantized weights).
        "repetition_penalty": 1.15,
    }
    if do_sample:
        kw["temperature"] = 0.7
        kw["top_p"] = 0.92
    return kw


def _replace_module_in_model(parent_module, target_name, new_module):
    parts = target_name.split(".")
    current = parent_module
    for p in parts[:-1]:
        current = getattr(current, p)
    setattr(current, parts[-1], new_module)


def _load_quantized_causal_lm(model_path: str) -> Dict[str, Any]:
    """
    Build tokenizer + quantized model on disk. Returns dict with keys:
    ok, model, tokenizer, device, base_model_name, replaced_count, loaded_size_gb, error
    """
    out: Dict[str, Any] = {"ok": False, "model_path": model_path}
    config_path = os.path.join(model_path, "quantization_config.json")
    if not os.path.exists(config_path):
        out["error"] = (
            "quantization_config.json not found. Run quantization first; "
            "folder must contain AutoQuant output."
        )
        return out

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            quant_config = json.load(f)
        base_model_name = quant_config["model_name"]

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="cpu",
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

        assignments = quant_config.get("bit_assignments", {})
        replaced_count = 0
        for name, bits in assignments.items():
            bits = int(bits)
            for module_name, child in list(model.named_modules()):
                if module_name == name and is_quantizable_weight_module(child):
                    if bits < 16:
                        lin = as_linear_for_quantize(child)
                        new_layer = DynamicQuantizedLinear(lin, bits)
                        _replace_module_in_model(model, module_name, new_layer)
                        replaced_count += 1
                    break

        checkpoint_file = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.isfile(checkpoint_file):
            out["error"] = f"Missing {checkpoint_file}"
            return out

        try:
            state_dict = torch.load(
                checkpoint_file, map_location="cpu", weights_only=True
            )
        except TypeError:
            state_dict = torch.load(checkpoint_file, map_location="cpu")
        model.load_state_dict(state_dict)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cast(nn.Module, model).to(torch.device(device))
        model.eval()

        size_gb = _estimate_loaded_size_gb(model)

        out["ok"] = True
        out["model"] = model
        out["tokenizer"] = tokenizer
        out["device"] = device
        out["base_model_name"] = base_model_name
        out["replaced_quantized_layers"] = replaced_count
        out["loaded_size_gb"] = size_gb
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


def chat_quantized(
    model_path: str,
    user_message: str,
    max_new_tokens: int = 256,
    do_sample: bool = False,
) -> Dict[str, Any]:
    """
    Single-turn chat-style reply: returns only newly generated text (prompt stripped).
    """
    bundle = _load_quantized_causal_lm(model_path)
    if not bundle.get("ok"):
        return {"ok": False, "error": bundle.get("error", "load failed")}

    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    device = bundle["device"]

    prompt = (user_message or "").strip()
    if not prompt:
        return {"ok": False, "error": "message is empty"}

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[1])

    gen_kw = _generate_kwargs(
        tokenizer,
        max_new_tokens=max(1, min(int(max_new_tokens), 1024)),
        do_sample=do_sample,
    )
    t0 = time.time()
    with torch.no_grad():
        gen = cast(Any, model).generate(**inputs, **gen_kw)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    new_ids = gen[0, prompt_len:]
    reply = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    full = tokenizer.decode(gen[0], skip_special_tokens=True)

    try:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "ok": True,
        "reply": reply,
        "full_text": full,
        "elapsed_seconds": round(elapsed, 2),
        "base_model_name": bundle.get("base_model_name"),
        "device": device,
    }


def _estimate_loaded_size_gb(model: nn.Module) -> float:
    total_bytes = 0
    for _name, child in model.named_modules():
        if isinstance(child, DynamicQuantizedLinear):
            dq = cast(DynamicQuantizedLinear, child)
            total_bytes += dq.weight_q.numel() * dq.weight_q.element_size()
            total_bytes += dq.scale.numel() * dq.scale.element_size()
            if dq.bias is not None:
                total_bytes += dq.bias.numel() * dq.bias.element_size()
        elif isinstance(child, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            for p in child.parameters():
                total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024**3)


def run_quick_eval(
    model_path: str,
    prompt: str = "The future of AI is",
    max_new_tokens: int = 50,
    do_sample: bool = False,
) -> Dict[str, Any]:
    """
    Load quantized checkpoint and run one short generation.

    Returns a dict suitable for JSON (API) or printing (CLI).
    """
    out: Dict[str, Any] = {"ok": False, "model_path": model_path}
    config_path = os.path.join(model_path, "quantization_config.json")
    if not os.path.exists(config_path):
        out["error"] = (
            "quantization_config.json not found. Run quantization first; "
            "folder must contain AutoQuant output."
        )
        return out

    try:
        bundle = _load_quantized_causal_lm(model_path)
        if not bundle.get("ok"):
            out["error"] = bundle.get("error", "load failed")
            return out

        model = bundle["model"]
        tokenizer = bundle["tokenizer"]
        device = bundle["device"]
        base_model_name = bundle["base_model_name"]
        replaced_count = bundle["replaced_quantized_layers"]
        size_gb = bundle["loaded_size_gb"]

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kw = _generate_kwargs(
            tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        t0 = time.time()
        with torch.no_grad():
            gen = cast(Any, model).generate(**inputs, **gen_kw)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        generated = tokenizer.decode(gen[0], skip_special_tokens=True)

        out.update(
            {
                "ok": True,
                "base_model_name": base_model_name,
                "replaced_quantized_layers": replaced_count,
                "loaded_size_gb": round(size_gb, 4),
                "device": device,
                "prompt": prompt,
                "generated_text": generated,
                "elapsed_seconds": round(elapsed, 2),
                "max_new_tokens": max_new_tokens,
            }
        )
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


def evaluate(model_path: str) -> None:
    """CLI-friendly printing wrapper."""
    print("=" * 60)
    print("📊 Evaluating structurally quantized model")
    print("=" * 60)
    print()
    print("What this does:")
    print("  1) Reads quantization_config.json (base model name + bit plan).")
    print("  2) Loads the same HF architecture in FP16 on CPU, swaps in INT layers.")
    print("  3) Loads pytorch_model.bin weights + tokenizer from this folder.")
    print("  4) Moves to GPU if available, runs one short generate().")
    print()

    r = run_quick_eval(model_path)
    if not r.get("ok"):
        print("❌", r.get("error", "Unknown error"))
        return

    print(f"📥 Base model: {r['base_model_name']}")
    print(f"🔧 Quantized linear layers active: {r['replaced_quantized_layers']}")
    print(f"📦 Estimated loaded weight footprint: {r['loaded_size_gb']:.2f} GB")
    print(f"🖥️  Device used for generation: {r['device']}")
    print()
    print(f"📝 Prompt: {r['prompt']}")
    print(f"📝 Generated:\n{r['generated_text']}")
    print()
    print(f"⏱️  Generation time: {r['elapsed_seconds']:.2f}s")
    print()
    print("How to interpret:")
    print("  • If text is fluent, structural quantization is at least self-consistent.")
    print("  • For rigorous quality, compare perplexity vs FP16 on a fixed dataset.")
    print("=" * 60)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "quantized_model"
    evaluate(path)
