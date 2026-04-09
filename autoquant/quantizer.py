"""
AutoQuantizer — Main orchestration class.

Usage:
    quantizer = AutoQuantizer("gpt2")
    sensitivity = quantizer.analyze_sensitivity(num_samples=100)
    config = quantizer.create_config(target_size_gb=0.5)
    quantizer.quantize("config.json", "output_dir/")
    report = quantizer.get_report()
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .allocator import allocate_bits, summarize_assignments
from .sensitivity import compute_sensitivity
from .utils import (
    as_linear_for_quantize,
    compute_model_size_gb,
    get_device,
    is_quantizable_weight_module,
)


class DynamicQuantizedLinear(nn.Module):
    """
    INT-backed linear layer that stores weights as quantized integers with
    a per-output-channel scale factor. During forward pass, weights are
    dequantized on the fly: weight_fp = weight_int * scale.
    """

    def __init__(self, original_linear: nn.Linear, bits: int):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.bits = bits

        weight_fp = original_linear.weight.data.float()

        max_val = weight_fp.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        num_levels = 2 ** (bits - 1) - 1
        scale = max_val / num_levels

        weight_q = (weight_fp / scale).round().clamp(-num_levels, num_levels).to(
            torch.int8
        )

        self.register_buffer("weight_q", weight_q)
        self.register_buffer("scale", scale.to(torch.float16))

        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.data.to(torch.float16))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_deq = self.weight_q.to(self.scale.dtype) * self.scale
        return F.linear(x, weight_deq, self.bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bits={self.bits}"


def _replace_layer(model: nn.Module, layer_name: str, new_module: nn.Module) -> None:
    """Navigate the module tree by dotted name and replace the target."""
    parts = layer_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


class AutoQuantizer:
    """
    High-level interface for AutoQuant mixed-precision quantization.

    Lifecycle:
        1. __init__ — load model + tokenizer
        2. analyze_sensitivity — compute per-layer Fisher scores
        3. create_config — solve bit allocation under budget
        4. quantize — apply quantization and save
        5. get_report — return compression statistics
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = get_device()

        print(f"📥 Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)

        self.original_size = compute_model_size_gb(self.model)
        print(f"   Original size: {self.original_size:.3f} GB")

        self.sensitivity_scores: Optional[Dict[str, float]] = None
        self.bit_assignments: Optional[Dict[str, int]] = None
        self.quantized_size: Optional[float] = None
        self.output_dir: Optional[str] = None

    def analyze_sensitivity(
        self,
        num_samples: int = 100,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, float]:
        """
        Compute Fisher Information sensitivity scores for all linear layers.

        ``progress_callback(current, total, label)`` is invoked after each
        calibration sample when provided (``label`` is ``"sensitivity"``).
        """
        print(f"🔬 Analyzing sensitivity ({num_samples} samples)...")

        def _cb(cur: int, tot: int, _phase: str) -> None:
            if progress_callback is not None:
                progress_callback(cur, tot, "sensitivity")

        self.sensitivity_scores = compute_sensitivity(
            self.model,
            self.tokenizer,
            num_samples,
            device=self.device,
            progress_callback=_cb if progress_callback is not None else None,
        )
        print(f"   Scored {len(self.sensitivity_scores)} layers")
        return self.sensitivity_scores

    def create_config(self, target_size_gb: float) -> dict:
        """
        Solve the bit-allocation optimization problem.
        """
        if self.sensitivity_scores is None:
            raise RuntimeError("Call analyze_sensitivity() first")

        print(f"🎯 Allocating bits for {target_size_gb:.2f} GB target...")

        self.bit_assignments, expected_size = allocate_bits(
            self.model, self.sensitivity_scores, target_size_gb
        )

        compression_ratio = (
            self.original_size / expected_size if expected_size > 0 else 1.0
        )

        config = {
            "model_name": self.model_name,
            "target_size_gb": target_size_gb,
            "original_size_gb": self.original_size,
            "expected_size_gb": expected_size,
            "compression_ratio": compression_ratio,
            "sensitivity_scores": self.sensitivity_scores,
            "bit_assignments": self.bit_assignments,
            "bit_distribution": summarize_assignments(self.bit_assignments),
        }

        dist = config["bit_distribution"]
        print(
            f"   Expected size: {expected_size:.3f} GB ({compression_ratio:.2f}x compression)"
        )
        print(f"   Bit distribution: {dist}")

        return {
            "config": config,
            "expected_size_gb": expected_size,
            "compression_ratio": compression_ratio,
        }

    def quantize(
        self,
        config_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        """
        Apply mixed-precision quantization and save to disk.

        ``progress_callback(current, total, layer_name)`` fires after each
        layer replaced (``total`` is the number of layers below 16-bit).
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        assignments_raw = config["bit_assignments"]
        assignments = {k: int(v) for k, v in assignments_raw.items()}

        os.makedirs(output_dir, exist_ok=True)

        n_quant = sum(1 for b in assignments.values() if b < 16)
        print(f"⚙️  Applying quantization to {n_quant} layers...")

        replaced = 0
        for name, bits in assignments.items():
            if bits >= 16:
                continue
            for module_name, module in list(self.model.named_modules()):
                if module_name == name and is_quantizable_weight_module(module):
                    lin = as_linear_for_quantize(module)
                    new_layer = DynamicQuantizedLinear(lin, bits)
                    _replace_layer(self.model, module_name, new_layer)
                    replaced += 1
                    if progress_callback is not None and n_quant > 0:
                        progress_callback(replaced, n_quant, name)
                    break

        print(f"   Replaced {replaced} layers with INT-backed quantized layers")

        weights_path = os.path.join(output_dir, "pytorch_model.bin")
        print(f"💾 Saving weights to {weights_path}...")
        torch.save(self.model.state_dict(), weights_path)

        self.tokenizer.save_pretrained(output_dir)

        quant_config = {
            "model_name": self.model_name,
            "bit_assignments": assignments,
            "bit_distribution": config.get("bit_distribution", {}),
            "original_size_gb": config.get("original_size_gb", self.original_size),
        }
        qpath = os.path.join(output_dir, "quantization_config.json")
        with open(qpath, "w", encoding="utf-8") as f:
            json.dump(quant_config, f, indent=2)

        self.quantized_size = compute_model_size_gb(self.model)
        self.output_dir = output_dir
        print(
            f"✅ Quantized model saved to '{output_dir}' ({self.quantized_size:.3f} GB)"
        )

    def get_report(self) -> dict:
        """Return a summary dict of quantization results."""
        if self.quantized_size is None:
            raise RuntimeError("Call quantize() first")

        compression = (
            self.original_size / self.quantized_size
            if self.quantized_size > 0
            else 1.0
        )
        space_saved = (1 - self.quantized_size / self.original_size) * 100

        return {
            "model_name": self.model_name,
            "original_size_gb": self.original_size,
            "quantized_size_gb": self.quantized_size,
            "compression_ratio": compression,
            "space_saved_percent": space_saved,
            "bit_distribution": summarize_assignments(self.bit_assignments or {}),
            "output_dir": self.output_dir,
        }

    def run(
        self,
        target_size_gb: float = 0.5,
        output_dir: str = "quantized_model",
        config_path: Optional[str] = None,
    ) -> dict:
        """
        End-to-end pipeline: sensitivity → config → quantize → report.

        Writes config JSON to ``config_path`` (default: ``config.json`` in cwd).
        """
        cfg_out = config_path or "config.json"
        self.analyze_sensitivity()
        result = self.create_config(target_size_gb)
        with open(cfg_out, "w", encoding="utf-8") as f:
            json.dump(result["config"], f, indent=2)
        self.quantize(cfg_out, output_dir)
        return self.get_report()
