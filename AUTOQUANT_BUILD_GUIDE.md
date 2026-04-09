# AutoQuant — Complete Project Build Guide
### For Cursor IDE Agent + Anaconda + Jupyter Notebooks

---

## 📌 Project Overview

**AutoQuant** is a Python framework that treats LLM quantization as an **optimization problem** rather than a manual tuning exercise. Instead of uniformly applying 4-bit or 8-bit precision to every layer (which either wastes compression opportunities or destroys accuracy), AutoQuant:

1. **Profiles layer sensitivity** using Hessian Trace or Fisher Information
2. **Solves a knapsack-style optimization** to allocate bit-widths per layer under a VRAM budget
3. **Outputs a mixed-precision model** (e.g., 8-bit perplexity quality at 4-bit storage size)

**Core claim:** Achieve `<2%` perplexity delta vs. full-precision FP16, at `2×` smaller memory footprint vs. INT8 baseline — fully automated, zero manual tuning.

**Supported architectures:** GPT-2, LLaMA-2 (7B/13B), Mistral-7B  
**Supported backends:** TensorRT, llama.cpp  
**Interface:** CLI (`quantize.py`), Flask API (`app.py`), Jupyter Notebooks

---

## 🗂️ Final Project Structure

```
autoquant/
├── autoquant/                    # Core library package
│   ├── __init__.py               # Exports AutoQuantizer
│   ├── sensitivity.py            # Hessian/Fisher sensitivity profiler
│   ├── allocator.py              # Knapsack bit-width optimizer
│   ├── quantizer.py              # Layer replacement + weight packing
│   └── utils.py                  # Size estimation, reporting helpers
│
├── notebooks/                    # Anaconda/Jupyter notebooks
│   ├── 01_sensitivity_analysis.ipynb
│   ├── 02_bit_allocation.ipynb
│   ├── 03_quantize_and_evaluate.ipynb
│   └── 04_benchmarking.ipynb
│
├── app.py                        # Flask REST API (already exists)
├── quantize.py                   # CLI entry point (already exists)
├── evaluate.py                   # Evaluation script (already exists)
├── test_direct.py                # Quick smoke test (already exists)
├── config.json                   # Example config output (already exists)
├── requirements.txt              # Python deps (already exists)
├── environment.yml               # Anaconda environment spec (TO CREATE)
├── .gitignore                    # (already exists)
└── README.md                     # (TO CREATE)
```

---

## 🖥️ Environment Setup (Anaconda)

### Step 1 — Install Anaconda

Download from https://www.anaconda.com/download and install. Then verify:

```bash
conda --version
```

### Step 2 — Create the `environment.yml`

Create this file at the project root (Cursor agent should create this):

```yaml
# environment.yml
name: autoquant
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults

dependencies:
  - python=3.10
  - pip
  - jupyter
  - jupyterlab
  - ipykernel
  - ipywidgets
  - matplotlib
  - seaborn
  - numpy>=1.24.0
  - tqdm>=4.65.0
  - pip:
      - torch>=2.0.0
      - transformers>=4.35.0
      - accelerate>=0.24.0
      - bitsandbytes>=0.41.0
      - sentencepiece>=0.1.99
      - protobuf>=3.20.0
      - flask>=2.3.0
      - flask-cors>=4.0.0
      - psutil>=5.9.0
```

### Step 3 — Create & Activate the Environment

```bash
conda env create -f environment.yml
conda activate autoquant
```

### Step 4 — Register as Jupyter Kernel

```bash
python -m ipykernel install --user --name autoquant --display-name "Python (autoquant)"
```

### Step 5 — Open in Cursor

```bash
# From project root
cursor .
```

In Cursor, open any `.ipynb` file and select the **"Python (autoquant)"** kernel from the top-right kernel picker.

---

## 🧱 Core Library — `autoquant/` Package

This is what's **missing** from the repo. The existing scripts (`quantize.py`, `app.py`, `evaluate.py`) all import `from autoquant import AutoQuantizer` — but the package itself needs to be built. This is the main engineering work.

---

### `autoquant/__init__.py`

```python
from .quantizer import AutoQuantizer

__all__ = ["AutoQuantizer"]
__version__ = "0.1.0"
```

---

### `autoquant/sensitivity.py`

**Purpose:** For each linear layer in the model, compute a sensitivity score (0.0 = irrelevant, 1.0 = critical). Uses Fisher Information approximation via gradient magnitude on calibration samples.

```python
"""
Sensitivity Profiler
Ranks each linear layer by how much quantization noise would hurt the model.
Uses Fisher Information (gradient^2 approximation) on calibration text samples.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import Dict


CALIBRATION_TEXT = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world of software engineering.",
    "Large language models require significant memory to run inference.",
    "Quantization reduces model size while preserving most of the accuracy.",
    "The transformer architecture relies on attention mechanisms.",
    # Add more diverse calibration sentences for better sensitivity estimates
]


def compute_sensitivity(
    model: nn.Module,
    tokenizer,
    num_samples: int = 100,
    device: str = "cpu",
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
    
    # Identify all Linear layers and register gradient accumulators
    sensitivity_accum: Dict[str, float] = {}
    hooks = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            sensitivity_accum[name] = 0.0

    # Run calibration passes
    texts = (CALIBRATION_TEXT * ((num_samples // len(CALIBRATION_TEXT)) + 1))[:num_samples]
    
    for text in tqdm(texts, desc="Computing sensitivity"):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(device)
        
        # Zero gradients
        model.zero_grad()
        
        # Forward pass with labels = input_ids for causal LM loss
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        
        # Accumulate gradient^2 for each linear layer (Fisher approx)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.grad is not None:
                grad_sq = module.weight.grad.detach().pow(2).mean().item()
                sensitivity_accum[name] += grad_sq
    
    # Normalize to [0, 1]
    max_val = max(sensitivity_accum.values()) if sensitivity_accum else 1.0
    if max_val == 0:
        max_val = 1.0
    
    normalized = {
        name: score / max_val
        for name, score in sensitivity_accum.items()
    }
    
    # Clean up
    model.zero_grad()
    for h in hooks:
        h.remove()
    
    return normalized
```

**Key design notes for Cursor agent:**
- Use `loss.backward()` not `torch.autograd.grad()` — simpler and compatible with all HF models
- Always call `model.zero_grad()` before each sample to prevent gradient accumulation
- `detach()` before `.item()` to avoid holding computation graph in memory
- Normalization is critical — raw Fisher values vary wildly across architectures

---

### `autoquant/allocator.py`

**Purpose:** Given sensitivity scores and a memory budget (GB), solve for the optimal per-layer bit-width using a greedy knapsack algorithm.

```python
"""
Bit-Width Allocator
Solves the constrained optimization: maximize model accuracy (proxy: keep
high-sensitivity layers at high precision) subject to a total memory budget.

Uses a greedy knapsack approach:
  1. Start all layers at the minimum bit-width (INT4)
  2. Sort layers by sensitivity score (descending)
  3. Upgrade layers from INT4 → INT8 → FP16 as long as budget allows
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn


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
    target_bytes = target_size_gb * (1024 ** 3)
    
    # Collect layers and their base info
    layer_info = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in sensitivity_scores:
            layer_info[name] = {
                "module": module,
                "sensitivity": sensitivity_scores[name],
                "size_at_bits": {
                    b: estimate_layer_size_bytes(module, b)
                    for b in BIT_WIDTH_OPTIONS
                }
            }
    
    # Count non-linear parameters (embeddings, layer norms) — always FP16
    non_linear_bytes = 0.0
    non_linear_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Embedding, nn.LayerNorm)) or (
            hasattr(module, "weight") and not isinstance(module, nn.Linear)
            and name not in layer_info
        ):
            for pname, param in module.named_parameters(recurse=False):
                non_linear_bytes += param.numel() * 2.0  # FP16 = 2 bytes
            if hasattr(module, "weight"):
                non_linear_layers[name] = 16
    
    # Start all linear layers at min_bits
    assignments = {name: min_bits for name in layer_info}
    current_bytes = non_linear_bytes + sum(
        info["size_at_bits"][min_bits] for info in layer_info.values()
    )
    
    # Sort by sensitivity descending — most sensitive layers get upgraded first
    sorted_layers = sorted(
        layer_info.items(),
        key=lambda x: x[1]["sensitivity"],
        reverse=True
    )
    
    # Greedy upgrade loop
    for bits_target in [8, 16]:  # Try upgrading to INT8 first, then FP16
        for name, info in sorted_layers:
            if assignments[name] >= bits_target:
                continue
            
            current_size = info["size_at_bits"][assignments[name]]
            new_size = info["size_at_bits"][bits_target]
            delta = new_size - current_size
            
            if current_bytes + delta <= target_bytes:
                current_bytes += delta
                assignments[name] = bits_target
    
    # Merge non-linear layer assignments
    final_assignments = {**assignments, **non_linear_layers}
    
    expected_gb = current_bytes / (1024 ** 3)
    return final_assignments, expected_gb


def summarize_assignments(assignments: Dict[str, int]) -> Dict[str, int]:
    """Returns a count of how many layers are at each bit-width."""
    from collections import Counter
    return dict(Counter(assignments.values()))
```

**Key design notes for Cursor agent:**
- Always account for non-linear layers (embeddings, LayerNorm) — they are always FP16 and cannot be quantized without severe accuracy loss
- The greedy two-pass approach (upgrade to 8 first, then 16) is more stable than a single-pass sort
- Budget math must match `evaluate.py`'s size calculation exactly — use the same `BYTES_PER_BIT` table

---

### `autoquant/quantizer.py`

**Purpose:** The main `AutoQuantizer` class that orchestrates sensitivity analysis, bit allocation, and weight packing. This is what all existing scripts import.

```python
"""
AutoQuantizer — Main orchestration class.

Usage:
    quantizer = AutoQuantizer("gpt2")
    sensitivity = quantizer.analyze_sensitivity(num_samples=100)
    config = quantizer.create_config(target_size_gb=0.5)
    quantizer.quantize("config.json", "output_dir/")
    report = quantizer.get_report()
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional

from .sensitivity import compute_sensitivity
from .allocator import allocate_bits, summarize_assignments
from .utils import compute_model_size_gb, get_device


class DynamicQuantizedLinear(nn.Module):
    """
    INT-backed linear layer that stores weights as quantized integers with
    a per-output-channel scale factor. During forward pass, weights are
    dequantized on the fly: weight_fp = weight_int * scale.
    
    This is structural quantization — the model file stores INT8 weights,
    reducing disk/memory footprint, while compute happens in FP16.
    """
    
    def __init__(self, original_linear: nn.Linear, bits: int):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.bits = bits
        
        # Quantize weights: compute scale per output channel
        weight_fp = original_linear.weight.data.float()
        
        # Per-channel symmetric quantization
        max_val = weight_fp.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        num_levels = 2 ** (bits - 1) - 1  # e.g., 127 for INT8
        scale = max_val / num_levels
        
        weight_q = (weight_fp / scale).round().clamp(-num_levels, num_levels).to(torch.int8)
        
        self.register_buffer("weight_q", weight_q)
        self.register_buffer("scale", scale.to(torch.float16))
        
        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.data.to(torch.float16))
        else:
            self.register_parameter("bias", None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize: INT8 * FP16 scale → FP16
        weight_deq = self.weight_q.to(self.scale.dtype) * self.scale
        return F.linear(x, weight_deq, self.bias)
    
    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bits={self.bits}"


def _replace_layer(model: nn.Module, layer_name: str, new_module: nn.Module):
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
        1. __init__        — load model + tokenizer
        2. analyze_sensitivity — compute per-layer Fisher scores
        3. create_config   — solve bit allocation under budget
        4. quantize        — apply quantization and save
        5. get_report      — return compression statistics
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
        
        self.original_size = compute_model_size_gb(self.model)
        print(f"   Original size: {self.original_size:.3f} GB")
        
        # State populated by pipeline steps
        self.sensitivity_scores: Optional[Dict[str, float]] = None
        self.bit_assignments: Optional[Dict[str, int]] = None
        self.quantized_size: Optional[float] = None
        self.output_dir: Optional[str] = None
    
    def analyze_sensitivity(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Compute Fisher Information sensitivity scores for all linear layers.
        
        Args:
            num_samples: Number of calibration text samples (more = more accurate but slower)
        
        Returns:
            Dict[layer_name -> sensitivity_score] normalized to [0, 1]
        """
        print(f"🔬 Analyzing sensitivity ({num_samples} samples)...")
        self.sensitivity_scores = compute_sensitivity(
            self.model, self.tokenizer, num_samples, device=self.device
        )
        print(f"   Scored {len(self.sensitivity_scores)} layers")
        return self.sensitivity_scores
    
    def create_config(self, target_size_gb: float) -> dict:
        """
        Solve the bit-allocation optimization problem.
        
        Args:
            target_size_gb: Maximum allowed model size after quantization (in GB).
                            If larger than original size, all layers stay at FP16.
        
        Returns:
            Dict with keys: config, expected_size_gb, compression_ratio
        """
        if self.sensitivity_scores is None:
            raise RuntimeError("Call analyze_sensitivity() first")
        
        print(f"🎯 Allocating bits for {target_size_gb:.2f} GB target...")
        
        self.bit_assignments, expected_size = allocate_bits(
            self.model, self.sensitivity_scores, target_size_gb
        )
        
        compression_ratio = self.original_size / expected_size if expected_size > 0 else 1.0
        
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
        print(f"   Expected size: {expected_size:.3f} GB ({compression_ratio:.2f}x compression)")
        print(f"   Bit distribution: {dist}")
        
        return {
            "config": config,
            "expected_size_gb": expected_size,
            "compression_ratio": compression_ratio,
        }
    
    def quantize(self, config_path: str, output_dir: str):
        """
        Apply mixed-precision quantization and save to disk.
        
        Reads bit assignments from config_path, replaces linear layers with
        DynamicQuantizedLinear where bits < 16, saves weights + config.
        
        Args:
            config_path: Path to config.json produced by create_config()
            output_dir: Directory to write quantized model files
        """
        with open(config_path, "r") as f:
            config = json.load(f)
        
        assignments = config["bit_assignments"]
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"⚙️  Applying quantization to {sum(1 for b in assignments.values() if b < 16)} layers...")
        
        replaced = 0
        for name, bits in assignments.items():
            if bits >= 16:
                continue
            for module_name, module in list(self.model.named_modules()):
                if module_name == name and isinstance(module, nn.Linear):
                    new_layer = DynamicQuantizedLinear(module, bits)
                    _replace_layer(self.model, module_name, new_layer)
                    replaced += 1
                    break
        
        print(f"   Replaced {replaced} layers with INT-backed quantized layers")
        
        # Save weights
        weights_path = os.path.join(output_dir, "pytorch_model.bin")
        print(f"💾 Saving weights to {weights_path}...")
        torch.save(self.model.state_dict(), weights_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save quantization config (needed by evaluate.py)
        quant_config = {
            "model_name": self.model_name,
            "bit_assignments": assignments,
            "bit_distribution": config.get("bit_distribution", {}),
            "original_size_gb": config.get("original_size_gb", self.original_size),
        }
        with open(os.path.join(output_dir, "quantization_config.json"), "w") as f:
            json.dump(quant_config, f, indent=2)
        
        self.quantized_size = compute_model_size_gb(self.model)
        self.output_dir = output_dir
        print(f"✅ Quantized model saved to '{output_dir}' ({self.quantized_size:.3f} GB)")
    
    def get_report(self) -> dict:
        """Return a summary dict of quantization results."""
        if self.quantized_size is None:
            raise RuntimeError("Call quantize() first")
        
        compression = self.original_size / self.quantized_size if self.quantized_size > 0 else 1.0
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
```

---

### `autoquant/utils.py`

```python
"""
Utility functions for size estimation and device detection.
"""

import torch
import torch.nn as nn
from typing import Optional


def compute_model_size_gb(model: nn.Module) -> float:
    """
    Compute the actual in-memory footprint of a model in gigabytes.
    Accounts for both parameters and registered buffers (important for
    DynamicQuantizedLinear which stores weights as buffers, not parameters).
    """
    total_bytes = 0
    
    # Parameters
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    
    # Buffers (quantized weights, scales, etc.)
    for buf in model.buffers():
        total_bytes += buf.numel() * buf.element_size()
    
    return total_bytes / (1024 ** 3)


def get_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def format_size(size_gb: float) -> str:
    """Human-readable size string."""
    if size_gb < 1.0:
        return f"{size_gb * 1024:.1f} MB"
    return f"{size_gb:.2f} GB"
```

---

## 📓 Jupyter Notebooks (Anaconda)

Create these notebooks in the `notebooks/` directory. Each notebook is self-contained and builds on the previous one.

---

### `notebooks/01_sensitivity_analysis.ipynb`

```python
# Cell 1 — Setup
import sys
sys.path.insert(0, "..")  # Add project root so we can import autoquant
import torch
from autoquant import AutoQuantizer

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

# Cell 2 — Load and analyze GPT-2
quantizer = AutoQuantizer("gpt2")
sensitivity = quantizer.analyze_sensitivity(num_samples=50)  # 50 for speed

# Cell 3 — Visualize sensitivity distribution
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.DataFrame([
    {"layer": name.split(".")[-2] + "." + name.split(".")[-1], 
     "full_name": name,
     "score": score}
    for name, score in sensitivity.items()
]).sort_values("score", ascending=False)

plt.figure(figsize=(14, 6))
colors = ["#d62728" if s > 0.7 else "#ff7f0e" if s > 0.4 else "#1f77b4" 
          for s in df["score"]]
plt.bar(range(len(df)), df["score"], color=colors)
plt.xlabel("Layer (sorted by sensitivity)")
plt.ylabel("Sensitivity Score")
plt.title("AutoQuant: Per-Layer Sensitivity Scores (GPT-2)\nRed=High (needs FP16), Blue=Low (can use INT4)")
plt.xticks([])
plt.tight_layout()
plt.savefig("sensitivity_plot.png", dpi=150)
plt.show()

# Cell 4 — Print top-10 most sensitive layers
print("🔴 Top 10 most sensitive layers (keep at FP16):")
for _, row in df.head(10).iterrows():
    bar = "█" * int(row["score"] * 20)
    print(f"  {row['full_name']:50s} {bar} {row['score']:.3f}")

print("\n🔵 Top 10 least sensitive layers (can compress to INT4):")
for _, row in df.tail(10).iterrows():
    bar = "█" * int(row["score"] * 20)
    print(f"  {row['full_name']:50s} {bar} {row['score']:.3f}")
```

---

### `notebooks/02_bit_allocation.ipynb`

```python
# Cell 1 — Run full pipeline up to config creation
import sys; sys.path.insert(0, "..")
from autoquant import AutoQuantizer
import json, matplotlib.pyplot as plt

quantizer = AutoQuantizer("gpt2")
quantizer.analyze_sensitivity(num_samples=50)

# Cell 2 — Try multiple target sizes and compare
targets = [0.1, 0.15, 0.2, 0.3]  # GB targets for GPT-2 (originally ~0.3 GB)
results = []

for target in targets:
    result = quantizer.create_config(target)
    results.append({
        "target": target,
        "expected": result["expected_size_gb"],
        "compression": result["compression_ratio"],
        "config": result["config"],
    })
    print(f"Target {target:.2f} GB → Expected {result['expected_size_gb']:.3f} GB "
          f"({result['compression_ratio']:.2f}x compression)")

# Cell 3 — Visualize bit distribution for chosen target
chosen = results[1]  # 0.15 GB target
assignments = chosen["config"]["bit_assignments"]

from collections import Counter
dist = Counter(assignments.values())

plt.figure(figsize=(8, 5))
colors = {"4": "#2ecc71", "8": "#f39c12", "16": "#e74c3c"}
bars = plt.bar(
    [f"INT{b}" if b < 16 else "FP16" for b in sorted(dist.keys())],
    [dist[b] for b in sorted(dist.keys())],
    color=[colors.get(str(b), "#3498db") for b in sorted(dist.keys())]
)
plt.title(f"Bit Distribution at {chosen['target']:.2f} GB Target\n"
          f"({chosen['expected']:.3f} GB actual, {chosen['compression']:.2f}x compression)")
plt.ylabel("Number of Layers")
for bar, count in zip(bars, [dist[b] for b in sorted(dist.keys())]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             str(count), ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

# Cell 4 — Save config for next notebook
import json
with open("../config.json", "w") as f:
    json.dump(chosen["config"], f, indent=2)
print(f"✅ Config saved to config.json")
```

---

### `notebooks/03_quantize_and_evaluate.ipynb`

```python
# Cell 1 — Quantize
import sys; sys.path.insert(0, "..")
from autoquant import AutoQuantizer

quantizer = AutoQuantizer("gpt2")
quantizer.analyze_sensitivity(num_samples=50)
quantizer.create_config(target_size_gb=0.15)

import json
with open("../config.json", "w") as f:
    json.dump(quantizer.bit_assignments, f, indent=2)  # quick save

quantizer.quantize("../config.json", "../quantized_gpt2")
report = quantizer.get_report()

print(f"\n📊 Quantization Report:")
print(f"   Original:   {report['original_size_gb']:.3f} GB")
print(f"   Quantized:  {report['quantized_size_gb']:.3f} GB")
print(f"   Compression:{report['compression_ratio']:.2f}x")
print(f"   Space saved:{report['space_saved_percent']:.1f}%")

# Cell 2 — Generate text with quantized model
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("../quantized_gpt2")

# Load quantized model using evaluate.py's approach
import subprocess
result = subprocess.run(
    ["python", "../evaluate.py", "../quantized_gpt2"],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)

# Cell 3 — Compare perplexity (FP16 vs quantized)
# This requires running both models on the same test set
# For a quick demo, we measure loss on a single prompt

from autoquant.quantizer import DynamicQuantizedLinear
import torch.nn as nn

def compute_loss_on_prompt(model, tokenizer, prompt, device="cpu"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return outputs.loss.item()

test_prompt = "The future of artificial intelligence depends on"
# (Run this with both models loaded to compare)
print("📝 Quantization complete! Run evaluate.py for full generation test.")
```

---

### `notebooks/04_benchmarking.ipynb`

```python
# Cell 1 — Benchmark inference speed: FP16 vs quantized
import sys; sys.path.insert(0, "..")
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

def benchmark_generation(model, tokenizer, prompt, n_tokens=50, n_runs=5, device="cpu"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model.generate(
                **inputs, max_new_tokens=n_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return sum(times) / len(times), min(times)

# Cell 2 — Load both models and compare
tokenizer_fp16 = AutoTokenizer.from_pretrained("gpt2")
model_fp16 = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16)

prompt = "The transformer architecture was introduced in 2017 and has since"
avg_fp16, min_fp16 = benchmark_generation(model_fp16, tokenizer_fp16, prompt)
print(f"FP16 baseline:  avg={avg_fp16:.3f}s, min={min_fp16:.3f}s")

# Load quantized model (if it exists)
import os
if os.path.exists("../quantized_gpt2"):
    # ... load and benchmark quantized model
    print("Load quantized model and compare here")

# Cell 3 — Memory comparison
def model_memory_gb(model):
    total = sum(
        (p.numel() * p.element_size()) for p in model.parameters()
    ) + sum(
        (b.numel() * b.element_size()) for b in model.buffers()
    )
    return total / (1024**3)

print(f"\nMemory usage:")
print(f"  FP16:      {model_memory_gb(model_fp16):.3f} GB")
# print(f"  Quantized: {model_memory_gb(model_quant):.3f} GB")
```

---

## ⚙️ Cursor IDE Agent Instructions

When opening this project in Cursor, give the agent these instructions:

### Prompt to bootstrap the project

```
I need you to build the `autoquant` Python package from scratch.
The project already has: app.py, quantize.py, evaluate.py, test_direct.py, config.json, requirements.txt.
All of these import `from autoquant import AutoQuantizer`.

Your job:
1. Create the directory `autoquant/` with `__init__.py`, `sensitivity.py`, `allocator.py`, `quantizer.py`, `utils.py` — use the exact code in AUTOQUANT_BUILD_GUIDE.md
2. Create `environment.yml` for Anaconda
3. Create `notebooks/` directory with 4 notebooks: 01_sensitivity_analysis.ipynb, 02_bit_allocation.ipynb, 03_quantize_and_evaluate.ipynb, 04_benchmarking.ipynb
4. Do NOT modify any existing files unless I explicitly ask

After creating, run: python test_direct.py
Fix any import errors before calling it done.
```

### Cursor Agent Settings (`.cursorrules` file)

Create a `.cursorrules` file at the project root:

```
# AutoQuant Project Rules

## Architecture
- The core logic lives in autoquant/ package
- DynamicQuantizedLinear MUST match the implementation in evaluate.py exactly
  (same buffer names: weight_q, scale, bias — evaluate.py loads these by name)
- Sensitivity scores MUST be normalized to [0, 1] range
- Size calculations MUST count buffers AND parameters (quantized layers use buffers)

## Style
- Use type hints everywhere
- Docstrings on every class and public method
- Print progress with emoji (model loading uses 📥, analysis uses 🔬, etc.)
- Never suppress exceptions silently

## Testing
- Always verify with: python test_direct.py
- The test expects a "quantized_test/" directory to be created
- The Flask API (app.py) should start cleanly: python app.py

## Compatibility
- Python 3.10+
- PyTorch 2.0+
- HuggingFace transformers 4.35+
```

---

## 🚀 Running the Project

### Option A: Jupyter Notebook (Recommended for learning/reviewing)

```bash
conda activate autoquant
cd autoquant/  # project root
jupyter lab
# Open notebooks/01_sensitivity_analysis.ipynb
# Select kernel: "Python (autoquant)"
# Run cells top to bottom
```

### Option B: CLI

```bash
conda activate autoquant

# Analyze only (no quantization)
python quantize.py --model gpt2 --target 0.15 --analyze-only

# Full quantization
python quantize.py --model gpt2 --target 0.15 --output quantized_gpt2

# Evaluate the result
python evaluate.py quantized_gpt2
```

### Option C: Flask API

```bash
conda activate autoquant
python app.py

# In another terminal:
curl -X POST http://localhost:5000/api/quantize \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gpt2"}'

# Check status
curl http://localhost:5000/api/status/0
```

### Option D: Quick smoke test

```bash
conda activate autoquant
python test_direct.py
```

---

## 🐛 Common Issues & Fixes

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'autoquant'` | Package not created yet | Create `autoquant/` directory with `__init__.py` |
| `RuntimeError: Error(s) in loading state_dict` | Buffer names mismatch between `quantizer.py` and `evaluate.py` | Make sure both use `weight_q`, `scale`, `bias` as buffer names |
| `OutOfMemoryError` during sensitivity analysis | Model too large for available VRAM | Reduce `num_samples` or use `device="cpu"` |
| `KeyError: 'transformer.h.0.attn...'` in allocator | Layer name format differs by model | Run `{name for name, _ in model.named_modules()}` to inspect actual names |
| `tokenizer has no pad_token` | GPT-2 has no pad token by default | Set `tokenizer.pad_token = tokenizer.eos_token` in `__init__` |
| Notebook kernel not found | Kernel not registered | Run `python -m ipykernel install --user --name autoquant` |
| Size after quantization is same as FP16 | All layers assigned 16-bit | Target size >= original size — try a smaller target |

---

## 🔬 Key Concepts for Understanding the Code

**Why Fisher Information?**  
The Fisher Information Matrix (FIM) measures how much the model's output distribution would change if we perturbed a weight. High FIM = the layer is doing important work = it needs high precision. Low FIM = the layer is redundant = safe to compress. We approximate the full FIM using the squared gradient, which is computationally cheap.

**Why the Knapsack framing?**  
Each layer has a "value" (accuracy contribution, proxied by sensitivity) and a "weight" (memory cost). We want to maximize total value under a memory budget constraint. This is exactly the 0/1 knapsack problem. AutoQuant uses a greedy relaxation (sort by sensitivity, greedily upgrade bits) which is fast and produces near-optimal solutions in practice.

**Why per-output-channel scales?**  
Transformer weight matrices often have a few output channels with much larger magnitude than others (outlier features). Using one global scale would waste precision on small-magnitude channels. Per-channel scaling ensures each output channel uses the full INT8 range, minimizing rounding error.

**Why register_buffer instead of nn.Parameter?**  
Quantized weights (`weight_q`) are not trained — they're fixed after quantization. Using `register_buffer` ensures they: (1) move with `.to(device)`, (2) are saved/loaded with `state_dict()`, but (3) are not included in `model.parameters()` or optimizer updates.

---

## 📈 Expected Results (GPT-2 Baseline)

| Config | Size | Compression | Notes |
|---|---|---|---|
| FP16 baseline | ~0.30 GB | 1.0x | All layers float16 |
| INT8 uniform | ~0.15 GB | 2.0x | All linear INT8 |
| AutoQuant mixed | ~0.15 GB | 2.0x | Sensitive=FP16, rest=INT8/INT4 |
| INT4 aggressive | ~0.08 GB | 3.8x | Perplexity degrades ~5-10% |

The AutoQuant advantage shows more clearly on larger models (LLaMA-7B, Mistral-7B) where layer heterogeneity is more pronounced and the sensitivity-guided allocation preserves accuracy that uniform INT4 would destroy.

---

## 📚 References

- **AutoQuant paper (this project):** See the 10-slide deck for full motivation and benchmarks
- **GPTQ** (Frantar et al., 2022): Hessian-based weight quantization — theoretical basis for sensitivity analysis
- **LLM.int8()** (Dettmers et al., 2022): Mixed-precision decomposition for outlier features
- **AWQ** (Lin et al., 2023): Activation-aware weight quantization — related approach
- **bitsandbytes library:** Used as optional backend for INT8/INT4 kernels
