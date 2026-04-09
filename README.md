# AutoQuant (SE-IN-AI)

Mixed-precision LLM quantization: **per-layer sensitivity** (Fisher-style gradient proxy), **greedy bit allocation** under a memory budget, and **INT-backed linear layers** with per-channel scales. Targets Hugging Face causal LMs (GPT-2, and LLaMA-style models that use `nn.Linear`).

## Quick start (pip)

```bash
cd /path/to/SE-IN-AI
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# Analyze + quantize + save to quantized_model/
python quantize.py --model gpt2 --target 0.15 --output quantized_gpt2

# Smoke test (writes config.json and quantized_test/)
python test_direct.py

# Load checkpoint and generate text
python evaluate.py quantized_gpt2
```

## Conda + Jupyter

See `environment.yml` and **AUTOQUANT_BUILD_GUIDE.md** for `conda env create`, kernel registration, and notebook order (`notebooks/01_*.ipynb` … `04_*.ipynb`).

## Web dashboard (recommended)

```bash
python app.py
```

Open **http://127.0.0.1:5000** — step-by-step UI (load → sensitivity → allocation → quantize), live progress, parameter vs memory stats, and GPU status. Uses **CUDA automatically** when PyTorch is built with CUDA and a GPU is visible.

**If the UI says “CPU only”:** the usual cause is a **CPU-only** PyTorch wheel (`2.x.x+cpu` from plain `pip install torch`).

**RTX 50-series / Blackwell (e.g. RTX 5070 Ti)** use **sm_120**. PyTorch builds with **cu124** only ship kernels up to **sm_90**, so you can see `cuda.is_available() == True` but get `no kernel image is available for execution on the device`. Use **CUDA 12.8 wheels** instead:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Older GPUs** (RTX 30/40, etc.) can use **cu124**:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Sanity check (should run with no error):

```bash
python -c "import torch; t=torch.randn(256,256,device='cuda'); print((t@t).sum().item())"
```

**Server message:** the app defaults to **Waitress** (no Flask “development server” warning). To use Flask’s debugger/reloader instead: `set FLASK_DEBUG=1` then `python app.py`.

## API

`POST /api/quantize` JSON body:

- `model_name` (required), e.g. `"gpt2"`
- `target_size_gb` (default `0.5`)
- `num_samples` (default `100`)
- `output_dir` (default `quantized_model_<task_id>`)

`GET /api/status/<task_id>` returns `running` | `done` | `error` and optional `report`.

## Layout

| Path | Role |
|------|------|
| `autoquant/sensitivity.py` | Fisher-style per-layer scores |
| `autoquant/allocator.py` | Greedy knapsack bit assignment |
| `autoquant/quantizer.py` | `AutoQuantizer`, `DynamicQuantizedLinear`, save/load layout |
| `autoquant/utils.py` | Model size, device, GPT-2 `Conv1D` → `Linear` helper |

GPT-2 uses `transformers.pytorch_utils.Conv1D` for many layers; the code converts those to `nn.Linear` for quantization so checkpoints stay consistent with `evaluate.py`.

## Docs

Full specification and theory: **AUTOQUANT_BUILD_GUIDE.md**.
