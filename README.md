# AutoQuant (SE-IN-AI)

Mixed-precision LLM quantization: **per-layer sensitivity** (Fisher-style gradient proxy), **greedy bit allocation** under a memory budget, and **INT-backed linear layers** with per-channel scales. Targets Hugging Face causal LMs (GPT-2, and LLaMA-style models that use `nn.Linear`).

## Quick start (pip)

```bash
cd /path/to/SE-IN-AI
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Cursor / VS Code: “Import could not be resolved” (Pylance / basedpyright)

The editor must use the **same Python** where you ran `pip install -r requirements.txt`.

1. **Command Palette** (`Ctrl+Shift+P`) → **Python: Select Interpreter**.
2. Pick **Python 3.12.x** (or your venv: `.venv\Scripts\python.exe` after you create it).
3. **Developer: Reload Window** if squiggles remain.

This repo sets `python.defaultInterpreterPath` to `${workspaceFolder}/.venv/Scripts/python.exe`. After creating `.venv` and installing requirements there, analysis errors should clear. If Windows blocks `.venv` scripts, select your global interpreter that already has `torch`, `flask`, and `transformers` installed.

**Prove which interpreter runs:** from the repo root:

```bash
python scripts/check_python_env.py
```

That prints `sys.executable`, `torch` / CUDA info, and whether `transformers` / `flask` import. Pick that exact interpreter in the IDE.

If **Windows Application Control** blocks `pip.exe`, install with:

```bash
python -m pip install -r requirements.txt
```

```bash
# Analyze + quantize + save to quantized_model/
python quantize.py --model gpt2 --target 0.15 --output quantized_gpt2

# Smoke test (writes config.json and quantized_test/)
python test_direct.py

# Load checkpoint and generate text
python evaluate.py quantized_gpt2
```

### Testing after quantization (what actually happens)

1. **Artifacts on disk** — The run writes a **folder under the project root** (for example `quantized_model_0/`) containing `pytorch_model.bin`, `quantization_config.json`, tokenizer files, and `_run_config.json` (bit plan used when packing weights).

2. **`evaluate.py`** — Loads the **same** Hugging Face architecture as the base model, replaces eligible `nn.Linear` modules with the project’s quantized linear implementation, loads packed weights from `pytorch_model.bin`, then runs **one short `generate()`** from your prompt. It prints timing, device, and a text sample. This is a **smoke test**, not a full benchmark (no perplexity sweep in-repo).

3. **Dashboard** — After a job finishes, the UI shows an **“After quantization — how to test”** card: the saved folder name, the exact CLI command, and a button that calls **`POST /api/evaluate`** with the same logic as the CLI.

4. **API** — `POST /api/evaluate` with JSON `{ "output_dir": "<folder_name>", "prompt": "...", "max_new_tokens": 48 }` returns a JSON object (`ok`, `generated_text`, `seconds`, errors if any).

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

- `model_name` (required), e.g. `"gpt2"` or `"facebook/opt-125m"`
- `target_size_gb` (default `0.2`)
- `num_samples` (integer, clamped **4–200**)
- `output_dir` (optional) — **single folder name** under the project root (no paths, no `..`). Default: `quantized_model_<task_id>`.

`GET /api/status/<task_id>` returns `running` | `done` | `error`, `log_lines`, `stats`, and when finished: `output_dir`, `evaluate_cli`, `evaluate_api_hint`.

**Single-flight:** only one quantization job at a time. A second `POST /api/quantize` while one is running returns **429** with a clear error.

`GET /api/health` — `project_root`, free disk (GB), and merged GPU / torch fields (same shape as device diagnostics).

`GET /api/tasks` — short summaries of tasks held in memory for this server process.

`POST /api/evaluate` — body `{ "output_dir": "quantized_model_0", "prompt": "...", "max_new_tokens": 40, "do_sample": true }`; folder must exist under project root.

`GET /api/ollama/models` — **optional** (phase D1): lists tag names from a local Ollama daemon (`OLLAMA_HOST` or `http://127.0.0.1:11434`). AutoQuant checkpoints are Hugging Face–format, not Ollama GGUF; this endpoint is only for convenience when Ollama is installed separately.

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
