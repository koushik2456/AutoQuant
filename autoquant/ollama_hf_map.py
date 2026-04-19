"""
Map common Ollama model names to Hugging Face model ids.

Ollama runs GGUF weights locally; AutoQuant operates on PyTorch checkpoints from Hugging Face.
There is no direct GGUF→INT pipeline here — we suggest an HF architecture that is often the
same family as the Ollama tag so you can quantize equivalent weights.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

# Base name (lowercase, no :tag) -> (hf_model_id, optional note)
OLLAMA_BASE_TO_HF: Dict[str, Tuple[str, Optional[str]]] = {
    "llama3.2": (
        "meta-llama/Llama-3.2-1B-Instruct",
        "Gated on Hugging Face — log in and accept the license, set HF_TOKEN if needed.",
    ),
    "llama3.1": (
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "Large / gated; ensure VRAM and HF access. For a small demo try TinyLlama manually.",
    ),
    "phi3": ("microsoft/Phi-3-mini-4k-instruct", None),
    "phi3.5": ("microsoft/Phi-3.5-mini-instruct", None),
    "gemma2": ("google/gemma-2-2b-it", None),
    "gemma3": (
        "google/gemma-2-2b-it",
        "Ollama’s Gemma 3 tag may differ; this maps to a small public Gemma 2 checkpoint — adjust if you need an exact match.",
    ),
    "qwen2.5": ("Qwen/Qwen2.5-0.5B-Instruct", None),
    "qwen2": ("Qwen/Qwen2-0.5B-Instruct", None),
    "tinyllama": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", None),
    "mistral": (
        "mistralai/Mistral-7B-v0.1",
        "7B model — needs significant disk/VRAM for full FP16 load.",
    ),
    "mixtral": (
        "mistralai/Mixtral-8x7B-v0.1",
        "Very large; quantization budget must fit your hardware.",
    ),
    "codellama": ("codellama/CodeLlama-7b-Instruct-hf", None),
    "deepseek-r1": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Distilled small variant for local runs."),
    "deepseek": ("deepseek-ai/deepseek-llm-7b-chat", None),
    "zephyr": ("HuggingFaceH4/zephyr-7b-beta", None),
    "openchat": ("openchat/openchat_3.5", None),
    "nous-hermes": ("NousResearch/Nous-Hermes-2-Mistral-7B-DPO", None),
    "gpt-oss": ("openai-community/gpt2", "Generic fallback — replace with the HF id that matches your Ollama build."),
}


def normalize_ollama_tag(name: str) -> str:
    s = (name or "").strip().lower()
    s = s.split(":")[0].strip()
    s = re.sub(r"[^a-z0-9.\-]+", "-", s)
    s = s.strip("-")
    return s


def suggest_hf_for_ollama(ollama_name: str) -> Dict[str, Any]:
    """
    Return a suggested Hugging Face repo id for an Ollama model tag, if known.
    """
    base = normalize_ollama_tag(ollama_name)
    out: Dict[str, Any] = {
        "ok": True,
        "ollama_input": ollama_name,
        "ollama_base": base,
        "matched": False,
        "hf_model_id": None,
        "note": None,
    }
    if not base:
        out["ok"] = False
        out["error"] = "empty Ollama name"
        return out

    if base in OLLAMA_BASE_TO_HF:
        hf_id, note = OLLAMA_BASE_TO_HF[base]
        out["matched"] = True
        out["hf_model_id"] = hf_id
        out["note"] = note
        return out

    # Prefix match (e.g. llama3.2-something)
    for key, (hf_id, note) in OLLAMA_BASE_TO_HF.items():
        if base.startswith(key) or key.startswith(base):
            out["matched"] = True
            out["hf_model_id"] = hf_id
            out["note"] = note or f"Heuristic match from prefix “{key}”. Verify on Hugging Face."
            return out

    out["note"] = (
        "No built-in mapping for this tag. Ollama stores GGUF binaries — pick the matching "
        "Hugging Face causal LM (same architecture) manually, or add an entry in "
        "autoquant/ollama_hf_map.py."
    )
    return out
