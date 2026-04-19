"""
Sensitivity Profiler
Ranks each linear layer by how much quantization noise would hurt the model.
Uses Fisher-style squared gradients with activation weighting, gradient clipping,
and a diversified calibration corpus.
"""

from typing import Callable, Dict, List, Optional, cast

import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import is_quantizable_weight_module

CALIBRATION_TEXT: List[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world of software engineering.",
    "Large language models require significant memory to run inference.",
    "Quantization reduces model size while preserving most of the accuracy.",
    "The transformer architecture relies on attention mechanisms.",
    "Machine learning systems improve with more data and careful engineering.",
    "Neural networks approximate functions by composing linear layers and nonlinearities.",
    "Efficient deployment of LLMs requires compression and hardware-aware optimization.",
    # Code & tooling
    "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n",
    "SELECT user_id, COUNT(*) FROM orders GROUP BY user_id HAVING COUNT(*) > 5;",
    "async function fetchData(url) { const r = await fetch(url); return r.json(); }",
    "import torch.nn as nn\nclass MLP(nn.Module):\n    def __init__(self, d): super().__init__(); self.lin = nn.Linear(d, d)",
    # Math / reasoning
    "If a train travels 120 km in 90 minutes, its average speed in km/h is 80 because 120/(1.5)=80.",
    "The derivative of x^3 with respect to x is 3x^2, and the integral of 2x is x^2 plus a constant.",
    "A prime number greater than 2 is odd; 17 is prime because it has no divisors other than 1 and itself.",
    # Dialogue
    "User: I feel stuck on this bug.\nAssistant: Let's reproduce it step by step and narrow the stack trace.",
    "A: Can you summarize the paper?\nB: It proposes a sparse attention variant that cuts memory on long sequences.",
    # Factual Q&A
    "The capital of France is Paris, and the Seine river flows through the city.",
    "Photosynthesis converts light energy into chemical energy in plants, using chlorophyll.",
    "HTTP 404 means the server could not find the requested resource for that URL.",
    # Multilingual
    "Bonjour, comment allez-vous aujourd'hui? Je voudrais un café s'il vous plaît.",
    "Hola, ¿cómo estás? Mañana iremos al mercado si no llueve.",
    "你好，今天天气很好。我们下午可以去公园散步。",
    "Guten Morgen, ich möchte eine Fahrkarte nach München kaufen.",
    "नमस्ते, आप कैसे हैं? मैं कल दिल्ली जा रहा हूँ।",
    "こんにちは、元気ですか。明日の会議の資料を送ってください。",
    # Mixed short prompts
    "Explain backpropagation in two sentences for a junior engineer.",
    "Rewrite the sentence using active voice: the report was written by the team.",
    "List three risks of deploying an LLM without guardrails in production.",
]


def compute_sensitivity(
    model: nn.Module,
    tokenizer,
    num_samples: int = 100,
    device: str = "cpu",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, float]:
    """
    Per-layer sensitivity: clipped squared-gradient mean × mean |activation|,
    accumulated over calibration texts, then min–max normalized to [0, 1].
    """
    model.eval()
    model.to(device)

    sensitivity_accum: Dict[str, float] = {}

    for name, module in model.named_modules():
        if is_quantizable_weight_module(module):
            sensitivity_accum[name] = 0.0

    texts = (CALIBRATION_TEXT * ((num_samples // len(CALIBRATION_TEXT)) + 1))[
        :num_samples
    ]

    n_texts = len(texts)
    if progress_callback is None:
        text_iter = enumerate(tqdm(texts, desc="Computing sensitivity"))
    else:
        text_iter = enumerate(texts)

    handles: list = []
    act_mag_step: Dict[str, float] = {}

    try:
        for name, module in model.named_modules():
            if not is_quantizable_weight_module(module):
                continue

            def _make_hook(nm: str):
                def _hook(
                    mod: nn.Module,
                    inp: tuple,
                    _out: object,
                ) -> None:
                    x = inp[0]
                    if x is None:
                        return
                    with torch.no_grad():
                        act_mag_step[nm] = float(x.detach().abs().mean().cpu())

                return _hook

            handles.append(module.register_forward_hook(_make_hook(name)))

        for i, text in text_iter:
            act_mag_step.clear()
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=128
            ).to(device)

            model.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = cast(torch.Tensor, outputs.loss)
                loss.backward()

            for name, module in model.named_modules():
                if not is_quantizable_weight_module(module):
                    continue
                if module.weight.grad is None:
                    continue
                g = module.weight.grad.detach().float()
                flat = g.abs().flatten()
                if flat.numel() == 0:
                    continue
                # torch.quantile has element-count limits on some builds; subsample safely.
                max_q = 2_000_000
                if flat.numel() > max_q:
                    step = max(1, flat.numel() // max_q)
                    flat = flat[::step]
                thr = torch.quantile(flat, 0.99)
                if not torch.isfinite(thr) or float(thr) <= 0:
                    thr = torch.tensor(1e-8, device=g.device, dtype=g.dtype)
                g_clipped = g.clamp(-thr, thr)
                grad_sq = float(g_clipped.pow(2).mean().cpu())
                act = max(act_mag_step.get(name, 0.0), 1e-8)
                sensitivity_accum[name] += grad_sq * act

            model.zero_grad()

            if progress_callback is not None:
                progress_callback(i + 1, n_texts, "sensitivity")
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    vals = list(sensitivity_accum.values())
    if not vals:
        return {}
    vmin, vmax = min(vals), max(vals)
    if vmax <= vmin + 1e-12:
        normalized = {n: 0.5 for n in sensitivity_accum}
    else:
        scale = vmax - vmin + 1e-8
        normalized = {n: (s - vmin) / scale for n, s in sensitivity_accum.items()}

    model.zero_grad()

    return normalized
