import copy
import datetime
import json
import os
import shutil
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict

import torch
from flask import Flask, jsonify, render_template, request

from autoquant import AutoQuantizer
from autoquant.config import OLLAMA_HOST
from autoquant.quantizer import QuantizationCancelled
from autoquant.api_helpers import sanitize_output_dir, validate_model_name
from autoquant.hf_estimate import estimate_fp16_footprint_gb
from autoquant.ollama_client import ollama_chat, ollama_pull
from autoquant.ollama_hf_map import suggest_hf_for_ollama
from autoquant.utils import (
    gpu_info_dict,
    model_parameter_stats,
    weighted_average_bits_for_quantizable,
)

PROJECT_ROOT = Path(__file__).resolve().parent

app = Flask(__name__)
tasks: Dict[str, Dict[str, Any]] = {}
task_lock = threading.Lock()
_job_lock = threading.Lock()
_job_active = False
_chat_lock = threading.Lock()


def _acquire_quantize_job() -> bool:
    with _job_lock:
        global _job_active
        if _job_active:
            return False
        _job_active = True
        return True


def _release_quantize_job() -> None:
    with _job_lock:
        global _job_active
        _job_active = False


STEP_DEFS = [
    {
        "id": "load",
        "title": "Load model",
        "detail": "Download (if needed) and load weights; use GPU when available.",
    },
    {
        "id": "sensitivity",
        "title": "Profile sensitivity",
        "detail": "Calibration passes: gradient energy per weight layer (Fisher-style).",
    },
    {
        "id": "allocate",
        "title": "Allocate bit widths",
        "detail": "Greedy knapsack: assign INT4 / INT8 / FP16 under your memory budget.",
    },
    {
        "id": "quantize",
        "title": "Quantize & save",
        "detail": "Replace layers with packed INT weights + scales; write checkpoint.",
    },
]


def _safe_dist(d: Dict[Any, Any]) -> Dict[str, int]:
    if not d:
        return {}
    return {str(k): int(v) for k, v in d.items()}


def _merge_stats(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(extra)
    return out


def _update_task(task_id: str, **kwargs: Any) -> None:
    with task_lock:
        if task_id not in tasks:
            tasks[task_id] = {}
        tasks[task_id].update(kwargs)


def _chart_payload(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Structured data for Chart.js (bit mix + size comparison)."""
    dist = stats.get("bit_distribution") or {}
    pairs: list[tuple[str, int]] = []
    for k in sorted(dist.keys(), key=lambda x: int(x)):
        b = int(k)
        lab = "FP16" if b >= 16 else f"INT{b}"
        pairs.append((lab, int(dist[k])))
    return {
        "bit_labels": [p[0] for p in pairs],
        "bit_counts": [p[1] for p in pairs],
        "sizes_gb": {
            "original": stats.get("original_size_gb"),
            "expected": stats.get("expected_size_gb"),
            "quantized": stats.get("quantized_size_gb"),
        },
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/lab")
def lab():
    """Chat with quantized HF checkpoint; optional Ollama baseline."""
    return render_template("chat.html")


@app.route("/api/model/estimate", methods=["GET"])
def model_estimate():
    """FP16 footprint estimate without full weight load (meta when supported)."""
    model_name = request.args.get("model_name") or ""
    ok_m, model_name_norm = validate_model_name(model_name)
    if not ok_m:
        return jsonify({"ok": False, "error": model_name_norm}), 400
    est = estimate_fp16_footprint_gb(model_name_norm)
    if not est.get("ok"):
        return jsonify(est), 400
    return jsonify(est)


@app.route("/api/ollama/suggest-hf", methods=["GET"])
def ollama_suggest_hf():
    """
    Map an Ollama model tag to a suggested Hugging Face causal LM id.
    Ollama serves GGUF; AutoQuant quantizes PyTorch weights from Hugging Face.
    """
    name = (request.args.get("name") or request.args.get("ollama") or "").strip()
    if not name:
        return jsonify({"ok": False, "error": "name query parameter is required"}), 400
    out = suggest_hf_for_ollama(name)
    if not out.get("ok"):
        return jsonify(out), 400
    return jsonify(out)


@app.route("/api/ollama/pull", methods=["POST"])
def ollama_pull_api():
    """Download an Ollama model (uses CLI or HTTP)."""
    data = request.get_json(silent=True) or {}
    name = data.get("model") or data.get("name") or ""
    result = ollama_pull(str(name).strip(), stream=False, timeout_sec=7200.0)
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.route("/api/ollama/chat", methods=["POST"])
def ollama_chat_api():
    """Chat completion via local Ollama."""
    data = request.get_json(silent=True) or {}
    model = data.get("model") or data.get("name") or ""
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        user_msg = (data.get("message") or data.get("prompt") or "").strip()
        if not user_msg:
            return jsonify({"ok": False, "error": "messages or message required"}), 400
        messages = [{"role": "user", "content": user_msg}]
    out = ollama_chat(str(model).strip(), messages, stream=False, timeout_sec=300.0)
    status = 200 if out.get("ok") else 400
    return jsonify(out), status


@app.route("/api/chat/quantized", methods=["POST"])
def chat_quantized_api():
    """Single-turn generation from a quantized project folder (Hugging Face checkpoint)."""
    data = request.get_json(silent=True) or {}
    folder = data.get("output_dir") or data.get("model_path")
    if not folder or not str(folder).strip():
        return (
            jsonify({"ok": False, "error": "output_dir is required (folder under project root)."}),
            400,
        )
    ok, msg, resolved = sanitize_output_dir(
        str(folder).strip(), default_name="quantized_model", project_root=str(PROJECT_ROOT)
    )
    if not ok:
        return jsonify({"ok": False, "error": msg}), 400
    if not os.path.isdir(resolved):
        return jsonify({"ok": False, "error": f"Folder not found: {folder}"}), 404

    user_message = (data.get("message") or data.get("prompt") or "").strip()
    if not user_message:
        return jsonify({"ok": False, "error": "message is required"}), 400
    try:
        max_new_tokens = int(data.get("max_new_tokens", 256))
    except (TypeError, ValueError):
        max_new_tokens = 256
    max_new_tokens = max(1, min(max_new_tokens, 1024))
    do_sample = bool(data.get("do_sample", False))

    from evaluate import chat_quantized

    with _chat_lock:
        result = chat_quantized(
            resolved,
            user_message,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.route("/api/metrics/quality", methods=["POST"])
def metrics_quality():
    """
    FP16 vs quantized loss on a short string (proxy for accuracy change — not full perplexity).
    """
    data = request.get_json(silent=True) or {}
    folder = data.get("output_dir") or data.get("model_path")
    if not folder or not str(folder).strip():
        return (
            jsonify({"ok": False, "error": "output_dir is required."}),
            400,
        )
    ok, msg, resolved = sanitize_output_dir(
        str(folder).strip(), default_name="quantized_model", project_root=str(PROJECT_ROOT)
    )
    if not ok:
        return jsonify({"ok": False, "error": msg}), 400
    if not os.path.isdir(resolved):
        return jsonify({"ok": False, "error": f"Folder not found: {folder}"}), 404

    text = (data.get("calibration_text") or "").strip() or (
        "The future of artificial intelligence depends on efficient models."
    )
    try:
        max_length = int(data.get("max_length", 128))
    except (TypeError, ValueError):
        max_length = 128
    max_length = max(16, min(max_length, 512))

    try:
        num_eval_strings = int(data.get("num_eval_strings", 5))
    except (TypeError, ValueError):
        num_eval_strings = 5
    num_eval_strings = max(1, min(num_eval_strings, 32))

    from autoquant.quality_metrics import compare_fp16_vs_quantized

    with _chat_lock:
        result = compare_fp16_vs_quantized(
            resolved,
            calibration_text=text,
            max_length=max_length,
            num_eval_strings=num_eval_strings,
        )
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.route("/api/health")
def health():
    """Lightweight readiness: Python, torch, CUDA, disk, project root."""
    try:
        du = shutil.disk_usage(str(PROJECT_ROOT))
        free_gb = round(du.free / (1024**3), 2)
    except OSError:
        free_gb = None
    g = gpu_info_dict()
    return jsonify(
        {
            "ok": True,
            "project_root": str(PROJECT_ROOT),
            "disk_free_gb": free_gb,
            **g,
        }
    )


@app.route("/api/tasks")
def list_tasks():
    """Summaries of all quantization tasks in this server process."""
    with task_lock:
        brief = {}
        for tid, body in tasks.items():
            brief[tid] = {
                "status": body.get("status"),
                "message": body.get("message"),
                "output_dir": body.get("output_dir"),
                "progress_percent": body.get("progress_percent"),
            }
        return jsonify({"task_ids": list(tasks.keys()), "tasks": brief})


@app.route("/api/ollama/models")
def ollama_models():
    """
    D1 — list tags from a local Ollama daemon (optional). Does not use quantized HF weights.
    """
    base = OLLAMA_HOST
    url = f"{base}/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            payload = json.loads(resp.read().decode())
        names = []
        for m in payload.get("models") or []:
            n = m.get("name") or m.get("model")
            if n:
                names.append(str(n))
        return jsonify(
            {
                "ok": True,
                "reachable": True,
                "base_url": base,
                "models": names,
                "docs_url": "https://ollama.com",
            }
        )
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as e:
        return jsonify(
            {
                "ok": True,
                "reachable": False,
                "base_url": base,
                "models": [],
                "hint": "Ollama not reachable. If it is installed, start the app; set OLLAMA_HOST if it is not on 127.0.0.1:11434.",
                "error": str(e),
                "docs_url": "https://ollama.com",
            }
        )


@app.route("/api/system")
def system_info():
    return jsonify(gpu_info_dict())


@app.route("/api/steps")
def steps():
    return jsonify({"steps": STEP_DEFS})


@app.route("/api/evaluate", methods=["POST"])
def evaluate_api():
    """
    Run a single short generation on a quantized folder under the project root.
    Body JSON: { "output_dir": "quantized_model_0", "prompt": "...", "max_new_tokens": 40 }
    """
    data = request.get_json(silent=True) or {}
    folder = data.get("output_dir") or data.get("model_path")
    if not folder or not str(folder).strip():
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "output_dir is required (e.g. quantized_model_0 — folder under project root).",
                }
            ),
            400,
        )
    ok, msg, resolved = sanitize_output_dir(
        str(folder).strip(), default_name="quantized_model", project_root=str(PROJECT_ROOT)
    )
    if not ok:
        return jsonify({"ok": False, "error": msg}), 400

    if not os.path.isdir(resolved):
        return jsonify(
            {
                "ok": False,
                "error": f"Folder not found: {folder}. Quantize first or check the name.",
            }
        ), 404

    prompt = (data.get("prompt") or "The future of AI is").strip() or "The future of AI is"
    try:
        max_new_tokens = int(data.get("max_new_tokens", 40))
    except (TypeError, ValueError):
        max_new_tokens = 40
    max_new_tokens = max(1, min(max_new_tokens, 256))
    do_sample = bool(data.get("do_sample", False))

    from evaluate import run_quick_eval

    result = run_quick_eval(
        resolved, prompt=prompt, max_new_tokens=max_new_tokens, do_sample=do_sample
    )
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.route("/api/quantize", methods=["POST"])
def quantize_api():
    data = request.get_json(silent=True) or {}
    model_name = data.get("model_name")
    ok_m, model_name_norm = validate_model_name(model_name)
    if not ok_m:
        return jsonify({"error": model_name_norm}), 400

    try:
        target_size_gb = float(data.get("target_size_gb", 0.2))
    except (TypeError, ValueError):
        return jsonify({"error": "target_size_gb must be a number"}), 400
    if not (0.05 <= target_size_gb <= 640.0):
        return jsonify({"error": "target_size_gb must be between 0.05 and 640"}), 400

    claimed = data.get("original_size_gb")
    if claimed is not None and str(claimed).strip() != "":
        try:
            cg = float(claimed)
            if cg > 0 and target_size_gb > cg + 1e-5:
                return (
                    jsonify(
                        {
                            "error": "target_size_gb cannot exceed the original model footprint "
                            f"({cg:.4f} GB). Lower the budget or pick a larger base model.",
                        }
                    ),
                    400,
                )
        except (TypeError, ValueError):
            pass

    try:
        num_samples = int(data.get("num_samples", 48))
    except (TypeError, ValueError):
        return jsonify({"error": "num_samples must be an integer"}), 400
    num_samples = max(4, min(num_samples, 200))

    default_out = f"quantized_model_{len(tasks)}"
    ok_o, msg_o, resolved_out = sanitize_output_dir(
        data.get("output_dir"), default_out, str(PROJECT_ROOT)
    )
    if not ok_o:
        return jsonify({"error": msg_o}), 400

    if not _acquire_quantize_job():
        return (
            jsonify(
                {
                    "error": "Another quantization job is already running. "
                    "Wait for it to finish, then try again."
                }
            ),
            429,
        )

    task_id = str(len(tasks))
    output_dir_name = os.path.basename(resolved_out)

    gpu = gpu_info_dict()
    _update_task(
        task_id,
        status="running",
        current_step_index=0,
        step_id=STEP_DEFS[0]["id"],
        message="Starting…",
        progress_percent=0,
        detail_line="",
        gpu=gpu,
        steps=STEP_DEFS,
        stats={},
        log_lines=[],
        cancel_requested=False,
    )

    def log_line(msg: str) -> None:
        with task_lock:
            t = tasks.get(task_id, {})
            lines = list(t.get("log_lines") or [])
            lines.append(msg)
            t["log_lines"] = lines[-80:]
            tasks[task_id] = t

    def job() -> None:
        try:
            def _raise_if_cancelled() -> None:
                with task_lock:
                    if tasks.get(task_id, {}).get("cancel_requested"):
                        raise QuantizationCancelled()

            log_line(
                f"Device: {gpu['device']}"
                + (f" ({gpu.get('device_name')})" if gpu.get("device_name") else "")
            )

            _update_task(
                task_id,
                current_step_index=0,
                step_id="load",
                message="Loading model and tokenizer…",
                progress_percent=2,
            )

            try:
                quantizer = AutoQuantizer(model_name_norm)
            except Exception as e:
                log_line(f"Load failed: {e}")
                _update_task(
                    task_id,
                    status="error",
                    error=f"Model load failed: {e}",
                    message="Failed at load",
                )
                return

            pstats = model_parameter_stats(quantizer.model)
            stats_load = _merge_stats(
                pstats,
                {
                    "original_size_gb": round(quantizer.original_size, 4),
                    "model_name": model_name_norm,
                    "dtype": "float16",
                },
            )
            log_line(
                f"Loaded {stats_load['total_parameters']:,} parameters "
                f"({stats_load['quantizable_layers']} quantizable layers)."
            )

            orig_gb = float(quantizer.original_size)
            if target_size_gb > orig_gb + 1e-5:
                log_line(
                    f"Budget {target_size_gb:.4f} GB exceeds original footprint {orig_gb:.4f} GB."
                )
                _update_task(
                    task_id,
                    status="error",
                    error=(
                        f"target_size_gb ({target_size_gb:.4f} GB) cannot exceed the loaded model "
                        f"size ({orig_gb:.4f} GB). Use “Estimate size” and set budget ≤ original."
                    ),
                    message="Budget too large",
                )
                return

            _update_task(
                task_id,
                current_step_index=0,
                step_id="load",
                message="Model ready.",
                progress_percent=12,
                stats=stats_load,
            )

            _update_task(
                task_id,
                current_step_index=1,
                step_id="sensitivity",
                message="Running calibration for sensitivity…",
                progress_percent=15,
                detail_line=f"0 / {num_samples} samples",
            )

            _raise_if_cancelled()

            def sens_cb(cur: int, tot: int, _label: str) -> None:
                _raise_if_cancelled()
                pct = 15 + int(55 * cur / max(tot, 1))
                _update_task(
                    task_id,
                    progress_percent=min(pct, 69),
                    detail_line=f"{cur} / {tot} calibration samples",
                )

            try:
                quantizer.analyze_sensitivity(
                    num_samples=num_samples, progress_callback=sens_cb
                )
            except QuantizationCancelled:
                log_line("Quantization cancelled during sensitivity.")
                _update_task(
                    task_id,
                    status="cancelled",
                    message="Cancelled",
                    detail_line="",
                )
                return
            except Exception as e:
                log_line(f"Sensitivity failed: {e}")
                _update_task(
                    task_id,
                    status="error",
                    error=f"Sensitivity failed: {e}",
                    message="Failed at sensitivity",
                )
                return

            n_layers = len(quantizer.sensitivity_scores or {})
            log_line(f"Sensitivity: scored {n_layers} weight layers.")

            _update_task(
                task_id,
                current_step_index=1,
                step_id="sensitivity",
                message="Sensitivity map complete.",
                progress_percent=70,
                detail_line="",
                stats=_merge_stats(
                    stats_load,
                    {
                        "sensitivity_layers": n_layers,
                    },
                ),
            )

            _update_task(
                task_id,
                current_step_index=2,
                step_id="allocate",
                message="Solving bit allocation under budget…",
                progress_percent=72,
            )

            _raise_if_cancelled()

            try:
                cfg_result = quantizer.create_config(target_size_gb)
            except Exception as e:
                log_line(f"Allocation failed: {e}")
                _update_task(
                    task_id,
                    status="error",
                    error=f"Allocation failed: {e}",
                    message="Failed at allocation",
                )
                return

            cfg = cfg_result["config"]
            assignments = quantizer.bit_assignments or {}
            avg_bits = weighted_average_bits_for_quantizable(
                quantizer.model, assignments
            )
            dist = _safe_dist(cfg.get("bit_distribution", {}))
            n_sub16 = sum(1 for b in assignments.values() if int(b) < 16)

            log_line(
                f"Allocation: target ≤ {target_size_gb:.2f} GB → "
                f"expected ~{cfg_result['expected_size_gb']:.3f} GB, "
                f"{n_sub16} layers below FP16."
            )

            _update_task(
                task_id,
                current_step_index=2,
                step_id="allocate",
                message="Bit plan ready.",
                progress_percent=78,
                stats=_merge_stats(
                    stats_load,
                    {
                        "target_size_gb": target_size_gb,
                        "expected_size_gb": round(cfg_result["expected_size_gb"], 4),
                        "compression_ratio_expected": round(
                            cfg_result["compression_ratio"], 3
                        ),
                        "bit_distribution": dist,
                        "weighted_avg_bits": round(avg_bits, 2),
                        "layers_quantized_planned": n_sub16,
                    },
                ),
            )

            os.makedirs(resolved_out, exist_ok=True)
            cfg_path = os.path.join(resolved_out, "_run_config.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

            n_quant = sum(1 for b in assignments.values() if int(b) < 16)

            _update_task(
                task_id,
                current_step_index=3,
                step_id="quantize",
                message="Replacing layers and saving checkpoint…",
                progress_percent=80,
                detail_line="",
            )

            def q_cb(cur: int, tot: int, layer_name: str) -> None:
                _raise_if_cancelled()
                sub = f"{cur} / {tot}: {layer_name}"
                log_line(sub)
                pct = 80 + int(18 * cur / max(tot, 1))
                _update_task(
                    task_id,
                    progress_percent=min(pct, 98),
                    detail_line=sub,
                )

            _raise_if_cancelled()

            def _cancel_requested() -> bool:
                with task_lock:
                    return bool(tasks.get(task_id, {}).get("cancel_requested"))

            try:
                quantizer.quantize(
                    cfg_path,
                    resolved_out,
                    progress_callback=q_cb if n_quant else None,
                    cancel_callback=_cancel_requested,
                )
            except QuantizationCancelled:
                log_line("Quantization cancelled during layer replacement.")
                _update_task(
                    task_id,
                    status="cancelled",
                    message="Cancelled",
                    detail_line="",
                )
                return
            except Exception as e:
                log_line(f"Quantize/save failed: {e}")
                _update_task(
                    task_id,
                    status="error",
                    error=f"Quantize failed: {e}",
                    message="Failed at quantize",
                )
                return

            report = quantizer.get_report()
            final_dist = _safe_dist(report.get("bit_distribution", {}))

            log_line(
                f"Done. Footprint {report['original_size_gb']:.3f} → "
                f"{report['quantized_size_gb']:.3f} GB "
                f"({report['compression_ratio']:.2f}×)."
            )

            eval_cmd = f'python evaluate.py "{output_dir_name}"'
            api_eval = (
                f'POST /api/evaluate with JSON '
                f'{{"output_dir": "{output_dir_name}", "prompt": "Your text", "max_new_tokens": 40}}'
            )
            merged_final = _merge_stats(
                stats_load,
                {
                    "target_size_gb": target_size_gb,
                    "expected_size_gb": round(cfg_result["expected_size_gb"], 4),
                    "quantized_size_gb": round(report["quantized_size_gb"], 4),
                    "original_size_gb": round(report["original_size_gb"], 4),
                    "compression_ratio": round(report["compression_ratio"], 3),
                    "space_saved_percent": round(report["space_saved_percent"], 1),
                    "bit_distribution": final_dist,
                    "weighted_avg_bits": round(avg_bits, 2),
                    "total_parameters": pstats["total_parameters"],
                    "quantizable_parameter_count": pstats[
                        "quantizable_parameter_count"
                    ],
                    "state_dict_warnings": report.get("state_dict_warnings", {}),
                    "chart_data": _chart_payload(
                        {
                            "bit_distribution": final_dist,
                            "original_size_gb": round(report["original_size_gb"], 4),
                            "expected_size_gb": round(cfg_result["expected_size_gb"], 4),
                            "quantized_size_gb": round(report["quantized_size_gb"], 4),
                        }
                    ),
                },
            )

            metrics_path = os.path.join(resolved_out, "_metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as mf:
                json.dump(
                    {
                        "model_name": model_name_norm,
                        "target_size_gb": target_size_gb,
                        "actual_size_gb": round(float(report["quantized_size_gb"]), 6),
                        "num_samples": num_samples,
                        "bit_distribution": final_dist,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z"),
                        "weighted_avg_bits": round(float(avg_bits), 4),
                    },
                    mf,
                    indent=2,
                )

            _update_task(
                task_id,
                status="done",
                current_step_index=3,
                step_id="quantize",
                message="Quantization complete.",
                progress_percent=100,
                detail_line=output_dir_name,
                output_dir=output_dir_name,
                report=report,
                evaluate_cli=eval_cmd,
                evaluate_api_hint=api_eval,
                lab_url=f"/lab?output_dir={output_dir_name}",
                stats=merged_final,
            )
        except QuantizationCancelled:
            log_line("Quantization cancelled.")
            _update_task(
                task_id,
                status="cancelled",
                message="Cancelled",
                detail_line="",
            )
        except Exception as e:
            log_line(f"Error: {e}")
            _update_task(task_id, status="error", error=str(e), message="Failed")
        finally:
            _release_quantize_job()

    threading.Thread(target=job, daemon=True).start()
    return jsonify(
        {
            "task_id": task_id,
            "output_dir": output_dir_name,
            "gpu": gpu,
            "message": "Job started. Poll GET /api/status/<task_id> or watch the UI.",
        }
    )


@app.route("/api/cancel/<task_id>", methods=["POST"])
def cancel_task(task_id: str):
    with task_lock:
        if task_id in tasks:
            tasks[task_id]["cancel_requested"] = True
            return jsonify({"status": "cancel_requested"})
    return jsonify({"error": "task not found"}), 404


@app.route("/api/status/<task_id>")
def status(task_id: str):
    with task_lock:
        t = tasks.get(task_id)
        if t is None:
            return jsonify({}), 404
        return jsonify(copy.deepcopy(t))


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    url = f"http://{host}:{port}"

    info = gpu_info_dict()
    print(f"Open {url} for the AutoQuant dashboard.")
    if info["cuda_available"]:
        print(f"GPU: {info.get('device_name')}")
    else:
        print("CUDA not active — quantization will use CPU (slower).")
        if info.get("nvidia_smi_summary"):
            print(f"Driver probe: {info['nvidia_smi_summary']}")
        if info.get("cuda_hint"):
            print(info["cuda_hint"])

    use_flask_dev = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    if use_flask_dev:
        app.run(debug=True, host=host, port=port, threaded=True, use_reloader=False)
    else:
        try:
            from waitress import serve

            print(
                "Serving with Waitress (set FLASK_DEBUG=1 to use Flask's dev server instead)."
            )
            serve(app, host=host, port=port, threads=6)
        except ImportError:
            print("waitress not installed; falling back to Flask. pip install waitress")
            app.run(debug=False, host=host, port=port, threaded=True)
