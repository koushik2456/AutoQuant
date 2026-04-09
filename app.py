import copy
import json
import os
import threading
from typing import Any, Dict

import torch
from flask import Flask, jsonify, render_template, request

from autoquant import AutoQuantizer
from autoquant.utils import (
    gpu_info_dict,
    model_parameter_stats,
    weighted_average_bits_for_quantizable,
)

app = Flask(__name__)
tasks: Dict[str, Dict[str, Any]] = {}
task_lock = threading.Lock()

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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/system")
def system_info():
    return jsonify(gpu_info_dict())


@app.route("/api/steps")
def steps():
    return jsonify({"steps": STEP_DEFS})


@app.route("/api/quantize", methods=["POST"])
def quantize_api():
    data = request.get_json(silent=True) or {}
    model_name = data.get("model_name")
    if not model_name:
        return jsonify({"error": "model_name is required"}), 400

    task_id = str(len(tasks))
    target_size_gb = float(data.get("target_size_gb", 0.2))
    num_samples = max(4, int(data.get("num_samples", 48)))
    output_dir = data.get("output_dir") or f"quantized_model_{task_id}"

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
            log_line(f"Device: {gpu['device']}" + (f" ({gpu.get('device_name')})" if gpu.get("device_name") else ""))

            _update_task(
                task_id,
                current_step_index=0,
                step_id="load",
                message="Loading model and tokenizer…",
                progress_percent=2,
            )

            quantizer = AutoQuantizer(model_name)

            pstats = model_parameter_stats(quantizer.model)
            stats_load = _merge_stats(
                pstats,
                {
                    "original_size_gb": round(quantizer.original_size, 4),
                    "model_name": model_name,
                    "dtype": "float16",
                },
            )
            log_line(
                f"Loaded {stats_load['total_parameters']:,} parameters "
                f"({stats_load['quantizable_layers']} quantizable layers)."
            )

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

            def sens_cb(cur: int, tot: int, _label: str) -> None:
                pct = 15 + int(55 * cur / max(tot, 1))
                _update_task(
                    task_id,
                    progress_percent=min(pct, 69),
                    detail_line=f"{cur} / {tot} calibration samples",
                )

            quantizer.analyze_sensitivity(
                num_samples=num_samples, progress_callback=sens_cb
            )

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

            cfg_result = quantizer.create_config(target_size_gb)
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

            os.makedirs(output_dir, exist_ok=True)
            cfg_path = os.path.join(output_dir, "_run_config.json")
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
                sub = f"{cur} / {tot}: {layer_name}"
                log_line(sub)
                pct = 80 + int(18 * cur / max(tot, 1))
                _update_task(
                    task_id,
                    progress_percent=min(pct, 98),
                    detail_line=sub,
                )

            quantizer.quantize(
                cfg_path, output_dir, progress_callback=q_cb if n_quant else None
            )

            report = quantizer.get_report()
            final_dist = _safe_dist(report.get("bit_distribution", {}))

            log_line(
                f"Done. Footprint {report['original_size_gb']:.3f} → "
                f"{report['quantized_size_gb']:.3f} GB "
                f"({report['compression_ratio']:.2f}×)."
            )

            _update_task(
                task_id,
                status="done",
                current_step_index=3,
                step_id="quantize",
                message="Quantization complete.",
                progress_percent=100,
                detail_line=output_dir,
                output_dir=output_dir,
                report=report,
                stats=_merge_stats(
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
                    },
                ),
            )
        except Exception as e:
            log_line(f"Error: {e}")
            _update_task(task_id, status="error", error=str(e), message="Failed")

    threading.Thread(target=job, daemon=True).start()
    return jsonify({"task_id": task_id, "output_dir": output_dir, "gpu": gpu})


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
        if info.get("cuda_hint"):
            print(info["cuda_hint"])

    # Default: Waitress (no "development server" warning). Set FLASK_DEBUG=1 for Flask's reloader.
    use_flask_dev = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    if use_flask_dev:
        app.run(debug=True, host=host, port=port, threaded=True, use_reloader=False)
    else:
        try:
            from waitress import serve

            print("Serving with Waitress (set FLASK_DEBUG=1 to use Flask's dev server instead).")
            serve(app, host=host, port=port, threads=6)
        except ImportError:
            print("waitress not installed; falling back to Flask. pip install waitress")
            app.run(debug=False, host=host, port=port, threaded=True)
