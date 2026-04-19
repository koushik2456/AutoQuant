"""
Smoke verification: programmatic GPT-2 quantization, then Flask /api/quantize
for _metrics.json, then a cancel race on a second job.
Run from repo root: python scripts/e2e_verify_quantization.py
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    _, p = s.getsockname()
    s.close()
    return int(p)


def programmatic_quantize() -> Path:
    from autoquant import AutoQuantizer

    out = ROOT / "quantized_e2e_smoke"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    q = AutoQuantizer("gpt2")
    q.analyze_sensitivity(num_samples=4)
    r = q.create_config(0.12)
    cfg_path = out / "_run_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(r["config"], f, indent=2)
    q.quantize(str(cfg_path), str(out))
    q.get_report()

    qc_path = out / "quantization_config.json"
    assert qc_path.is_file(), "missing quantization_config.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    assert qc.get("schema_version") == "1.1", qc
    assert "dynamic_quant_group_size" in qc
    assert (out / "pytorch_model.bin").is_file()
    print("[ok] programmatic quantize:", out)
    print("     schema_version:", qc.get("schema_version"))
    return out


def flask_metrics_and_cancel(port: int) -> None:
    import app as flask_app

    def _run() -> None:
        flask_app.app.run(
            host="127.0.0.1",
            port=port,
            threaded=True,
            use_reloader=False,
            debug=False,
        )

    th = threading.Thread(target=_run, daemon=True)
    th.start()

    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            urllib.request.urlopen(base + "/api/health", timeout=1.0)
            break
        except (urllib.error.URLError, OSError):
            time.sleep(0.15)
    else:
        raise RuntimeError("Flask did not become ready")

    out_name = "quantized_flask_e2e"
    out_dir = ROOT / out_name
    if out_dir.exists():
        shutil.rmtree(out_dir)

    body = json.dumps(
        {
            "model_name": "gpt2",
            "target_size_gb": 0.12,
            "num_samples": 4,
            "output_dir": out_name,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/api/quantize",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
    task_id = resp["task_id"]
    print("[info] flask quantize task_id=", task_id)

    deadline = time.time() + 900.0
    status = {}
    while time.time() < deadline:
        st = json.loads(
            urllib.request.urlopen(f"{base}/api/status/{task_id}", timeout=30)
            .read()
            .decode()
        )
        status = st
        s = st.get("status")
        if s in ("done", "error", "cancelled"):
            break
        time.sleep(1.5)

    assert status.get("status") == "done", status
    metrics_path = out_dir / "_metrics.json"
    assert metrics_path.is_file(), f"missing _metrics.json: {metrics_path}"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for k in (
        "model_name",
        "target_size_gb",
        "actual_size_gb",
        "num_samples",
        "bit_distribution",
        "timestamp",
        "weighted_avg_bits",
    ):
        assert k in metrics, metrics
    print("[ok] flask _metrics.json keys:", list(metrics.keys()))

    # Cancel race: long sensitivity, cancel early
    out_cancel = ROOT / "quantized_flask_cancel"
    if out_cancel.exists():
        shutil.rmtree(out_cancel)
    body2 = json.dumps(
        {
            "model_name": "gpt2",
            "target_size_gb": 0.12,
            "num_samples": 120,
            "output_dir": "quantized_flask_cancel",
        }
    ).encode("utf-8")
    req2 = urllib.request.Request(
        f"{base}/api/quantize",
        data=body2,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp2 = json.loads(urllib.request.urlopen(req2, timeout=30).read().decode())
    tid2 = resp2["task_id"]
    time.sleep(0.4)
    cancel_req = urllib.request.Request(
        f"{base}/api/cancel/{tid2}",
        method="POST",
    )
    urllib.request.urlopen(cancel_req, timeout=10).read()

    deadline2 = time.time() + 120.0
    st2 = {}
    while time.time() < deadline2:
        st2 = json.loads(
            urllib.request.urlopen(f"{base}/api/status/{tid2}", timeout=30)
            .read()
            .decode()
        )
        if st2.get("status") in ("cancelled", "done", "error"):
            break
        time.sleep(0.4)

    assert st2.get("status") == "cancelled", st2
    print("[ok] cancel flow status=cancelled task", tid2)


def main() -> None:
    programmatic_quantize()
    port = _free_port()
    flask_metrics_and_cancel(port)
    print("All e2e checks passed.")


if __name__ == "__main__":
    main()
