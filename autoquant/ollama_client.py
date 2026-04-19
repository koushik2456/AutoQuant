"""
HTTP helpers for a local Ollama daemon (pull, tags, chat).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from autoquant.config import OLLAMA_HOST


def _base_url() -> str:
    return OLLAMA_HOST


def ollama_tags() -> Tuple[bool, List[str], Optional[str]]:
    """Return (reachable, model_names, error_message)."""
    url = f"{_base_url()}/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            payload = json.loads(resp.read().decode())
        names: List[str] = []
        for m in payload.get("models") or []:
            n = m.get("name") or m.get("model")
            if n:
                names.append(str(n))
        return True, names, None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as e:
        return False, [], str(e)


_OLLAMA_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:\-]{0,190}$")


def validate_ollama_model_name(name: Optional[str]) -> Tuple[bool, str]:
    if not name or not str(name).strip():
        return False, "model name is required"
    s = str(name).strip()
    if ".." in s or "/" in s or "\\" in s:
        return False, "invalid Ollama model name"
    if not _OLLAMA_NAME_RE.match(s):
        return False, "invalid Ollama model name"
    return True, s


def ollama_pull_cli(model_name: str, timeout_sec: float = 7200.0) -> Dict[str, Any]:
    """Run `ollama pull <name>` — reliable on Windows when the CLI is on PATH."""
    ok, norm = validate_ollama_model_name(model_name)
    if not ok:
        return {"ok": False, "error": norm}
    kwargs: Dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "timeout": timeout_sec,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    try:
        r = subprocess.run(["ollama", "pull", norm], **kwargs)
        out = (r.stdout or "") + ("\n" + r.stderr if r.stderr else "")
        if r.returncode != 0:
            return {
                "ok": False,
                "error": out.strip() or f"exit code {r.returncode}",
                "via": "cli",
            }
        return {"ok": True, "via": "cli", "log_tail": out.strip()[-4000:]}
    except FileNotFoundError:
        return {"ok": False, "error": "ollama CLI not found on PATH", "via": "cli"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "ollama pull timed out", "via": "cli"}
    except OSError as e:
        return {"ok": False, "error": str(e), "via": "cli"}


def ollama_pull(
    model_name: str, stream: bool = False, timeout_sec: float = 7200.0
) -> Dict[str, Any]:
    """
    Try `ollama pull` first; on failure fall back to POST /api/pull on the daemon.
    """
    cli = ollama_pull_cli(model_name, timeout_sec=timeout_sec)
    if cli.get("ok"):
        return cli

    ok, norm = validate_ollama_model_name(model_name)
    if not ok:
        return {"ok": False, "error": norm}
    url = f"{_base_url()}/api/pull"
    body = json.dumps({"name": norm, "stream": stream}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        if stream:
            lines = [ln for ln in raw.splitlines() if ln.strip()]
            last: Dict[str, Any] = {}
            for ln in lines:
                try:
                    last = json.loads(ln)
                except json.JSONDecodeError:
                    continue
            return {"ok": True, "stream": True, "last": last, "raw_lines": len(lines), "via": "http"}
        try:
            payload = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            payload = {"raw": raw[:2000]}
        return {"ok": True, "stream": False, "response": payload, "via": "http"}
    except urllib.error.HTTPError as e:
        err: Dict[str, Any] = {
            "ok": False,
            "error": f"HTTP {e.code}: {e.reason}",
            "via": "http",
        }
        if cli.get("error"):
            err["cli_error"] = cli.get("error")
        return err
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        err = {"ok": False, "error": str(e), "via": "http"}
        if cli.get("error"):
            err["cli_error"] = cli.get("error")
        return err


def ollama_chat(
    model_name: str,
    messages: List[Dict[str, str]],
    stream: bool = False,
    timeout_sec: float = 300.0,
) -> Dict[str, Any]:
    """POST /api/chat — OpenAI-style messages: [{role, content}, ...]."""
    ok, norm = validate_ollama_model_name(model_name)
    if not ok:
        return {"ok": False, "error": norm}
    url = f"{_base_url()}/api/chat"
    body = json.dumps(
        {
            "model": norm,
            "messages": messages,
            "stream": stream,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        payload = json.loads(raw) if raw.strip() else {}
        msg = payload.get("message") or {}
        content = msg.get("content") or ""
        return {"ok": True, "content": content, "raw": payload}
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        return {"ok": False, "error": f"HTTP {e.code}: {err_body or e.reason}"}
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        return {"ok": False, "error": str(e)}


def ollama_has_model(pulled_name: str, tags: List[str]) -> bool:
    """Match 'llama3.2' to 'llama3.2:latest' etc."""
    if not pulled_name:
        return False
    base = pulled_name.split(":")[0].strip().lower()
    for t in tags:
        tb = t.split(":")[0].strip().lower()
        if tb == base or t == pulled_name:
            return True
    return False
