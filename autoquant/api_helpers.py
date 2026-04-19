"""
Shared validation for Flask API and CLI-style callers.
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple

# Hugging Face repo id: one optional "org/name" segment; no path traversal.
_PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")


def validate_model_name(model_name: Optional[str]) -> Tuple[bool, str]:
    if not model_name or not str(model_name).strip():
        return False, "model_name is required"
    s = str(model_name).strip()
    if ".." in s or "\\" in s:
        return False, "model_name must not contain .. or backslashes"
    if s.count("/") > 1:
        return (
            False,
            "Invalid model id: use gpt2 or a single org/model form (at most one '/').",
        )
    parts = s.split("/")
    for p in parts:
        if not p or not _PART_RE.match(p):
            return (
                False,
                "Invalid model id: use a Hugging Face id like gpt2 or org/model-name "
                "(letters, digits, -, _, . only).",
            )
    if len(s) > 200:
        return False, "model_name is too long"
    return True, s


def sanitize_output_dir(
    output_dir: Optional[str], default_name: str, project_root: str
) -> Tuple[bool, str, str]:
    """
    Resolve output directory under project_root only (no absolute paths, no ..).
    Returns (ok, message_or_path, resolved_abs_path).
    """
    name = (output_dir or default_name).strip()
    if not name:
        return False, "output_dir is required (single folder name under the project)", ""
    if ".." in name or name.startswith(("/", "\\")):
        return False, "output_dir must be a simple folder name under the project (no .. or absolute paths)", ""
    if any(c in name for c in (":", "<", ">", "|", "*", "?", '"')):
        return False, "output_dir contains invalid characters", ""
    parts = [p for p in name.replace("\\", "/").split("/") if p]
    if len(parts) != 1:
        return False, "output_dir must be a single folder name (no nested paths)", ""
    if not re.match(r"^[A-Za-z0-9_.\-]+$", parts[0]):
        return False, "output_dir: use only letters, digits, underscore, hyphen, dot", ""
    root_abs = Path(project_root).resolve()
    resolved = (root_abs / parts[0]).resolve()
    try:
        resolved.relative_to(root_abs)
    except ValueError:
        return False, "output_dir resolves outside project", ""
    return True, parts[0], str(resolved)
