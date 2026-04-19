"""
Environment-driven defaults for AutoQuant (no new runtime dependencies).
"""

import os

OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
HF_HOME: str | None = os.environ.get("HF_HOME")
DEFAULT_OUTPUT_ROOT: str = os.environ.get("AUTOQUANT_OUTPUT_ROOT", ".")
INT16_UPGRADE_THRESHOLD: float = float(
    os.environ.get("AUTOQUANT_INT16_THRESHOLD", "0.75")
)
DEFAULT_GROUP_SIZE: int = int(os.environ.get("AUTOQUANT_GROUP_SIZE", "128"))
