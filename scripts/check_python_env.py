"""
Print which Python is running and whether core deps match the project.
Use this when the IDE shows "import could not be resolved" or CUDA mismatches.

Run from project root:
  python scripts/check_python_env.py
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    print("=== Python interpreter ===")
    print(sys.executable)
    print("version:", sys.version.split()[0])

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("\n=== Project root (expected repo) ===")
    print(root)
    print("cwd:", os.getcwd())

    print("\n=== Packages ===")
    try:
        import torch

        print("torch:", torch.__version__)
        print("torch file:", getattr(torch, "__file__", "?"))
        print("cuda.is_available():", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("device:", torch.cuda.get_device_name(0))
        try:
            b = getattr(torch.version, "cuda", None)
            if b:
                print("torch.version.cuda:", b)
        except Exception:
            pass
    except ImportError as e:
        print("torch: NOT IMPORTABLE", e)

    for mod in ("transformers", "flask", "waitress"):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            print(f"{mod}: {ver}")
        except ImportError:
            print(f"{mod}: NOT INSTALLED")

    print("\n=== IDE hint ===")
    print(
        "In VS Code / Cursor: Command Palette → Python: Select Interpreter → pick the "
        "same path as sys.executable above (often .venv\\Scripts\\python.exe)."
    )
    print("If pip.exe is blocked by policy, use: python -m pip install -r requirements.txt")


if __name__ == "__main__":
    main()
