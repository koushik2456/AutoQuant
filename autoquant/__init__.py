import sys

from .quantizer import AutoQuantizer, DynamicQuantizedLinear

__all__ = ["AutoQuantizer", "DynamicQuantizedLinear"]
__version__ = "0.1.0"

# Windows consoles often default to cp1252; progress messages use Unicode.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (OSError, ValueError, AttributeError):
            pass
