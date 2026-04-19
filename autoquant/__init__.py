import sys

from . import config
from .quantizer import (
    AutoQuantizer,
    DynamicQuantizedLinear,
    QuantizationCancelled,
    infer_dynamic_quant_group_size,
)

__all__ = [
    "AutoQuantizer",
    "DynamicQuantizedLinear",
    "QuantizationCancelled",
    "infer_dynamic_quant_group_size",
    "config",
]
__version__ = "0.1.0"

# Windows consoles often default to cp1252; progress messages use Unicode.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (OSError, ValueError, AttributeError):
            pass
