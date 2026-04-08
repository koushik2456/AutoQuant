"""
AutoQuant - Automated Mixed-Precision Quantization for LLMs
Implements multi-metric sensitivity analysis with 4 complementary methods
"""

__version__ = "2.0.0"

from .quantizer import AutoQuantizer
from .sensitivity import MultiMetricSensitivityAnalyzer

__all__ = ['AutoQuantizer', 'MultiMetricSensitivityAnalyzer']