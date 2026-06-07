"""
RGB - Refined Gradient Boosting Feature Selection

A sophisticated feature selection method for high-dimensional biological data
with integrated MSA preprocessing capabilities.
"""

__version__ = "0.1.0"
__author__ = "Qiuyi Li"
__license__ = "MIT"

from rgb.feature_selection import RefinedGradientBoosting, Config
from rgb.msa_preprocessor import MSAPreprocessor

__all__ = [
    "RefinedGradientBoosting",
    "Config",
    "MSAPreprocessor",
]
