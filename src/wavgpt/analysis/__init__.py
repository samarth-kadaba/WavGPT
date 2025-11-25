"""WavGPT Analysis Module - Tools for understanding learned frequency filtering.

This module provides comprehensive visualization and analysis tools for
understanding how WavGPT models learn to compress hidden states using
wavelet-based frequency filtering.

Main Components:
- analyze_frequency_importance: Visualize learned frequency band priorities
- analyze_coefficient_selection: Analyze which coefficients are kept/dropped
- analyze_frequency_content: Analyze frequency preservation in reconstructions
- full_filter_analysis: Run complete analysis suite
"""

from .utils import ensure_analysis_dir, DEFAULT_ANALYSIS_DIR
from .frequency_importance import analyze_frequency_importance
from .coefficient_selection import analyze_coefficient_selection
from .frequency_content import analyze_frequency_content
from .runner import full_filter_analysis

__all__ = [
    # Utilities
    "ensure_analysis_dir",
    "DEFAULT_ANALYSIS_DIR",
    
    # Individual analysis functions
    "analyze_frequency_importance",
    "analyze_coefficient_selection",
    "analyze_frequency_content",
    
    # Full analysis runner
    "full_filter_analysis",
]

