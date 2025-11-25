"""Utility functions for WavGPT analysis."""

from pathlib import Path


# Default output directory for all analysis plots
DEFAULT_ANALYSIS_DIR = "analysis_outputs"


def ensure_analysis_dir(output_dir: str = None) -> Path:
    """
    Create and return path to analysis output directory.
    
    Args:
        output_dir: Custom output directory path (optional)
        
    Returns:
        Path object to the output directory
    """
    if output_dir is None:
        output_dir = DEFAULT_ANALYSIS_DIR
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

