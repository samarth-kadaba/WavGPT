"""Convenience imports for checkpoint loading."""

from wavgpt.utils.save_checkpoint import (
    load_checkpoint_for_training,
    load_checkpoint_for_inference,
)

__all__ = [
    'load_checkpoint_for_training',
    'load_checkpoint_for_inference',
]
