"""Training utilities for Context Extension.

This package provides training infrastructure for learning context extension
via GRPO (Group Relative Policy Optimization).
"""

from wavgpt.training.grpo import GRPOTrainer, create_grpo_trainer

__all__ = [
    "GRPOTrainer",
    "create_grpo_trainer",
]
