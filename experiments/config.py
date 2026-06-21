"""The settings shared by all experiments (saved with every figure)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Default hyperparameters for the base (stateless) Museum Heist task."""

    grid_height: int = 3
    grid_width: int = 3
    beta: float = 1.0        # thief prudence
    tau: float = 1.0         # softmax temperature
    alpha: float = 0.2       # learning rate
    gamma: float = 0.99      # discount
    batch_size: int = 16     # trajectories per gradient step
    n_episodes: int = 8000   # training episodes per seed
    max_steps: int = 50      # per-episode horizon (harness truncation)
