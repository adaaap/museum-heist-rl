"""Plotting helpers that return matplotlib figures.

Each function builds a figure and returns it instead of calling ``plt.show()``,
so it works in both notebooks and scripts. It uses the Agg backend, so it also
runs with no screen (on a server or in CI).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend; a notebook can override before importing

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)
import numpy as np  # noqa: E402


def plot_learning_curves(curves, labels, *, x=None, xlabel: str = "episode",
                         ylabel: str = "return", title: str | None = None):
    """Plot one mean line with a +/-1 std band per curve.

    Parameters
    ----------
    curves:
        Iterable of 2D arrays shaped ``(n_seeds, n_points)``, one per series.
    labels:
        Series labels, aligned with ``curves``.
    x:
        Shared x-axis values; defaults to ``arange(n_points)``.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for data, label in zip(curves, labels):
        data = np.asarray(data, dtype=float)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        xs = np.arange(mean.size) if x is None else np.asarray(x)
        ax.plot(xs, mean, label=label)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_room_probability_heatmap(probs: np.ndarray, topology, *,
                                  title: str = "Learned watch distribution"):
    """Colour each room cell on the museum grid by its watch probability."""
    grid = np.full((topology.height, topology.width), np.nan)
    for room, (row, col) in enumerate(topology.rooms):
        grid[row, col] = probs[room]
    fig, ax = plt.subplots(figsize=(5, 5))
    image = ax.imshow(grid, cmap="viridis", origin="upper")
    fig.colorbar(image, ax=ax, label="P(watch room)")
    for room, (row, col) in enumerate(topology.rooms):
        ax.text(col, row, f"{probs[room]:.2f}", ha="center", va="center",
                color="white", fontsize=8)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_sweep(x, catch_mean, catch_std, entropy_mean, entropy_std, *,
               xlabel: str, title: str | None = None):
    """Twin-axis sweep: catch rate (left) and policy entropy (right) vs a parameter."""
    x = np.asarray(x, dtype=float)
    fig, ax_catch = plt.subplots(figsize=(7, 4))
    ax_catch.errorbar(x, catch_mean, yerr=catch_std, marker="o", capsize=3, color="tab:blue")
    ax_catch.set_xlabel(xlabel)
    ax_catch.set_ylabel("catch rate", color="tab:blue")
    ax_catch.tick_params(axis="y", labelcolor="tab:blue")
    ax_entropy = ax_catch.twinx()
    ax_entropy.errorbar(x, entropy_mean, yerr=entropy_std, marker="s", capsize=3, color="tab:red")
    ax_entropy.set_ylabel("policy entropy (nats)", color="tab:red")
    ax_entropy.tick_params(axis="y", labelcolor="tab:red")
    if title:
        ax_catch.set_title(title)
    ax_catch.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_grouped_bars(categories, series, *, ylabel: str, title: str | None = None,
                      errors=None):
    """Grouped bar chart; ``series`` maps a label to one value per category.

    ``errors`` (optional) maps the same labels to per-category error-bar sizes
    (for example std across seeds), drawn as symmetric whiskers.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    n_categories = len(categories)
    n_series = len(series)
    bar_width = 0.8 / n_series
    positions = np.arange(n_categories)
    for i, (label, values) in enumerate(series.items()):
        offset = i * bar_width - 0.4 + bar_width / 2
        yerr = errors.get(label) if errors else None
        ax.bar(positions + offset, values, bar_width, label=label, yerr=yerr, capsize=4)
    ax.set_xticks(positions)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
