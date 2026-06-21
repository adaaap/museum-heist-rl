"""Experiment 5: try different softmax temperatures (``tau``).

A low ``tau`` makes the guard too predictable, so the thief beats it. A high
``tau`` makes it watch everywhere with no focus. The best score is in the
middle, which shows that some randomness helps.
"""
from __future__ import annotations

import os

import numpy as np

from experiments._common import SWEEP_SEEDS, train_and_eval
from museum_heist.envs.topology import open_grid
from museum_heist.utils.plotting import plot_sweep

TAUS = [0.25, 0.5, 1.0, 2.0, 4.0]


def run() -> None:
    topology = open_grid(3, 3)
    catch_mean, catch_std, entropy_mean, entropy_std = [], [], [], []
    for tau in TAUS:
        results = [train_and_eval(topology, tau=tau, beta=1.0, seed=s) for s in SWEEP_SEEDS]
        catches = [r["catch_rate_stochastic"] for r in results]
        entropies = [r["entropy"] for r in results]
        catch_mean.append(np.mean(catches)); catch_std.append(np.std(catches))
        entropy_mean.append(np.mean(entropies)); entropy_std.append(np.std(entropies))
        print(f"  tau={tau:<4} catch_rate={np.mean(catches):.3f}  entropy={np.mean(entropies):.2f}")

    figure = plot_sweep(TAUS, catch_mean, catch_std, entropy_mean, entropy_std,
                        xlabel="softmax temperature (tau)", title="Temperature sweep (open 3x3)")
    os.makedirs("figures", exist_ok=True)
    figure.savefig("figures/exp05_temperature_sweep.png", dpi=120)
    print("saved figures/exp05_temperature_sweep.png")


if __name__ == "__main__":
    run()
