"""Experiment 6: try different thief prudence values (``beta``).

``beta`` is how much the thief avoids rooms the guard just watched. With
``beta=0`` the thief is predictable and the guard can focus. A high ``beta``
makes the thief dodge cameras, so the guard has to spread its watching out.
"""
from __future__ import annotations

import os

import numpy as np

from experiments._common import SWEEP_SEEDS, train_and_eval
from museum_heist.envs.topology import open_grid
from museum_heist.utils.plotting import plot_sweep

BETAS = [0.0, 0.5, 1.0, 2.0, 4.0]


def run() -> None:
    topology = open_grid(3, 3)
    catch_mean, catch_std, entropy_mean, entropy_std = [], [], [], []
    for beta in BETAS:
        results = [train_and_eval(topology, tau=1.0, beta=beta, seed=s) for s in SWEEP_SEEDS]
        catches = [r["catch_rate_stochastic"] for r in results]
        entropies = [r["entropy"] for r in results]
        catch_mean.append(np.mean(catches)); catch_std.append(np.std(catches))
        entropy_mean.append(np.mean(entropies)); entropy_std.append(np.std(entropies))
        print(f"  beta={beta:<4} catch_rate={np.mean(catches):.3f}  entropy={np.mean(entropies):.2f}")

    figure = plot_sweep(BETAS, catch_mean, catch_std, entropy_mean, entropy_std,
                        xlabel="thief prudence (beta)", title="Prudence sweep (open 3x3)")
    os.makedirs("figures", exist_ok=True)
    figure.savefig("figures/exp06_prudence_sweep.png", dpi=120)
    print("saved figures/exp06_prudence_sweep.png")


if __name__ == "__main__":
    run()
