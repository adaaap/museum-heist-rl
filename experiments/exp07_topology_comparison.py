"""Experiment 7: an open 3x3 museum vs the same museum with walls.

It trains the guard on both and compares a random (stochastic) watch with a
fixed (greedy) one. On the open museum the random watch wins, which shows the
game needs randomness. A wall forces one doorway, and then the fixed watch is
enough.
"""
from __future__ import annotations

import os

import numpy as np

from experiments._common import SWEEP_SEEDS, train_and_eval
from museum_heist.envs.topology import grid_with_walls, open_grid
from museum_heist.utils.plotting import plot_grouped_bars

# A vertical interior wall across the 3x3 square, with one gap on the bottom
# row, so the left column reaches the rest through only one door (a chokepoint).
WALLS = [((0, 0), (0, 1)), ((1, 0), (1, 1))]

TOPOLOGIES = {
    "open square 3x3": open_grid(3, 3),
    "walled square 3x3": grid_with_walls(3, 3, walls=WALLS),
}


def run() -> None:
    names, stochastic, greedy, entropy, s_err, g_err = [], [], [], [], [], []
    for name, topology in TOPOLOGIES.items():
        results = [train_and_eval(topology, tau=1.0, beta=1.0, seed=s) for s in SWEEP_SEEDS]
        sv = [r["catch_rate_stochastic"] for r in results]
        gv = [r["catch_rate_greedy"] for r in results]
        s, g = float(np.mean(sv)), float(np.mean(gv))
        e = float(np.mean([r["entropy"] for r in results]))
        names.append(name); stochastic.append(s); greedy.append(g); entropy.append(e)
        s_err.append(float(np.std(sv))); g_err.append(float(np.std(gv)))
        print(f"  {name:<18} catch stochastic={s:.3f}±{np.std(sv):.3f}  "
              f"greedy={g:.3f}±{np.std(gv):.3f}  entropy={e:.2f}")

    figure = plot_grouped_bars(
        names, {"stochastic policy": stochastic, "greedy policy": greedy},
        ylabel="catch rate", title="Walls comparison - open vs walled 3x3 square (±1 std, 8 seeds)",
        errors={"stochastic policy": s_err, "greedy policy": g_err},
    )
    os.makedirs("figures", exist_ok=True)
    figure.savefig("figures/exp07_topology_comparison.png", dpi=120)
    print("saved figures/exp07_topology_comparison.png")


if __name__ == "__main__":
    run()
