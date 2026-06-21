"""Rebuild every figure with one command.

    python -m experiments.run_all

Each experiment uses a fixed seed, so the figures come out the same every time.
They are written to ``figures/``.
"""
from __future__ import annotations

from experiments import (
    exp03_policy_heatmap,
    exp05_temperature_sweep,
    exp06_prudence_sweep,
    exp07_topology_comparison,
    sync_lab,
)

EXPERIMENTS = [
    ("exp03 learned watch heatmap", exp03_policy_heatmap),
    ("exp05 temperature sweep", exp05_temperature_sweep),
    ("exp06 thief-prudence sweep", exp06_prudence_sweep),
    ("exp07 walls comparison", exp07_topology_comparison),
]


def main() -> None:
    for label, module in EXPERIMENTS:
        print(f"\n=== {label} ===")
        module.run()
    # config.py is the single source of truth, so keep the browser lab's defaults in sync.
    print("\n=== sync interactive_lab.html ===")
    sync_lab.run()


if __name__ == "__main__":
    main()
