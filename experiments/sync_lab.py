"""Copy the default settings from ``experiments/config.py`` into the browser lab.

The lab (``interactive_lab.html``) cannot read ``config.py`` while it runs, so
this script writes the same values into the lab. Run it after you change
``config.py`` (``run_all`` also does it for you).

    python -m experiments.sync_lab
"""
from __future__ import annotations

import re
from pathlib import Path

from experiments.config import ExperimentConfig

LAB = Path(__file__).resolve().parent.parent / "interactive_lab.html"
START = "// === DEFAULTS (generated from experiments/config.py, do not edit by hand) ==="
END = "// === END DEFAULTS ==="


def run() -> None:
    cfg = ExperimentConfig()
    if cfg.grid_height != cfg.grid_width:
        raise SystemExit("the lab assumes a square grid (grid_height must equal grid_width).")
    block = (
        f"{START}\n"
        f"const CFG = {{ N:{cfg.grid_height}, tau:{cfg.tau}, beta:{cfg.beta}, "
        f"alpha:{cfg.alpha}, gamma:{cfg.gamma}, batch:{cfg.batch_size}, "
        f"episodes:{cfg.n_episodes}, maxSteps:{cfg.max_steps} }};\n"
        f"{END}"
    )
    # encoding="utf-8" is required because the lab HTML has non-ASCII chars (such as ×).
    # Without it, Windows would default to cp1252 and fail to read/write the file.
    html = LAB.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(html):
        raise SystemExit(f"DEFAULTS markers not found in {LAB.name}.")
    LAB.write_text(pattern.sub(block, html), encoding="utf-8")
    print(f"synced {LAB.name} from config.py: N={cfg.grid_height} tau={cfg.tau} "
          f"beta={cfg.beta} alpha={cfg.alpha} gamma={cfg.gamma} "
          f"batch={cfg.batch_size} episodes={cfg.n_episodes} max_steps={cfg.max_steps}")


if __name__ == "__main__":
    run()
