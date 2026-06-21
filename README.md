# Museum Heist

Train a **museum surveillance agent** to catch a thief using **policy gradient**
methods, in a strategic setting that needs a **stochastic policy**.

**Author:** Ada Pena

## Problem in one paragraph

A thief moves through a museum (a grid of rooms) to steal a painting and escape.
A guard can watch only **one** camera at a time. Each round, the guard must choose which
room to display. A hacker tells the thief which camera is currently active, so a
predictable guard is easy to avoid. This is why the guard must learn a
**stochastic** watching strategy. We model the guard as the RL agent: its action
is the room to watch, and its policy is a temperature-`τ` softmax over rooms. It is
trained with **REINFORCE** to make catching the thief more likely. The
thief is a fixed shortest-path adversary that avoids rooms it watched recently
(prudence `β`).

## Repository structure

```
museum_heist/        # core package (reusable logic lives here)
  envs/              # Gymnasium environment + thief adversary + museum topologies
  agents/            # policy-gradient agent (softmax REINFORCE + baseline)
  utils/             # training loop, evaluation, plotting
experiments/         # scripts reproducing the reported results
```

## Setup

```bash
python -m venv .venv

# activate it (macOS / Linux):
source .venv/bin/activate
# Windows (PowerShell):   .venv\Scripts\Activate.ps1
# Windows (cmd):          .venv\Scripts\activate.bat

pip install -r requirements.txt
```

## Reproduce all results

```bash
python -m experiments.run_all   # regenerates every figure under figures/
```

## Interactive lab

- [interactive_lab.html](interactive_lab.html): a standalone browser lab (just open the
  file) to watch the guard learn live and place walls. Its defaults are kept
  in sync with `config.py` by `experiments/sync_lab.py` (run automatically by `run_all`).

## Reproducibility

One master seed is passed to both the environment and the agent. Results are
reported across multiple seeds (see `experiments/config.py`). Each script under
`experiments/` regenerates one figure deterministically.
