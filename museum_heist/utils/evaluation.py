"""Measuring how well the guard plays, with easy-to-read scores.

Training uses a single reward number, but that hides the details. This module
splits performance into clear scores: how often the guard catches the thief, how
often the thief escapes or runs out of time, and how long a catch takes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EvaluationResult:
    """KPIs over a frozen batch of evaluation episodes."""

    n_episodes: int
    catch_rate: float
    escape_rate: float
    timeout_rate: float
    detect_rate: float  # guard spotted the empty painting room after the theft
    mean_return: float
    std_return: float
    mean_time_to_catch: float  # over caught episodes only; NaN if none
    returns: np.ndarray


def evaluate_policy(agent, env, n_episodes: int, *, max_steps: int,
                    seed: int | None = None, greedy: bool = True) -> EvaluationResult:
    """Roll out ``n_episodes`` with learning disabled and aggregate the KPIs.

    ``greedy=True`` is the deterministic evaluation. ``greedy=False`` is the honest
    stochastic operating point, because a deterministic guard is easy to take
    advantage of. When ``seed`` is given, the env episode stream is seeded (first
    reset only) AND the agent's sampling RNG is reseeded, so the whole evaluation
    (including the stochastic policy) is reproducible from ``seed`` alone.
    """
    if seed is not None and hasattr(agent, "reset_rng"):
        agent.reset_rng(seed)  # make stochastic evaluation reproducible from `seed`
    returns = np.empty(n_episodes)
    catches = escapes = timeouts = detects = 0
    catch_times: list[int] = []
    for i in range(n_episodes):
        state, _ = env.reset(seed=seed if i == 0 else None)
        total = 0.0
        outcome = "timeout"  # if we never break, the horizon was hit
        step = 0
        for step in range(1, max_steps + 1):
            action = agent.select_action(state, greedy=greedy)
            state, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                outcome = info["outcome"]
                break
        returns[i] = total
        if outcome == "catch":
            catches += 1
            catch_times.append(step)
        elif outcome == "escape":
            escapes += 1
        elif outcome == "detect":
            detects += 1
        else:
            timeouts += 1
    return EvaluationResult(
        n_episodes=n_episodes,
        catch_rate=catches / n_episodes,
        escape_rate=escapes / n_episodes,
        timeout_rate=timeouts / n_episodes,
        detect_rate=detects / n_episodes,
        mean_return=float(returns.mean()),
        std_return=float(returns.std()),
        mean_time_to_catch=float(np.mean(catch_times)) if catch_times else float("nan"),
        returns=returns,
    )


def policy_entropy(probs: np.ndarray) -> float:
    """Shannon entropy of a probability vector.

    Shannon entropy in nats; 0 = a deterministic policy.
    """
    p = np.asarray(probs, dtype=float)
    nonzero = p[p > 0]
    return float(-(nonzero * np.log(nonzero)).sum())
