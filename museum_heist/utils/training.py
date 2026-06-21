"""The training and evaluation loop, which works with any agent.

An agent only needs ``select_action``, ``update``, and ``end_episode`` methods
(no shared base class), so the same loop can drive any of them. Only the first
``env.reset`` gets the seed; after that the random generator runs on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


def _derive_seeds(master: int, n: int) -> list[int]:
    """Derive ``n`` independent integer seeds from one master seed."""
    sequence = np.random.SeedSequence(master)
    return [int(child.generate_state(1)[0]) for child in sequence.spawn(n)]


@runtime_checkable
class Agent(Protocol):
    """Structural contract every agent meets (duck typing, no inheritance)."""

    def select_action(self, state: Any, *, greedy: bool = False) -> int: ...
    def update(self, state: Any, action: int, reward: float, next_state: Any,
               terminated: bool, *, next_action: int | None = None) -> None: ...
    def end_episode(self) -> None: ...


@dataclass
class TrainingHistory:
    """Per-episode return trace."""

    episode_returns: list[float] = field(default_factory=list)


def _run_episode(agent: Agent, env, *, max_steps: int, greedy: bool, learn: bool,
                 reset_seed: int | None = None) -> tuple[float, int]:
    """Roll out one episode; return ``(undiscounted_return, length)``."""
    state, _ = env.reset(seed=reset_seed)
    total = 0.0
    step = 0
    for step in range(1, max_steps + 1):
        action = agent.select_action(state, greedy=greedy)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total += reward
        if learn:
            agent.update(state, action, reward, next_state, terminated)
        state = next_state
        if terminated or truncated:
            break
    if learn:
        agent.end_episode()  # Monte-Carlo agents apply their gradient here
    return total, step


def evaluate(agent: Agent, env, n_episodes: int, *, max_steps: int,
             seed: int | None = None) -> tuple[float, float, np.ndarray]:
    """Greedy evaluation; returns ``(mean_return, std_return, returns_array)``."""
    returns = np.empty(n_episodes)
    for i in range(n_episodes):
        # Seed only the first reset; later resets advance the same RNG.
        ret, _ = _run_episode(agent, env, max_steps=max_steps, greedy=True, learn=False,
                              reset_seed=seed if i == 0 else None)
        returns[i] = ret
    return float(returns.mean()), float(returns.std()), returns


def train(agent: Agent, env, n_episodes: int, *, max_steps_per_episode: int,
          seed: int | None = None) -> TrainingHistory:
    """Train ``agent`` on ``env`` for ``n_episodes``; returns a ``TrainingHistory``."""
    history = TrainingHistory()
    env_seed = seed
    if seed is not None and hasattr(agent, "reset_rng"):
        # Pass one master seed to BOTH stochastic objects (env + agent) so a
        # single `seed` fully determines the run through the harness.
        env_seed, agent_seed = _derive_seeds(seed, 2)
        agent.reset_rng(agent_seed)
    for episode in range(1, n_episodes + 1):
        ret, _ = _run_episode(
            agent, env, max_steps=max_steps_per_episode, greedy=False, learn=True,
            reset_seed=env_seed if episode == 1 else None,  # seed the FIRST env reset only
        )
        history.episode_returns.append(ret)
    return history
