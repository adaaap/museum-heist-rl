"""The Museum Heist game, written as a reinforcement learning environment.

Each round the guard watches one room and the thief reacts. A game ends when the
guard catches the thief (reward ``+1``), when the guard checks the painting's
room after it was stolen (reward ``+0.5``), or when the thief steals the painting
and gets back to the start (reward ``-1``).
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from museum_heist.envs.thief import Thief
from museum_heist.envs.topology import MuseumTopology


class MuseumHeistEnv(gym.Env):
    """Gymnasium environment for the surveillance/guard agent."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        topology: MuseumTopology,
        *,
        beta: float = 1.0,                 # thief prudence
        catch_reward: float = 1.0,
        escape_reward: float = -1.0,
        detect_reward: float = 0.5,        # guard spots the empty painting room after the theft
        fixed_start: int | None = None,    # pin s0 for tests/debugging; None => sampled
        fixed_painting: int | None = None,  # pin g for tests/debugging; None => sampled
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        # Defensive validation with specific exceptions.
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}.")
        for name, value in (("catch_reward", catch_reward), ("escape_reward", escape_reward), ("detect_reward", detect_reward)):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}.")
        n = topology.n_rooms
        if fixed_start is not None and not (0 <= fixed_start < n):
            raise ValueError(f"fixed_start {fixed_start} is out of range [0, {n}).")
        if fixed_painting is not None and not (0 <= fixed_painting < n):
            raise ValueError(f"fixed_painting {fixed_painting} is out of range [0, {n}).")
        if fixed_start is not None and fixed_painting is not None and fixed_start == fixed_painting:
            raise ValueError("fixed_start and fixed_painting must differ.")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}.")

        self.topology = topology
        self.thief = Thief(topology, beta=beta)
        self.catch_reward = float(catch_reward)
        self.escape_reward = float(escape_reward)
        self.detect_reward = float(detect_reward)
        self._fixed_start = fixed_start
        self._fixed_painting = fixed_painting
        self.render_mode = render_mode

        # The guard picks one room to watch (action id == room index == softmax
        # parameter index). The base policy is stateless, so the observation is just
        # a single dummy state (no context).
        self.action_space = spaces.Discrete(n)
        self.observation_space = spaces.Discrete(1)
        self._round: int | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)  # seeds self.np_random; the harness seeds the first reset only
        n = self.topology.n_rooms
        # The thief's start and painting are hidden from the guard, so they are
        # sampled each episode.
        # Sample whichever of start/painting is not fixed, and always keep them
        # different (the thief never starts on the painting). This handles all four
        # fixed/random combinations.
        start, painting = self._fixed_start, self._fixed_painting
        if start is None and painting is None:
            start = int(self.np_random.integers(n))
            painting = int(self.np_random.integers(n))
            while painting == start:
                painting = int(self.np_random.integers(n))
        elif start is None:  # painting fixed -> sample a start elsewhere
            start = int(self.np_random.integers(n))
            while start == painting:
                start = int(self.np_random.integers(n))
        elif painting is None:  # start fixed -> sample a painting elsewhere
            painting = int(self.np_random.integers(n))
            while painting == start:
                painting = int(self.np_random.integers(n))
        # (both fixed => validated distinct in __init__)
        self.thief.reset(start=start, painting=painting)
        self._round = 0
        info = {"thief_room": start, "painting": painting, "start": start, "has_stolen": False, "outcome": "ongoing"}
        return 0, info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        if self._round is None:
            raise RuntimeError("step() called before reset().")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r} for {self.action_space}.")
        self._round += 1
        watched = int(action)

        # Pre-move capture: the guard commits its camera. If it is the thief's
        # current room, the thief is caught before it can move this round.
        if watched == self.thief.pos:
            info = self._info("catch", watched)
            return 0, self.catch_reward, True, False, info

        # Theft detection: once the painting is gone, watching its (now-empty) room
        # means the guard notices the heist (selecting the target room after the
        # painting has been stolen). This is a partial success: the crime is seen,
        # but the thief is not caught.
        if self.thief.has_stolen and watched == self.thief.painting:
            info = self._info("detect", watched)
            return 0, self.detect_reward, True, False, info

        # The thief learns the live camera (hacker channel) and reacts.
        self.thief.observe_watch(watched)
        self.thief.step()

        # Heist success: the thief has stolen the painting and returned to start.
        if self.thief.has_stolen and self.thief.pos == self.thief.start:
            info = self._info("escape", watched)
            return 0, self.escape_reward, True, False, info

        # Ongoing. The env never truncates itself; the harness handles max-steps.
        return 0, 0.0, False, False, self._info("ongoing", watched)

    def _info(self, outcome: str, watched: int) -> dict[str, Any]:
        return {
            "outcome": outcome,
            "watched": watched,
            "thief_room": self.thief.pos,
            "has_stolen": self.thief.has_stolen,
            "round": self._round,
        }

    def render(self) -> np.ndarray | None:
        # No rendering; the experiment scripts build the figures directly.
        return None

    def close(self) -> None:
        pass
