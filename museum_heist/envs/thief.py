"""The thief: a fixed opponent that takes the shortest path and avoids watched rooms.

Each round it finds the shortest route to its goal, treating rooms the guard
watched recently as more expensive, then moves one step. Its goal is the
painting; once the painting is stolen, the goal becomes the exit (the start
room). The thief is run by the environment, not by a learning agent.
"""
from __future__ import annotations

import numpy as np

from museum_heist.envs.topology import MuseumTopology

# "Never watched recently": 2**-LARGE underflows to 0, so the surcharge goes away
# and entry costs are ~1 everywhere at the start of an episode.
INITIAL_ROUNDS_SINCE_WATCHED = 1_000


class Thief:
    """Fixed adversary; the mutable per-episode state lives on the instance."""

    def __init__(self, topology: MuseumTopology, *, beta: float) -> None:
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}.")
        self.topology = topology
        self.beta = float(beta)
        # rounds_since_watched[room] drives the entry-cost surcharge.
        self._rounds_since_watched = np.full(
            topology.n_rooms, INITIAL_ROUNDS_SINCE_WATCHED, dtype=np.int64
        )
        # Episode state, set by reset().
        self._pos: int | None = None
        self._start: int | None = None
        self._painting: int | None = None
        self._target: int | None = None
        self._has_stolen: bool = False

    def reset(self, *, start: int, painting: int) -> None:
        if start == painting:
            raise ValueError("start and painting rooms must differ.")
        self._pos = int(start)
        self._start = int(start)
        self._painting = int(painting)
        self._target = int(painting)
        self._has_stolen = False
        self._rounds_since_watched[:] = INITIAL_ROUNDS_SINCE_WATCHED

    def observe_watch(self, watched_room: int) -> None:
        """Hacker channel: one round has passed and ``watched_room`` is the live camera."""
        self._rounds_since_watched += 1               # age every room by one round
        self._rounds_since_watched[watched_room] = 0  # just watched => strongest deterrence

    def _entry_costs(self) -> np.ndarray:
        # With n = rounds_since_watched + 1, the spec's 2**-(n-1) becomes
        # 2**-rounds_since_watched: a just-watched room (counter 0) costs 1 + beta
        # (the maximum); long-unwatched rooms decay toward cost 1.
        return 1.0 + self.beta * np.exp2(-self._rounds_since_watched.astype(np.float64))

    def plan_move(self) -> int:
        """Room the thief intends to enter next round (no RNG; ties broken by the graph)."""
        return self.topology.next_step_towards(self._pos, self._target, costs=self._entry_costs())

    def step(self) -> int:
        """Commit the move; flip the target to the escape room once the painting is taken."""
        self._pos = self.plan_move()
        if not self._has_stolen and self._pos == self._painting:
            self._has_stolen = True
            self._target = self._start  # now heading home to escape
        return self._pos

    @property
    def pos(self) -> int:
        return self._pos

    @property
    def start(self) -> int:
        return self._start

    @property
    def painting(self) -> int:
        return self._painting

    @property
    def has_stolen(self) -> bool:
        return self._has_stolen

    @property
    def rounds_since_watched(self) -> np.ndarray:
        """A copy of the per-room watch-history counters (for tests / diagnostics)."""
        return self._rounds_since_watched.copy()
