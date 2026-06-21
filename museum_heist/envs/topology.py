"""The museum layout: rooms joined by doors on a 2D grid.

A layout is a fixed, checked list of rooms and doors. All path-finding lives in
``museum_heist.utils.graph``, so the environment, the thief, and the tests share
one shortest-path function. We study how adding walls between rooms changes what
the guard learns. The single factory ``grid_with_walls`` builds both the open
square and any walled version.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from museum_heist.utils.graph import (
    bfs_shortest_path,
    dijkstra_shortest_path,
    is_connected,
)

Coord = tuple[int, int]          # (row, col) cell on the grid
Edge = tuple[Coord, Coord]       # a door (or, when walled, a removed edge)


@dataclass(frozen=True)
class MuseumTopology:
    """An immutable museum map.

    Parameters
    ----------
    height, width:
        Grid dimensions; used only for layout and rendering.
    rooms:
        Ordered ``(row, col)`` cells that are rooms. The order DEFINES the
        standard room index ``0..n_rooms-1``, which is also the environment
        action id and the softmax parameter index, so it must stay stable.
    adjacency:
        ``adjacency[i]`` is the set of room indices reachable from room ``i``
        through a door in one step. It must be symmetric and self-loop free.

    Notes
    -----
    ``n_rooms`` and an internal coord -> index map are built in ``__post_init__``.
    The dataclass is frozen, so they are set via ``object.__setattr__``. They are
    not constructor fields.
    """

    height: int
    width: int
    rooms: tuple[Coord, ...]
    adjacency: tuple[frozenset[int], ...]

    def __post_init__(self) -> None:
        if not self.rooms:
            raise ValueError("a topology must have at least one room.")
        if len(set(self.rooms)) != len(self.rooms):
            raise ValueError("duplicate room coordinates are not allowed.")
        for (row, col) in self.rooms:
            if not (0 <= row < self.height and 0 <= col < self.width):
                raise ValueError(f"room {(row, col)} is outside the {self.height}x{self.width} grid.")
        if len(self.adjacency) != len(self.rooms):
            raise ValueError("adjacency and rooms must have the same length.")
        n = len(self.rooms)
        for i, neighbours in enumerate(self.adjacency):
            for j in neighbours:
                if not (0 <= j < n):
                    raise ValueError(f"adjacency[{i}] references invalid room {j}.")
                if j == i:
                    raise ValueError(f"self-loop at room {i} is not allowed.")
                if i not in self.adjacency[j]:
                    raise ValueError(f"adjacency is not symmetric between {i} and {j}.")
        if not is_connected(self.adjacency):
            raise ValueError("the museum graph is disconnected: some rooms are unreachable.")
        # Derived caches, set on the frozen instance.
        object.__setattr__(self, "n_rooms", n)
        object.__setattr__(self, "_coord_to_index", {coord: i for i, coord in enumerate(self.rooms)})

    def index_of(self, coord: Coord) -> int:
        """Room index for a ``(row, col)`` cell; ``ValueError`` if not a room."""
        try:
            return self._coord_to_index[coord]
        except KeyError:
            raise ValueError(f"{coord} is not a room in this topology.") from None

    def neighbours(self, room: int) -> frozenset[int]:
        """Door-neighbours of ``room`` (excludes the room itself)."""
        return self.adjacency[room]

    def shortest_path(self, source: int, target: int, *, costs: np.ndarray | None = None) -> list[int]:
        """Node list ``[source, ..., target]``; unit-cost BFS unless ``costs`` given."""
        if costs is None:
            return bfs_shortest_path(self.adjacency, source, target)
        return dijkstra_shortest_path(self.adjacency, source, target, costs)

    def next_step_towards(self, source: int, target: int, *, costs: np.ndarray | None = None) -> int:
        """Room to enter this round (2nd node of the path), or ``source`` if already there."""
        path = self.shortest_path(source, target, costs=costs)
        return path[1] if len(path) > 1 else source


# --------------------------------------------------------------------------- #
# Square gridworld factory: the open square and any walled variant.            #
# --------------------------------------------------------------------------- #
def grid_with_walls(height: int, width: int, walls: Iterable[Edge] = ()) -> MuseumTopology:
    """Full ``height x width`` grid minus the given wall edges.

    Each wall is the pair of adjacent cells it separates. This is the main topology:
    ``walls=()`` is the open square, while passing edges adds interior walls the
    thief must route around.
    """
    rooms = tuple((r, c) for r in range(height) for c in range(width))
    index = {coord: i for i, coord in enumerate(rooms)}
    wall_set = {frozenset((a, b)) for a, b in walls}
    adjacency: list[set[int]] = [set() for _ in rooms]
    for (row, col) in rooms:
        here = index[(row, col)]
        for (d_row, d_col) in ((-1, 0), (1, 0), (0, -1), (0, 1)):  # 4-neighbourhood
            neighbour = (row + d_row, col + d_col)
            if neighbour in index and frozenset(((row, col), neighbour)) not in wall_set:
                adjacency[here].add(index[neighbour])
    return MuseumTopology(
        height=height, width=width, rooms=rooms,
        adjacency=tuple(frozenset(neighbours) for neighbours in adjacency),
    )


def open_grid(height: int = 5, width: int = 5) -> MuseumTopology:
    """A fully connected ``height x width`` square (no walls)."""
    return grid_with_walls(height, width)
