"""Shortest-path routing on a room graph.

This is the one place in the project that computes routes. The museum layout and
the thief both use it, and the tests check it directly. Ties are broken by the
lowest room number, so routing is deterministic and every game can be reproduced
from its seed.
"""
from __future__ import annotations

import heapq
from collections import deque
from typing import Sequence

import numpy as np

# An adjacency is a sequence where entry `i` lists the room indices reachable
# from room `i` through a single door (any iterable of ints).
Adjacency = Sequence["frozenset[int]"]


def is_connected(adjacency: Adjacency) -> bool:
    """Return True if every room is reachable from room 0 (undirected graph)."""
    n = len(adjacency)
    if n == 0:
        return True
    seen = {0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for neighbour in adjacency[node]:
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return len(seen) == n


def bfs_shortest_path(adjacency: Adjacency, source: int, target: int) -> list[int]:
    """Unit-cost shortest path as a node list ``[source, ..., target]``.

    Neighbours are explored in ascending index order, so the path is a
    deterministic function of ``(adjacency, source, target)``. Raises
    ``ValueError`` if the target is unreachable.
    """
    if source == target:
        return [source]
    predecessor: dict[int, int] = {source: source}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbour in sorted(adjacency[node]):  # deterministic exploration order
            if neighbour not in predecessor:
                predecessor[neighbour] = node
                if neighbour == target:
                    return _reconstruct(predecessor, source, target)
                queue.append(neighbour)
    raise ValueError(f"target {target} is unreachable from source {source}.")


def dijkstra_shortest_path(
    adjacency: Adjacency, source: int, target: int, entry_costs: np.ndarray
) -> list[int]:
    """Least-cost path where ``entry_costs[room]`` is the cost to ENTER ``room``.

    The source's own cost is not paid. ``entry_costs`` must be strictly positive
    (the thief's cost is ``c(room) = 1 + beta * 2**-(n-1) >= 1``), which keeps
    Dijkstra valid. Ties are broken in a fixed way (the heap key ``(distance,
    room)`` fixes the order rooms are finalised), so the path is a deterministic
    function of the inputs. Raises ``ValueError`` if the target is unreachable.
    """
    if source == target:
        return [source]
    n = len(adjacency)
    dist = np.full(n, np.inf)
    dist[source] = 0.0
    predecessor: dict[int, int] = {source: source}
    # Heap entries are (distance, room); the room index tie-breaks equal distances.
    heap: list[tuple[float, int]] = [(0.0, source)]
    visited: set[int] = set()
    while heap:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == target:
            return _reconstruct(predecessor, source, target)
        for neighbour in adjacency[node]:
            if neighbour in visited:
                continue
            candidate = d + float(entry_costs[neighbour])
            if candidate < dist[neighbour]:  # strict: first (cheapest) relaxation wins
                dist[neighbour] = candidate
                predecessor[neighbour] = node
                heapq.heappush(heap, (candidate, neighbour))
    raise ValueError(f"target {target} is unreachable from source {source}.")


def _reconstruct(predecessor: dict[int, int], source: int, target: int) -> list[int]:
    """Walk predecessors from target back to source, returning the forward path."""
    path = [target]
    while path[-1] != source:
        path.append(predecessor[path[-1]])
    path.reverse()
    return path
