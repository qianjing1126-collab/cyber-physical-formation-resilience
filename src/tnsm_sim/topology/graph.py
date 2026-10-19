from __future__ import annotations

from itertools import combinations
import numpy as np


Edge = tuple[int, int]


def canonical_edges(edges: tuple[Edge, ...] | list[Edge]) -> tuple[Edge, ...]:
    """Return sorted, duplicate-free undirected edges."""
    return tuple(sorted({(min(i, j), max(i, j)) for i, j in edges if i != j}))


def complete_edges(n: int) -> tuple[Edge, ...]:
    return tuple(combinations(range(1, n + 1), 2))


def ring_edges(n: int) -> tuple[Edge, ...]:
    if n < 2:
        return ()
    edges = [(i, i + 1) for i in range(1, n)]
    if n > 2:
        edges.append((1, n))
    return tuple(edges)


def residual_edges(edges: tuple[Edge, ...], active_ids: set[int]) -> tuple[Edge, ...]:
    return tuple((i, j) for i, j in edges if i in active_ids and j in active_ids)


def laplacian(node_ids: list[int], edges: tuple[Edge, ...], weights: dict[Edge, float] | None = None) -> np.ndarray:
    index = {node: k for k, node in enumerate(node_ids)}
    matrix = np.zeros((len(node_ids), len(node_ids)), dtype=float)
    for i, j in edges:
        if i not in index or j not in index:
            continue
        key = (min(i, j), max(i, j))
        weight = 1.0 if weights is None else float(weights.get(key, weights.get((i, j), 0.0)))
        a, b = index[i], index[j]
        matrix[a, a] += weight
        matrix[b, b] += weight
        matrix[a, b] -= weight
        matrix[b, a] -= weight
    return matrix


def algebraic_connectivity(node_ids: list[int], edges: tuple[Edge, ...]) -> float:
    if len(node_ids) < 2:
        return 0.0
    eigenvalues = np.linalg.eigvalsh(laplacian(node_ids, edges))
    return max(0.0, float(eigenvalues[1]))


def spectral_bounds(
    node_ids: list[int],
    edges: tuple[Edge, ...],
    weights: dict[Edge, float] | None = None,
) -> tuple[float, float]:
    if len(node_ids) < 2:
        return 0.0, 0.0
    eigenvalues = np.linalg.eigvalsh(laplacian(node_ids, edges, weights))
    return max(0.0, float(eigenvalues[1])), max(0.0, float(eigenvalues[-1]))


def largest_component_ratio(node_ids: list[int], edges: tuple[Edge, ...]) -> float:
    if not node_ids:
        return 0.0
    adjacency = {node: set() for node in node_ids}
    for i, j in edges:
        if i in adjacency and j in adjacency:
            adjacency[i].add(j)
            adjacency[j].add(i)
    unseen = set(node_ids)
    largest = 0
    while unseen:
        stack = [unseen.pop()]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            new = adjacency[node] & unseen
            unseen -= new
            stack.extend(new)
        largest = max(largest, size)
    return largest / len(node_ids)
