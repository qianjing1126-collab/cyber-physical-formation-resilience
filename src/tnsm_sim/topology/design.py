from __future__ import annotations

import random

from .certify import TopologyCertificate, certify_topology
from .graph import Edge, canonical_edges, complete_edges


def circulant_regular_edges(n: int, degree: int) -> tuple[Edge, ...]:
    if degree < 2 or degree >= n or (n * degree) % 2:
        raise ValueError("degree must define a simple regular graph")
    edges: set[Edge] = set()
    for distance in range(1, degree // 2 + 1):
        for i in range(n):
            j = (i + distance) % n
            edges.add((min(i + 1, j + 1), max(i + 1, j + 1)))
    if degree % 2:
        if n % 2:
            raise ValueError("odd degree requires even n")
        for i in range(n // 2):
            edges.add((i + 1, i + 1 + n // 2))
    return canonical_edges(list(edges))


def minimum_degree_seed_edges(n: int, minimum_degree: int) -> tuple[Edge, ...]:
    """Build a sparse deterministic seed at the handshaking lower bound.

    When ``n * minimum_degree`` is even, the seed is regular.  When both are
    odd, a regular graph is impossible; the minimum-edge degree sequence has
    one node of degree ``minimum_degree + 1`` and all others at
    ``minimum_degree``.  The latter is obtained from a circulant graph of the
    next even degree by deleting a matching that covers nodes 2 through n.
    """
    if minimum_degree < 2 or minimum_degree >= n:
        raise ValueError("minimum_degree must be between 2 and n-1")
    if (n * minimum_degree) % 2 == 0:
        return circulant_regular_edges(n, minimum_degree)
    if n % 2 == 0 or minimum_degree % 2 == 0 or minimum_degree + 1 >= n:
        raise ValueError("unsupported handshaking-parity case")
    edges = set(circulant_regular_edges(n, minimum_degree + 1))
    removed = {(node, node + 1) for node in range(2, n, 2)}
    if not removed.issubset(edges):
        raise RuntimeError("circulant seed does not contain the required matching")
    edges.difference_update(removed)
    return canonical_edges(list(edges))


def random_connected_edges(n: int, edge_count: int, seed: int) -> tuple[Edge, ...]:
    candidates = list(complete_edges(n))
    if not (n - 1 <= edge_count <= len(candidates)):
        raise ValueError("invalid edge_count")
    rng = random.Random(seed)
    for _ in range(20_000):
        trial = tuple(rng.sample(candidates, edge_count))
        if certify_topology(n, trial, 0).connectivity_rate == 1.0:
            return canonical_edges(trial)
    raise RuntimeError("could not sample a connected graph")


def greedy_edge_removal(
    n: int,
    edge_count: int,
    f: int,
    robust: bool,
) -> tuple[tuple[Edge, ...], TopologyCertificate]:
    """Deterministic offline heuristic; it is not an MISDP optimality claim."""
    edges = list(complete_edges(n))
    if not (n - 1 <= edge_count <= len(edges)):
        raise ValueError("invalid edge_count")
    certificate = certify_topology(n, edges, f if robust else 0)
    while len(edges) > edge_count:
        best: tuple[tuple[float, ...], Edge, TopologyCertificate] | None = None
        for edge in edges:
            trial = tuple(candidate for candidate in edges if candidate != edge)
            cert = certify_topology(n, trial, f if robust else 0)
            score = (
                cert.connectivity_rate,
                cert.worst_lambda2,
                -cert.degree_imbalance,
                -float(edge[0]),
                -float(edge[1]),
            )
            if best is None or score > best[0]:
                best = (score, edge, cert)
        assert best is not None
        edges.remove(best[1])
        certificate = best[2]
    final_edges = canonical_edges(edges)
    return final_edges, certify_topology(n, final_edges, f)
