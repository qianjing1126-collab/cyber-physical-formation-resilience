from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np

from .graph import Edge, canonical_edges, residual_edges, spectral_bounds


@dataclass(frozen=True)
class ResidualScenario:
    failed: tuple[int, ...]
    surviving_nodes: int
    surviving_edges: int
    connected: bool
    lambda2: float
    lambda_max: float


@dataclass(frozen=True)
class TopologyCertificate:
    n: int
    f: int
    edge_count: int
    minimum_degree: int
    necessary_edge_bound: int
    necessary_count_met: bool
    scenario_count: int
    connected_scenarios: int
    connectivity_rate: float
    worst_lambda2: float
    worst_failure: tuple[int, ...]
    degree_imbalance: float
    scenarios: tuple[ResidualScenario, ...]

    def to_dict(self, include_scenarios: bool = True) -> dict:
        result = asdict(self)
        if not include_scenarios:
            result.pop("scenarios")
        return result


def failure_sets(n: int, f: int, include_empty: bool = True) -> tuple[tuple[int, ...], ...]:
    first = 0 if include_empty else 1
    return tuple(combo for size in range(first, f + 1) for combo in combinations(range(1, n + 1), size))


def certify_topology(
    n: int,
    edges: tuple[Edge, ...] | list[Edge],
    f: int,
    weights: dict[Edge, float] | None = None,
) -> TopologyCertificate:
    edges = canonical_edges(edges)
    degrees = {i: 0 for i in range(1, n + 1)}
    for i, j in edges:
        degrees[i] += 1
        degrees[j] += 1
    mean_degree = 2.0 * len(edges) / n
    imbalance = sum((degree - mean_degree) ** 2 for degree in degrees.values())
    records: list[ResidualScenario] = []
    for failed in failure_sets(n, f):
        active = set(range(1, n + 1)) - set(failed)
        residual = residual_edges(edges, active)
        lam2, lam_max = spectral_bounds(sorted(active), residual, weights)
        connected = len(active) <= 1 or lam2 > 1e-9
        records.append(ResidualScenario(failed, len(active), len(residual), connected, lam2, lam_max))
    worst = min(records, key=lambda item: (item.lambda2, item.failed))
    connected_count = sum(record.connected for record in records)
    bound = int(np.ceil(n * (f + 1) / 2))
    return TopologyCertificate(
        n=n,
        f=f,
        edge_count=len(edges),
        minimum_degree=min(degrees.values()),
        necessary_edge_bound=bound,
        necessary_count_met=len(edges) >= bound,
        scenario_count=len(records),
        connected_scenarios=connected_count,
        connectivity_rate=connected_count / len(records),
        worst_lambda2=worst.lambda2,
        worst_failure=worst.failed,
        degree_imbalance=imbalance,
        scenarios=tuple(records),
    )
