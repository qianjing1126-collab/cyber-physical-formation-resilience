from __future__ import annotations

from hashlib import sha256
import json
from typing import Iterable

import numpy as np

from .certify import TopologyCertificate
from .graph import Edge, canonical_edges, largest_component_ratio, residual_edges


def graph_sha256(n: int, edges: Iterable[Edge]) -> str:
    payload = {"n": int(n), "edges": [list(edge) for edge in canonical_edges(list(edges))]}
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def degree_sequence(n: int, edges: Iterable[Edge]) -> tuple[int, ...]:
    degrees = [0] * n
    for i, j in canonical_edges(list(edges)):
        degrees[i - 1] += 1
        degrees[j - 1] += 1
    return tuple(degrees)


def _component_count(node_ids: Iterable[int], edges: Iterable[Edge]) -> int:
    nodes = set(node_ids)
    if not nodes:
        return 0
    adjacency = {node: set() for node in nodes}
    for i, j in edges:
        if i in adjacency and j in adjacency:
            adjacency[i].add(j)
            adjacency[j].add(i)
    unseen = set(nodes)
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            node = stack.pop()
            new = adjacency[node] & unseen
            unseen -= new
            stack.extend(new)
    return count


def articulation_points(n: int, edges: Iterable[Edge]) -> tuple[int, ...]:
    canonical = canonical_edges(list(edges))
    nodes = set(range(1, n + 1))
    baseline = _component_count(nodes, canonical)
    result = []
    for removed in sorted(nodes):
        active = nodes - {removed}
        if _component_count(active, residual_edges(canonical, active)) > baseline:
            result.append(removed)
    return tuple(result)


def residual_rows(
    topology_id: str,
    method: str,
    seed: int | None,
    n: int,
    edges: tuple[Edge, ...],
    certificate: TopologyCertificate,
    scope: str,
) -> list[dict]:
    rows: list[dict] = []
    all_nodes = set(range(1, n + 1))
    for scenario in certificate.scenarios:
        active = all_nodes - set(scenario.failed)
        remaining = residual_edges(edges, active)
        rows.append({
            "scope": scope,
            "topology_id": topology_id,
            "method": method,
            "graph_seed": "" if seed is None else seed,
            "failure_count": len(scenario.failed),
            "failed_nodes": json.dumps(scenario.failed, separators=(",", ":")),
            "surviving_nodes": scenario.surviving_nodes,
            "surviving_edges": scenario.surviving_edges,
            "connected": int(scenario.connected),
            "lambda2": scenario.lambda2,
            "lambda_max": scenario.lambda_max,
            "S_c": largest_component_ratio(sorted(active), remaining),
        })
    return rows


def _quantiles(values: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_p05": float(np.quantile(array, 0.05)),
        f"{prefix}_q25": float(np.quantile(array, 0.25)),
        f"{prefix}_median": float(np.quantile(array, 0.50)),
        f"{prefix}_q75": float(np.quantile(array, 0.75)),
        f"{prefix}_p95": float(np.quantile(array, 0.95)),
        f"{prefix}_max": float(np.max(array)),
    }


def topology_record(
    *,
    scope: str,
    method: str,
    seed: int | None,
    n: int,
    edges: tuple[Edge, ...],
    certificate: TopologyCertificate,
    runtime_s: float,
) -> tuple[dict, list[dict]]:
    canonical = canonical_edges(edges)
    digest = graph_sha256(n, canonical)
    topology_id = f"{scope}_{method}_N{n}_E{len(canonical)}_seed{'fixed' if seed is None else seed}_{digest[:12]}"
    scenarios = residual_rows(topology_id, method, seed, n, canonical, certificate, scope)
    lambda2 = [float(row["lambda2"]) for row in scenarios]
    giant = [float(row["S_c"]) for row in scenarios]
    by_failure = {}
    for failure_count in sorted({int(row["failure_count"]) for row in scenarios}):
        subset = [row for row in scenarios if int(row["failure_count"]) == failure_count]
        by_failure[f"connectivity_rate_f{failure_count}"] = sum(int(row["connected"]) for row in subset) / len(subset)
        by_failure[f"S_c_min_f{failure_count}"] = min(float(row["S_c"]) for row in subset)
        by_failure[f"lambda2_min_f{failure_count}"] = min(float(row["lambda2"]) for row in subset)
    record = {
        "scope": scope,
        "topology_id": topology_id,
        "method": method,
        "graph_seed": "" if seed is None else seed,
        "graph_sha256": digest,
        "n": n,
        "f": certificate.f,
        "edge_count": len(canonical),
        "edges": json.dumps(canonical, separators=(",", ":")),
        "degree_sequence": json.dumps(degree_sequence(n, canonical), separators=(",", ":")),
        "minimum_degree": certificate.minimum_degree,
        "degree_imbalance": certificate.degree_imbalance,
        "articulation_points": json.dumps(articulation_points(n, canonical), separators=(",", ":")),
        "articulation_count": len(articulation_points(n, canonical)),
        "necessary_edge_bound": certificate.necessary_edge_bound,
        "necessary_count_met": int(certificate.necessary_count_met),
        "scenario_count": certificate.scenario_count,
        "connected_scenarios": certificate.connected_scenarios,
        "connectivity_rate": certificate.connectivity_rate,
        "worst_lambda2": certificate.worst_lambda2,
        "worst_failure": json.dumps(certificate.worst_failure, separators=(",", ":")),
        "runtime_s": runtime_s,
        **_quantiles(lambda2, "lambda2"),
        **_quantiles(giant, "S_c"),
        **by_failure,
    }
    return record, scenarios
