from __future__ import annotations

import cvxpy as cp
import numpy as np

from tnsm_sim.topology.certify import certify_topology, failure_sets
from tnsm_sim.topology.graph import Edge, canonical_edges

from .common import solve_problem
from .stage1 import _residual_affine_laplacian


def solve_stage2_sdp(
    n: int,
    f: int,
    edges: tuple[Edge, ...] | list[Edge],
    lambda_cap: float,
    total_budget: float,
    w_max: float,
    solver: str,
    time_limit_s: float,
) -> dict:
    edges = canonical_edges(edges)
    weights = cp.Variable(len(edges), name="w")
    gamma = cp.Variable(name="gamma")
    constraints: list[cp.Constraint] = [
        weights >= 0, weights <= w_max, cp.sum(weights) <= total_budget, gamma >= 0,
    ]
    for failed in failure_sets(n, f):
        active = sorted(set(range(1, n + 1)) - set(failed))
        projector = np.eye(len(active)) - np.ones((len(active), len(active))) / len(active)
        laplacian = _residual_affine_laplacian(active, edges, weights)
        constraints.extend([laplacian - gamma * projector >> 0, lambda_cap * projector - laplacian >> 0])
    problem = cp.Problem(cp.Maximize(gamma), constraints)
    record = solve_problem(problem, solver, time_limit_s)
    values = None if weights.value is None else np.maximum(np.asarray(weights.value).ravel(), 0.0)
    weight_map = None if values is None else {edge: float(value) for edge, value in zip(edges, values, strict=True)}
    certificate = None if weight_map is None else certify_topology(n, edges, f, weight_map)
    if certificate is None:
        numerical = None
    else:
        numerical = {
            "worst_lambda2": min(item.lambda2 for item in certificate.scenarios),
            "worst_lambda_max": max(item.lambda_max for item in certificate.scenarios),
            "lower_lmi_violation": max(0.0, float(gamma.value) - min(item.lambda2 for item in certificate.scenarios)),
            "upper_lmi_violation": max(0.0, max(item.lambda_max for item in certificate.scenarios) - lambda_cap),
            "weight_sum": float(values.sum()),
            "weight_upper_violation": max(0.0, float(values.max()) - w_max),
        }
    record.update({
        "problem": "stage2_shared_weight_sdp",
        "n": n,
        "f": f,
        "edge_count": len(edges),
        "failure_scenario_count": len(failure_sets(n, f)),
        "lambda_cap": lambda_cap,
        "total_budget": total_budget,
        "w_max": w_max,
        "gamma": None if gamma.value is None else float(gamma.value),
        "edges": [list(edge) for edge in edges],
        "weights": None if weight_map is None else {f"{i}-{j}": value for (i, j), value in weight_map.items()},
        "numerical_recertification": numerical,
    })
    return record
