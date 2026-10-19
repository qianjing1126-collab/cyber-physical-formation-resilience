from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tnsm_sim.topology.certify import certify_topology
from tnsm_sim.topology.graph import Edge, canonical_edges


@dataclass(frozen=True)
class WeightDesign:
    name: str
    weights: dict[Edge, float]
    worst_lambda2: float
    worst_lambda_max: float
    scale: float


def _candidate_weights(edges: tuple[Edge, ...], values: np.ndarray) -> dict[Edge, float]:
    return {edge: float(value) for edge, value in zip(edges, values, strict=True)}


def _evaluate(n: int, f: int, edges: tuple[Edge, ...], values: np.ndarray) -> tuple[float, float]:
    certificate = certify_topology(n, edges, f, _candidate_weights(edges, values))
    worst_lambda2 = min(record.lambda2 for record in certificate.scenarios)
    worst_lambda_max = max(record.lambda_max for record in certificate.scenarios)
    return worst_lambda2, worst_lambda_max


def pilot_weight_designs(
    n: int,
    f: int,
    edges: tuple[Edge, ...] | list[Edge],
    total_budget: float,
    lambda_cap: float,
    samples: int,
    seed: int,
) -> tuple[WeightDesign, ...]:
    """Deterministic random-search pilot for baselines and the shared-weight SDP logic."""
    edges = canonical_edges(edges)
    rng = np.random.default_rng(seed)
    uniform = np.full(len(edges), total_budget / len(edges), dtype=float)
    uniform_l2, uniform_lmax = _evaluate(n, f, edges, uniform)
    convergence_best = (uniform_l2, -uniform_lmax, uniform.copy())
    delay_best: tuple[float, float, np.ndarray, float] | None = None

    candidates = [uniform]
    candidates.extend(
        total_budget * raw / raw.sum()
        for raw in rng.lognormal(mean=0.0, sigma=0.55, size=(samples, len(edges)))
    )
    for values in candidates:
        lam2, lam_max = _evaluate(n, f, edges, values)
        if (lam2, -lam_max) > convergence_best[:2]:
            convergence_best = (lam2, -lam_max, values.copy())
        scale = min(1.0, lambda_cap / lam_max) if lam_max > 0 else 1.0
        scaled = values * scale
        scaled_l2, scaled_lmax = _evaluate(n, f, edges, scaled)
        candidate = (scaled_l2, -scaled_lmax, scaled.copy(), scale)
        if delay_best is None or candidate[:2] > delay_best[:2]:
            delay_best = candidate

    assert delay_best is not None
    conv_values = convergence_best[2]
    conv_l2, conv_lmax = _evaluate(n, f, edges, conv_values)
    delay_values = delay_best[2]
    delay_l2, delay_lmax = _evaluate(n, f, edges, delay_values)
    return (
        WeightDesign("B7_uniform", _candidate_weights(edges, uniform), uniform_l2, uniform_lmax, 1.0),
        WeightDesign("B8_convergence_priority", _candidate_weights(edges, conv_values), conv_l2, conv_lmax, 1.0),
        WeightDesign("Stage2_delay_aware_pilot", _candidate_weights(edges, delay_values), delay_l2, delay_lmax, delay_best[3]),
    )
