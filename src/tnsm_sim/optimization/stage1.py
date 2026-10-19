from __future__ import annotations

from itertools import combinations
from time import perf_counter

import cvxpy as cp
import numpy as np

from tnsm_sim.topology.certify import TopologyCertificate, certify_topology, failure_sets
from tnsm_sim.topology.graph import Edge, canonical_edges, complete_edges

from .common import solve_problem


def _residual_affine_laplacian(
    active: list[int],
    candidate_edges: tuple[Edge, ...],
    variables: cp.Expression,
) -> cp.Expression:
    index = {node: offset for offset, node in enumerate(active)}
    matrix: cp.Expression = cp.Constant(np.zeros((len(active), len(active))))
    for edge_index, (i, j) in enumerate(candidate_edges):
        if i not in index or j not in index:
            continue
        incidence = np.zeros(len(active), dtype=float)
        incidence[index[i]], incidence[index[j]] = 1.0, -1.0
        matrix = matrix + variables[edge_index] * np.outer(incidence, incidence)
    return matrix


def solve_stage1_relaxation(
    n: int,
    f: int,
    edge_count: int,
    solver: str,
    time_limit_s: float,
    candidate_edges: tuple[Edge, ...] | None = None,
) -> dict:
    """Max-z SDP relaxation for the fixed-edge-budget P0-1 comparison."""
    candidate_edges = complete_edges(n) if candidate_edges is None else canonical_edges(candidate_edges)
    x = cp.Variable(len(candidate_edges), name="x")
    z = cp.Variable(name="z")
    constraints: list[cp.Constraint] = [x >= 0, x <= 1, cp.sum(x) == edge_count, z >= 0]
    for failed in failure_sets(n, f):
        active = sorted(set(range(1, n + 1)) - set(failed))
        projector = np.eye(len(active)) - np.ones((len(active), len(active))) / len(active)
        laplacian = _residual_affine_laplacian(active, candidate_edges, x)
        constraints.append(laplacian - z * projector >> 0)
    problem = cp.Problem(cp.Maximize(z), constraints)
    record = solve_problem(problem, solver, time_limit_s)
    fractions = None if x.value is None else [float(value) for value in np.asarray(x.value).ravel()]
    record.update({
        "problem": "stage1_max_worst_residual_lambda2_relaxation",
        "n": n,
        "f": f,
        "edge_count": edge_count,
        "candidate_edge_count": len(candidate_edges),
        "failure_scenario_count": len(failure_sets(n, f)),
        "relaxation_upper_bound": None if z.value is None else float(z.value),
        "candidate_edges": [list(edge) for edge in candidate_edges],
        "edge_fractions": fractions,
    })
    return record


def round_relaxed_topology(
    n: int,
    f: int,
    edge_count: int,
    candidate_edges: tuple[Edge, ...] | list[Edge],
    fractions: list[float],
    max_swap_rounds: int = 30,
    random_restarts: int = 12,
    seed: int = 20260806,
    initial_edges: tuple[Edge, ...] | list[Edge] | None = None,
    preserve_degree: bool = False,
) -> tuple[tuple[Edge, ...], TopologyCertificate, dict]:
    candidate_edges = canonical_edges(candidate_edges)
    ranked = sorted(zip(fractions, candidate_edges, strict=True), key=lambda item: (-item[0], item[1]))

    def score(edges: set[Edge]) -> tuple[float, float, float]:
        certificate = certify_topology(n, tuple(edges), f)
        return certificate.connectivity_rate, certificate.worst_lambda2, -certificate.degree_imbalance

    rng = np.random.default_rng(seed)
    starts: list[set[Edge]] = []
    if initial_edges is not None:
        certified_start = set(canonical_edges(initial_edges))
        if len(certified_start) != edge_count or not certified_start.issubset(set(candidate_edges)):
            raise ValueError("initial_edges must contain edge_count distinct candidate edges")
        starts.append(certified_start)
    else:
        starts.append({edge for _, edge in ranked[:edge_count]})
    probability = np.maximum(np.asarray(fractions, dtype=float), 1e-8)
    probability /= probability.sum()
    for _ in range(random_restarts):
        indices = rng.choice(len(candidate_edges), edge_count, replace=False, p=probability)
        starts.append({candidate_edges[int(index)] for index in indices})

    initial_score = score(starts[0])
    best_selected, best_overall_score = starts[0], initial_score
    total_rounds = 0
    evaluated_swaps = 0
    for selected in starts:
        rounds = 0
        while rounds < max_swap_rounds:
            current_score = score(selected)
            best_score, best_swap = current_score, None
            if preserve_degree:
                for first, second in combinations(sorted(selected), 2):
                    a, b = first
                    c, d = second
                    if len({a, b, c, d}) != 4:
                        continue
                    for added_pair in (
                        (tuple(sorted((a, c))), tuple(sorted((b, d)))),
                        (tuple(sorted((a, d))), tuple(sorted((b, c)))),
                    ):
                        added = set(added_pair)
                        removed = {first, second}
                        if len(added) != 2 or any(edge not in candidate_edges for edge in added):
                            continue
                        if (added - removed) & (selected - removed):
                            continue
                        evaluated_swaps += 1
                        trial = (selected - removed) | added
                        trial_score = score(trial)
                        if trial_score > best_score:
                            best_score, best_swap = trial_score, (removed, added)
            else:
                excluded = set(candidate_edges) - selected
                for removed in sorted(selected):
                    for added in sorted(excluded):
                        evaluated_swaps += 1
                        trial = (selected - {removed}) | {added}
                        trial_score = score(trial)
                        if trial_score > best_score:
                            best_score, best_swap = trial_score, ({removed}, {added})
            if best_swap is None:
                break
            selected.difference_update(best_swap[0])
            selected.update(best_swap[1])
            rounds += 1
        total_rounds += rounds
        final_score = score(selected)
        if final_score > best_overall_score:
            best_selected, best_overall_score = selected.copy(), final_score
    edges = canonical_edges(list(best_selected))
    certificate = certify_topology(n, edges, f)
    return edges, certificate, {
        "method": "relaxation_ranking_with_optional_feasible_start_and_certified_one_edge_swaps",
        "initial_score": list(initial_score),
        "final_score": [certificate.connectivity_rate, certificate.worst_lambda2, -certificate.degree_imbalance],
        "random_restarts": random_restarts,
        "swap_type": "degree_preserving_two_switch" if preserve_degree else "one_edge_exchange",
        "total_swap_rounds": total_rounds,
        "evaluated_swaps": evaluated_swaps,
    }


def _regular_graphs(n: int, degree: int):
    remaining = [degree] * n
    edges: list[Edge] = []

    def recurse():
        try:
            node = next(index for index, value in enumerate(remaining) if value > 0)
        except StopIteration:
            yield tuple(edges)
            return
        needed = remaining[node]
        candidates = [other for other in range(node + 1, n) if remaining[other] > 0]
        if len(candidates) < needed:
            return
        remaining[node] = 0
        for neighbors in combinations(candidates, needed):
            for neighbor in neighbors:
                remaining[neighbor] -= 1
                edges.append((node + 1, neighbor + 1))
            if all(value >= 0 for value in remaining):
                yield from recurse()
            for neighbor in reversed(neighbors):
                edges.pop()
                remaining[neighbor] += 1
        remaining[node] = needed

    yield from recurse()


def enumerate_exact_max_z(n: int, f: int, edge_count: int) -> dict:
    candidates = complete_edges(n)
    started = perf_counter()
    best_edges: tuple[Edge, ...] | None = None
    best_certificate: TopologyCertificate | None = None
    evaluated = 0
    average_degree = 2 * edge_count / n
    degree_reduction_valid = float(average_degree).is_integer() and int(average_degree) == f + 1
    topology_iterator = _regular_graphs(n, int(average_degree)) if degree_reduction_valid else combinations(candidates, edge_count)
    for edges in topology_iterator:
        evaluated += 1
        certificate = certify_topology(n, edges, f)
        if best_certificate is None:
            best_edges, best_certificate = edges, certificate
            continue
        score = (certificate.worst_lambda2, -certificate.degree_imbalance, tuple(edges))
        best_score = (best_certificate.worst_lambda2, -best_certificate.degree_imbalance, tuple(best_edges or ()))
        if score > best_score:
            best_edges, best_certificate = edges, certificate
    assert best_edges is not None and best_certificate is not None
    return {
        "status": "OPTIMAL_BY_COMPLETE_ENUMERATION",
        "n": n,
        "f": f,
        "edge_count": edge_count,
        "evaluated_topologies": evaluated,
        "enumeration_domain": "all_(f+1)-regular_graphs" if degree_reduction_valid else "all_fixed-cardinality_graphs",
        "domain_proof": (
            "At the necessary edge-count lower bound, all feasible graphs must have minimum degree f+1; "
            "the handshaking equality therefore forces every degree to equal f+1."
            if degree_reduction_valid else "Complete fixed-cardinality enumeration."
        ),
        "runtime_s": perf_counter() - started,
        "optimal_worst_lambda2": best_certificate.worst_lambda2,
        "edges": [list(edge) for edge in best_edges],
        "certificate": best_certificate.to_dict(include_scenarios=False),
    }
