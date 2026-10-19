from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from tnsm_sim.optimization import solve_stage2_sdp
from tnsm_sim.topology import certify_topology, graph_sha256
from tnsm_sim.weights import lambda_cap_for_delay_margin, modal_delay_margin

from .artifacts import ArtifactStore


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _relative_difference(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), 1e-12)


def evaluate_weights(n: int, f: int, edges: tuple[tuple[int, int], ...], values: np.ndarray,
                     k_p: float, k_v: float) -> dict[str, Any]:
    weights = {edge: float(value) for edge, value in zip(edges, values, strict=True)}
    certificate = certify_topology(n, edges, f, weights)
    lower = min(certificate.scenarios, key=lambda item: (item.lambda2, item.failed))
    upper = max(certificate.scenarios, key=lambda item: (item.lambda_max, tuple(-x for x in item.failed)))
    return {
        "worst_lambda2": float(lower.lambda2),
        "worst_lambda2_failure": list(lower.failed),
        "worst_lambda_max": float(upper.lambda_max),
        "worst_lambda_max_failure": list(upper.failed),
        "delay_margin_s": float(modal_delay_margin(upper.lambda_max, k_p, k_v)),
        "weight_sum": float(values.sum()),
        "weight_max": float(values.max()),
        "weight_min": float(values.min()),
    }


def _candidate_cloud(config: dict[str, Any], edges: tuple[tuple[int, int], ...]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(config["candidate_seed"]))
    count = int(config["candidate_samples"])
    budget = float(config["total_weight_budget"])
    w_max = float(config["w_max"])
    k_p, k_v = float(config["k_p"]), float(config["k_v"])
    rows: list[dict[str, Any]] = []
    for candidate_id in range(1, count + 1):
        sigma = float(rng.uniform(0.10, 1.00))
        raw = rng.lognormal(mean=0.0, sigma=sigma, size=len(edges))
        target_sum = float(rng.uniform(0.12 * budget, budget))
        values = raw * target_sum / raw.sum()
        values *= min(1.0, w_max / float(values.max()))
        metrics = evaluate_weights(int(config["n"]), int(config["f"]), edges, values, k_p, k_v)
        rows.append({
            "candidate_id": candidate_id,
            "sigma": sigma,
            **metrics,
            **{f"w_{i}_{j}": float(value) for (i, j), value in zip(edges, values, strict=True)},
        })
    return rows


def _frontier(config: dict[str, Any], edges: tuple[tuple[int, int], ...]) -> list[dict[str, Any]]:
    rows = []
    for cap in (float(value) for value in config["frontier_lambda_caps"]):
        result = solve_stage2_sdp(
            int(config["n"]), int(config["f"]), edges, cap,
            float(config["total_weight_budget"]), float(config["w_max"]),
            str(config["primary_solver"]), float(config["time_limit_s"]),
        )
        recertification = result["numerical_recertification"]
        if recertification is None or result["weights"] is None:
            raise RuntimeError(f"frontier solve did not return weights for lambda cap {cap}")
        values = np.asarray([float(result["weights"][f"{i}-{j}"]) for i, j in edges])
        direct = evaluate_weights(
            int(config["n"]), int(config["f"]), edges, values,
            float(config["k_p"]), float(config["k_v"]),
        )
        rows.append({
            "lambda_cap": cap,
            "solver": result["solver"],
            "status": result["status"],
            "accepted_status": bool(result["accepted_status"]),
            "gamma": float(result["gamma"]),
            **direct,
            "upper_lmi_violation": float(recertification["upper_lmi_violation"]),
            "lower_lmi_violation": float(recertification["lower_lmi_violation"]),
            "weights": result["weights"],
        })
    return rows


def _method_record(name: str, values: np.ndarray, config: dict[str, Any],
                   edges: tuple[tuple[int, int], ...]) -> dict[str, Any]:
    return {
        "method": name,
        **evaluate_weights(
            int(config["n"]), int(config["f"]), edges, values,
            float(config["k_p"]), float(config["k_v"]),
        ),
        "weights": {f"{i}-{j}": float(value) for (i, j), value in zip(edges, values, strict=True)},
    }


def run_v21_g3(config_path: Path, output_root: Path, project_root: Path = PROJECT_ROOT) -> Path:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = project_root.resolve()
    g1_source = (project_root / config["g1_source"]).resolve()
    exact = json.loads((g1_source / "exact_enumeration.json").read_text(encoding="utf-8"))
    topology_rows = config.get("topology_edges", exact["edges"])
    edges = tuple(tuple(int(value) for value in edge) for edge in topology_rows)
    if len(edges) != int(config["edge_count"]):
        raise RuntimeError("Frozen G3 topology does not match the declared edge count")
    unweighted_certificate = certify_topology(int(config["n"]), edges, int(config["f"]))

    delay_requirement = dict(config["delay_requirement"])
    component_sum = sum(float(delay_requirement[key]) for key in (
        "state_message_period_s", "control_period_s", "admitted_network_tail_s", "engineering_allowance_s"
    ))
    target_delay = float(delay_requirement["target_delay_margin_s"])
    if abs(component_sum - target_delay) > 1e-12:
        raise RuntimeError("delay-requirement components do not sum to target_delay_margin_s")
    selected_cap = lambda_cap_for_delay_margin(
        target_delay, float(config["k_p"]), float(config["k_v"])
    )
    resolved_config = dict(config)
    resolved_config["derived_selected_lambda_cap"] = selected_cap
    resolved_config["frontier_lambda_caps"] = sorted(set(
        float(value) for value in config["frontier_lambda_caps"]
    ) | {selected_cap})

    cloud = _candidate_cloud(resolved_config, edges)
    frontier = _frontier(resolved_config, edges)
    selected_primary = next(row for row in frontier if abs(row["lambda_cap"] - selected_cap) < 1e-12)
    selected_values = np.asarray([float(selected_primary["weights"][f"{i}-{j}"]) for i, j in edges])

    independent = solve_stage2_sdp(
        int(config["n"]), int(config["f"]), edges, selected_cap,
        float(config["total_weight_budget"]), float(config["w_max"]),
        str(config["independent_solver"]), float(config["time_limit_s"]),
    )
    independent_values = np.asarray([float(independent["weights"][f"{i}-{j}"]) for i, j in edges])
    independent_direct = evaluate_weights(
        int(config["n"]), int(config["f"]), edges, independent_values,
        float(config["k_p"]), float(config["k_v"]),
    )

    uniform_values = np.full(len(edges), float(config["total_weight_budget"]) / len(edges))
    convergence_result = solve_stage2_sdp(
        int(config["n"]), int(config["f"]), edges, 100.0,
        float(config["total_weight_budget"]), float(config["w_max"]),
        str(config["primary_solver"]), float(config["time_limit_s"]),
    )
    convergence_values = np.asarray([float(convergence_result["weights"][f"{i}-{j}"]) for i, j in edges])
    methods = [
        _method_record("B7_uniform", uniform_values, config, edges),
        _method_record("B8_convergence_priority", convergence_values, config, edges),
        _method_record("S2_delay_constrained", selected_values, config, edges),
    ]

    cloud_best_by_cap = {}
    for row in frontier:
        eligible = [candidate["worst_lambda2"] for candidate in cloud
                    if candidate["worst_lambda_max"] <= row["lambda_cap"] + 1e-12]
        cloud_best_by_cap[str(row["lambda_cap"])] = max(eligible, default=None)

    tolerance = float(config["numerical_feasibility_tolerance"])
    cross_tolerance = float(config["cross_solver_relative_tolerance"])
    frontier_feasible = all(
        row["accepted_status"]
        and row["upper_lmi_violation"] <= tolerance
        and row["lower_lmi_violation"] <= tolerance
        and row["weight_sum"] <= float(config["total_weight_budget"]) + tolerance
        and row["weight_max"] <= float(config["w_max"]) + tolerance
        for row in frontier
    )
    cross_difference = _relative_difference(selected_primary["gamma"], float(independent["gamma"]))
    selected_direct_difference = _relative_difference(
        selected_primary["worst_lambda2"], independent_direct["worst_lambda2"]
    )
    cloud_dominance_pass = all(
        best is None or row["worst_lambda2"] + tolerance >= best
        for row in frontier
        for best in [cloud_best_by_cap[str(row["lambda_cap"])]]
    )
    passed = frontier_feasible and cross_difference <= cross_tolerance and cloud_dominance_pass

    summary = {
        "gate": "V2.1-G3",
        "passed": passed,
        "topology": {
            "source": str(g1_source),
            "selection_policy": config.get("topology_selection_policy", "G1 exact max-z tie-break solution"),
            "graph_sha256": graph_sha256(int(config["n"]), edges),
            "edges": [list(edge) for edge in edges],
            "stage1_worst_lambda2_unweighted": float(min(
                item.lambda2 for item in unweighted_certificate.scenarios
            )),
            "stage1_connectivity_rate": float(unweighted_certificate.connectivity_rate),
        },
        "candidate_cloud": {
            "sample_count": len(cloud),
            "seed": int(config["candidate_seed"]),
            "cloud_best_by_cap": cloud_best_by_cap,
            "frontier_dominates_cloud_pass": cloud_dominance_pass,
        },
        "frontier": {
            "point_count": len(frontier),
            "all_points_feasible": frontier_feasible,
            "claim": "constraint-generated robust frontier over all declared failure sets; no intrinsic knee is asserted",
        },
        "selection": {
            "rule": config["selection_rule"],
            "lambda_cap": selected_cap,
            "delay_requirement": {**delay_requirement, "component_sum_s": component_sum},
            "primary": selected_primary,
            "independent_solver": independent["solver"],
            "independent_status": independent["status"],
            "independent_direct": independent_direct,
            "cross_solver_gamma_relative_difference": cross_difference,
            "direct_lambda2_relative_difference": selected_direct_difference,
            "cross_solver_pass": cross_difference <= cross_tolerance,
        },
        "methods": methods,
        "claim_boundary": (
            "The Pareto evidence is conditional on the fixed G1 S1 topology, N=8, f=2, the declared 37 failure sets, "
            "the second-order common time-aligned delay model, and the frozen controller gains. It does not certify "
            "heterogeneous time-varying ages or establish a universal data-free knee point."
        ),
    }

    store = ArtifactStore(output_root.resolve(), "V21_G3_stage2_weights_N8_f2")
    store.write_json("protocol.json", resolved_config)
    store.write_json("source_identity.json", {
        "config_sha256": sha256(config_path.read_bytes()).hexdigest(),
        "g1_manifest_sha256": sha256((g1_source / "manifest.sha256").read_bytes()).hexdigest(),
    })
    cloud_fields = list(cloud[0].keys())
    store.write_csv("candidate_cloud.csv", cloud, cloud_fields)
    frontier_rows = [{**row, "weights": json.dumps(row["weights"], sort_keys=True)} for row in frontier]
    store.write_csv("pareto_frontier.csv", frontier_rows, list(frontier_rows[0].keys()))
    store.write_json("selected_primary.json", selected_primary)
    store.write_json("selected_independent.json", {**independent, "direct_recertification": independent_direct})
    store.write_json("method_comparison.json", methods)
    store.write_json("summary.json", summary)
    return store.finalize()
