from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import cvxpy as cp

from tnsm_sim.optimization import (
    enumerate_exact_max_z,
    round_relaxed_topology,
    solve_stage1_relaxation,
    solve_stage2_sdp,
)
from tnsm_sim.plotting.g3 import plot_g3_solver_gate

from .artifacts import ArtifactStore


def _relative_difference(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), 1e-12)


def run_g3_solver_gate(policy_path: Path, output_root: Path) -> Path:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    stage1_config, stage2_config = policy["stage1"], policy["stage2"]
    n, f, edge_count = (int(stage1_config[key]) for key in ("n", "f", "edge_count"))
    time_limit = float(policy["time_limit_s"])
    primary, independent = policy["primary_sdp_solver"], policy["independent_sdp_solver"]

    stage1_primary = solve_stage1_relaxation(n, f, edge_count, primary, time_limit)
    stage1_independent = solve_stage1_relaxation(n, f, edge_count, independent, time_limit)
    if stage1_primary["edge_fractions"] is None:
        raise RuntimeError("Stage 1 primary relaxation did not return an incumbent")
    edges, rounded_certificate, rounding = round_relaxed_topology(
        n, f, edge_count,
        [tuple(edge) for edge in stage1_primary["candidate_edges"]],
        stage1_primary["edge_fractions"],
        random_restarts=int(stage1_config["rounding_restarts"]),
        seed=int(stage1_config["rounding_seed"]),
    )
    exact = enumerate_exact_max_z(n, f, edge_count) if stage1_config["exact_enumeration"] else None
    relaxation_upper = float(stage1_primary["relaxation_upper_bound"])
    certified_lower = rounded_certificate.worst_lambda2
    exact_value = None if exact is None else float(exact["optimal_worst_lambda2"])
    stage1_summary = {
        "primary_solver": stage1_primary["solver"],
        "primary_status": stage1_primary["status"],
        "independent_solver": stage1_independent["solver"],
        "independent_status": stage1_independent["status"],
        "cross_solver_rel_difference": _relative_difference(
            relaxation_upper, float(stage1_independent["relaxation_upper_bound"])
        ),
        "relaxation_upper_bound": relaxation_upper,
        "certified_incumbent": certified_lower,
        "exact_optimum": exact_value,
        "relaxation_bound_gap": (relaxation_upper - certified_lower) / relaxation_upper,
        "certified_exact_gap": None if exact_value is None else (exact_value - certified_lower) / exact_value,
        "full_scenario_certificate": rounded_certificate.to_dict(include_scenarios=False),
        "z_req_pass": certified_lower >= float(stage1_config["z_req"]),
        "rounding": rounding,
        "exact_enumeration": exact,
        "selected_edges": [list(edge) for edge in edges],
    }

    stage2_primary = solve_stage2_sdp(
        n, f, edges, float(stage2_config["lambda_cap"]),
        float(stage2_config["total_weight_budget"]), float(stage2_config["w_max"]),
        primary, time_limit,
    )
    stage2_independent = solve_stage2_sdp(
        n, f, edges, float(stage2_config["lambda_cap"]),
        float(stage2_config["total_weight_budget"]), float(stage2_config["w_max"]),
        independent, time_limit,
    )
    gamma_primary, gamma_independent = float(stage2_primary["gamma"]), float(stage2_independent["gamma"])
    feasibility_tolerance = float(policy["numerical_feasibility_tolerance"])
    cross_tolerance = float(policy["cross_solver_relative_tolerance"])
    primary_rec = stage2_primary["numerical_recertification"]
    independent_rec = stage2_independent["numerical_recertification"]
    stage2_summary = {
        "primary_solver": primary,
        "primary_status": stage2_primary["status"],
        "independent_solver": independent,
        "independent_status": stage2_independent["status"],
        "primary_gamma": gamma_primary,
        "independent_gamma": gamma_independent,
        "cross_solver_rel_difference": _relative_difference(gamma_primary, gamma_independent),
        "primary_recertification": primary_rec,
        "independent_recertification": independent_rec,
        "cross_solver_pass": _relative_difference(gamma_primary, gamma_independent) <= cross_tolerance,
        "primary_feasibility_pass": max(
            primary_rec["lower_lmi_violation"], primary_rec["upper_lmi_violation"],
            primary_rec["weight_upper_violation"],
        ) <= feasibility_tolerance,
        "independent_feasibility_pass": max(
            independent_rec["lower_lmi_violation"], independent_rec["upper_lmi_violation"],
            independent_rec["weight_upper_violation"],
        ) <= feasibility_tolerance,
    }
    overall_pass = (
        stage1_primary["accepted_status"]
        and stage1_independent["accepted_status"]
        and stage1_summary["cross_solver_rel_difference"] <= cross_tolerance
        and stage1_summary["z_req_pass"]
        and rounded_certificate.connectivity_rate == 1.0
        and (exact_value is None or abs(exact_value - certified_lower) <= feasibility_tolerance)
        and stage2_primary["accepted_status"]
        and stage2_independent["accepted_status"]
        and stage2_summary["cross_solver_pass"]
        and stage2_summary["primary_feasibility_pass"]
        and stage2_summary["independent_feasibility_pass"]
    )
    summary = {
        "gate": "G3-A_formal_optimization",
        "passed": overall_pass,
        "claim_boundary": (
            "Stage 1 is exact only for this N=8 lower-bound-edge case because all feasible graphs are 3-regular "
            "and all 19,355 labeled candidates were enumerated. Larger cases retain SDP bounds and certified incumbents."
        ),
        "stage1": stage1_summary,
        "stage2": stage2_summary,
    }
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "cvxpy": cp.__version__,
        "installed_solvers": cp.installed_solvers(),
    }
    store = ArtifactStore(output_root, "G3A_formal_optimization_N8_f2")
    store.write_json("solver_policy.json", policy)
    store.write_json("environment.json", environment)
    store.write_json("stage1_primary.json", stage1_primary)
    store.write_json("stage1_independent.json", stage1_independent)
    store.write_json("stage2_primary.json", stage2_primary)
    store.write_json("stage2_independent.json", stage2_independent)
    store.write_json("summary.json", summary)
    plot_g3_solver_gate(store.path)
    return store.finalize()
