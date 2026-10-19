from __future__ import annotations

import json
import platform
import sys
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

import cvxpy as cp

from tnsm_sim.optimization import (
    enumerate_exact_max_z,
    round_relaxed_topology,
    solve_stage1_relaxation,
    solve_stage2_sdp,
)
from tnsm_sim.topology import certify_topology, minimum_degree_seed_edges
from tnsm_sim.topology.graph import complete_edges
from tnsm_sim.weights import lambda_cap_for_delay_margin, modal_delay_margin

from .artifacts import ArtifactStore


def _elapsed_call(function, *args, **kwargs):
    started = perf_counter()
    value = function(*args, **kwargs)
    return value, perf_counter() - started


def _none_or_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _relative_gap(upper: float | None, achieved: float | None) -> float | None:
    if upper is None or achieved is None:
        return None
    return max(0.0, float(upper) - float(achieved)) / max(abs(float(upper)), 1e-12)


def _hardware_record() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version,
        "cvxpy": cp.__version__,
        "installed_solvers": cp.installed_solvers(),
    }


def run_v21_g9(config_path: Path, output_root: Path) -> Path:
    """Run the offline Stage-1/Stage-2 scale study with full recertification."""
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    node_counts = [int(value) for value in config["node_counts"]]
    f = int(config["f"])
    degree = int(config.get("minimum_degree", config.get("minimum_regular_degree", f + 1)))
    if degree != f + 1:
        raise ValueError("minimum_degree must equal f+1 for the declared minimum-edge study")

    delay = dict(config["delay_requirement"])
    component_keys = (
        "state_message_period_s",
        "control_period_s",
        "admitted_network_tail_s",
        "engineering_allowance_s",
    )
    component_sum = sum(float(delay[key]) for key in component_keys)
    target_delay = float(delay["target_delay_margin_s"])
    if abs(component_sum - target_delay) > 1e-12:
        raise ValueError("delay-requirement components do not sum to target_delay_margin_s")
    k_p, k_v = float(config["k_p"]), float(config["k_v"])
    lambda_cap = lambda_cap_for_delay_margin(target_delay, k_p, k_v)

    store = ArtifactStore(output_root.resolve(), "V21_G9_offline_stage12_scale")
    resolved = dict(config)
    resolved["derived_lambda_cap"] = lambda_cap
    resolved["hardware"] = _hardware_record()
    store.write_json("protocol.json", resolved)
    store.write_json("source_identity.json", {
        "config_path": str(config_path),
        "config_sha256": sha256(config_path.read_bytes()).hexdigest(),
    })

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    exact_counts = {int(value) for value in config.get("exact_enumeration_node_counts", [])}
    solver = str(config["solver"])
    time_limit_s = float(config["time_limit_s"])
    random_restarts = int(config.get("rounding_random_restarts", 2))
    max_swap_rounds = int(config.get("rounding_max_swap_rounds", 10))
    seed = int(config.get("rounding_seed", 20260826))

    for n in node_counts:
        edge_count = (degree * n + 1) // 2
        candidates = complete_edges(n)
        relaxation, relaxation_total_s = _elapsed_call(
            solve_stage1_relaxation,
            n, f, edge_count, solver, time_limit_s, candidates,
        )
        fractions = relaxation.get("edge_fractions")
        if fractions is None:
            row = {
                "n": n,
                "f": f,
                "candidate_edges": len(candidates),
                "selected_edges": edge_count,
                "failure_scenarios": int(relaxation["failure_scenario_count"]),
                "stage1_status": relaxation.get("status"),
                "stage1_error": relaxation.get("error") or "no relaxation solution returned",
                "pipeline_pass": False,
            }
            rows.append(row)
            details[str(n)] = {"relaxation": relaxation, "summary_row": row}
            continue

        rounded, rounding_total_s = _elapsed_call(
            round_relaxed_topology,
            n, f, edge_count, candidates, fractions,
            max_swap_rounds=max_swap_rounds,
            random_restarts=random_restarts,
            seed=seed + n,
            initial_edges=minimum_degree_seed_edges(n, degree),
            preserve_degree=True,
        )
        rounded_edges, rounded_certificate, rounding_record = rounded
        exact = None
        exact_total_s = None
        selected_edges = rounded_edges
        selection_source = "continuous_upper_bound_plus_certified_minimum_degree_start_two_switch_repair"
        if n in exact_counts:
            exact, exact_total_s = _elapsed_call(enumerate_exact_max_z, n, f, edge_count)
            selected_edges = tuple(tuple(int(value) for value in edge) for edge in exact["edges"])
            selection_source = "complete_exact_enumeration"

        certificate, certification_total_s = _elapsed_call(certify_topology, n, selected_edges, f)
        stage2, stage2_total_s = _elapsed_call(
            solve_stage2_sdp,
            n, f, selected_edges, lambda_cap, float(edge_count),
            float(config["w_max"]), solver, time_limit_s,
        )
        numerical = stage2.get("numerical_recertification") or {}
        worst_lambda_max = _none_or_float(numerical.get("worst_lambda_max"))
        delay_margin = None if not worst_lambda_max or worst_lambda_max <= 0 else modal_delay_margin(
            worst_lambda_max, k_p, k_v
        )
        relaxation_bound = _none_or_float(relaxation.get("relaxation_upper_bound"))
        certified_z = float(certificate.worst_lambda2)
        exact_z = None if exact is None else float(exact["optimal_worst_lambda2"])
        tolerance = float(config["numerical_feasibility_tolerance"])
        stage2_pass = bool(
            stage2.get("accepted_status")
            and numerical
            and float(numerical["upper_lmi_violation"]) <= tolerance
            and float(numerical["lower_lmi_violation"]) <= tolerance
            and delay_margin is not None
            and delay_margin + tolerance >= target_delay
        )
        row = {
            "n": n,
            "f": f,
            "candidate_edges": len(candidates),
            "selected_edges": edge_count,
            "failure_scenarios": int(relaxation["failure_scenario_count"]),
            "stage1_status": relaxation.get("status"),
            "stage1_solver_s": _none_or_float(relaxation.get("wall_time_s")),
            "stage1_total_s": relaxation_total_s,
            "stage1_iterations": relaxation.get("num_iters"),
            "stage1_relaxation_upper_bound": relaxation_bound,
            "stage1_selection_source": selection_source,
            "rounding_repair_s": rounding_total_s,
            "recertification_s": certification_total_s,
            "certified_worst_lambda2": certified_z,
            "relaxation_upper_bound_gap": _relative_gap(relaxation_bound, certified_z),
            "connectivity_rate": float(certificate.connectivity_rate),
            "minimum_degree": int(certificate.minimum_degree),
            "degree_imbalance": float(certificate.degree_imbalance),
            "exact_enumeration_s": exact_total_s,
            "exact_evaluated_topologies": None if exact is None else int(exact["evaluated_topologies"]),
            "exact_worst_lambda2": exact_z,
            "selected_to_exact_gap": _relative_gap(exact_z, certified_z),
            "stage2_status": stage2.get("status"),
            "stage2_solver_s": _none_or_float(stage2.get("wall_time_s")),
            "stage2_total_s": stage2_total_s,
            "stage2_iterations": stage2.get("num_iters"),
            "stage2_gamma": _none_or_float(stage2.get("gamma")),
            "weighted_worst_lambda2": _none_or_float(numerical.get("worst_lambda2")),
            "weighted_worst_lambda_max": worst_lambda_max,
            "certified_delay_margin_s": delay_margin,
            "lower_lmi_violation": _none_or_float(numerical.get("lower_lmi_violation")),
            "upper_lmi_violation": _none_or_float(numerical.get("upper_lmi_violation")),
            "offline_pipeline_s": (
                relaxation_total_s + rounding_total_s + certification_total_s + stage2_total_s
                + (0.0 if exact_total_s is None else exact_total_s)
            ),
            "stage1_certificate_pass": certificate.connectivity_rate == 1.0 and certified_z > 1e-9,
            "stage2_certificate_pass": stage2_pass,
            "pipeline_pass": bool(
                relaxation.get("accepted_status")
                and certificate.connectivity_rate == 1.0
                and certified_z > 1e-9
                and stage2_pass
            ),
        }
        rows.append(row)
        details[str(n)] = {
            "relaxation": relaxation,
            "rounding": rounding_record,
            "rounded_edges": [list(edge) for edge in rounded_edges],
            "rounded_certificate": rounded_certificate.to_dict(include_scenarios=False),
            "exact_enumeration": exact,
            "selected_edges": [list(edge) for edge in selected_edges],
            "selected_certificate": certificate.to_dict(include_scenarios=False),
            "stage2": stage2,
            "summary_row": row,
        }
        store.write_json("progress.json", {
            "completed_node_counts": [int(item["n"]) for item in rows],
            "rows": rows,
        })

    fields = list(rows[0].keys())
    store.write_csv("offline_scale_summary.csv", rows, fields)
    store.write_json("offline_scale_details.json", details)
    summary = {
        "gate": "V2.1-G9",
        "experiment": "offline_stage1_stage2_scalability",
        "passed": len(rows) == len(node_counts) and all(bool(row.get("pipeline_pass")) for row in rows),
        "node_counts": node_counts,
        "lambda_cap": lambda_cap,
        "target_delay_margin_s": target_delay,
        "rows": rows,
        "claim_boundary": (
            "Complete-enumeration optimality is claimed only for node counts listed in exact_enumeration_node_counts. "
            "For larger fleets, the continuous relaxation is an upper bound and the rounded topology and Stage-2 "
            "weights are accepted only after explicit recertification over every declared failure set."
        ),
    }
    store.write_json("summary.json", summary)
    return store.finalize()
