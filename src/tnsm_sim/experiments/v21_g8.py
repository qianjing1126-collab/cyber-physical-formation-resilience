from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from tnsm_sim.recovery import build_recovery_plan
from tnsm_sim.topology import minimum_degree_seed_edges
from tnsm_sim.topology.graph import spectral_bounds
from tnsm_sim.types import VehicleState

from .artifacts import ArtifactStore
from .v21_g6 import layout_offsets, simulate_sequential_recovery


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAFE_HOLD_METHOD = "SAFE_HOLD"


def scale_failure_pairs(n: int) -> dict[str, tuple[int, int]]:
    if n < 6:
        raise ValueError("G8 requires a fleet size of at least six")
    adjacent = (n // 2, n // 2 + 1)
    nonadjacent = (max(1, n // 3), min(n, 2 * n // 3 + 1))
    if nonadjacent[1] - nonadjacent[0] < 2:
        raise ValueError("normalized G8 nonadjacent pair is invalid")
    return {"A_adjacent": adjacent, "N_nonadjacent": nonadjacent}


def scale_topology(n: int, spectral_upper_bound: float) -> tuple[
    tuple[tuple[int, int], ...], dict[str, float], float, float
]:
    edges = minimum_degree_seed_edges(n, 3)
    _, unweighted_max = spectral_bounds(list(range(1, n + 1)), edges)
    scalar = spectral_upper_bound / unweighted_max
    weights = {f"{first}-{second}": scalar for first, second in edges}
    lambda2, lambda_max = spectral_bounds(
        list(range(1, n + 1)), edges,
        {(first, second): scalar for first, second in edges},
    )
    return edges, weights, lambda2, lambda_max


def certified_scale_topology(
    n: int, offline_designs: dict[str, Any],
) -> tuple[tuple[tuple[int, int], ...], dict[str, float], float, float]:
    """Return the Stage-1/Stage-2 design reported in the offline scale study."""
    try:
        design = offline_designs[str(n)]
    except KeyError as exc:
        raise ValueError(f"offline scale design is missing N={n}") from exc
    if not bool(design["summary_row"]["pipeline_pass"]):
        raise ValueError(f"offline scale design for N={n} did not pass recertification")
    edges = tuple(tuple(int(value) for value in edge) for edge in design["selected_edges"])
    weights = {
        str(edge): float(value) for edge, value in design["stage2"]["weights"].items()
    }
    recertification = design["stage2"]["numerical_recertification"]
    return (
        edges,
        weights,
        float(recertification["worst_lambda2"]),
        float(recertification["worst_lambda_max"]),
    )


def communication_overhead(
    n: int, *, duration_s: float, failure_time_s: float,
    message_period_s: float, packet_size_byte: int, failed_count: int = 2,
) -> list[dict[str, Any]]:
    pre_updates = int(round(failure_time_s / message_period_s))
    post_updates = int(round((duration_s - failure_time_s) / message_period_s))
    active_updates = n * pre_updates + (n - failed_count) * post_updates
    result = []
    for method, transmissions_per_vehicle in (("B11", 2), ("B10", 2), ("S3", 1)):
        packets = transmissions_per_vehicle * active_updates
        if method == "B11":
            counting_rule = "state uplink plus per-vehicle command downlink"
        elif method == "B10":
            counting_rule = "state uplink plus infrastructure relay broadcast; control remains distributed"
        else:
            counting_rule = "one local broadcast per active vehicle update"
        result.append({
            "n": n,
            "method": method,
            "wireless_packets": packets,
            "wireless_bytes": packets * packet_size_byte,
            "wireless_megabytes": packets * packet_size_byte / 1.0e6,
            "counting_rule": counting_rule,
        })
    return result


def _bootstrap(values: list[float], samples: int, seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def architecture_path_interruption(
    method_means: dict[str, float], architecture: str,
) -> dict[str, Any]:
    """Map a declared message-path interruption to the tested control policy.

    B10/B11 lose their relay/controller path and issue no new coordinated
    recovery plan.  Their vehicles keep the last valid slot references under
    low-level tracking and safety control.  S3 does not use the interrupted
    path, so its direct-V2V recovery remains available.
    """
    required = {"S3", SAFE_HOLD_METHOD}
    missing = required - set(method_means)
    if missing:
        raise ValueError(f"missing architecture diagnostic methods: {sorted(missing)}")
    if architecture not in {"B11", "B10", "S3"}:
        raise ValueError(f"unsupported architecture {architecture}")
    normal = float(method_means["S3"])
    interrupted = (
        normal if architecture == "S3" else float(method_means[SAFE_HOLD_METHOD])
    )
    return {
        "normal_R_Te": normal,
        "path_interruption_R_Te": interrupted,
        "path_interruption_retention_percent": 100.0 * interrupted / normal,
        "interruption_policy": (
            "declared relay/controller path is not used; direct V2V recovery remains available"
            if architecture == "S3"
            else "declared relay/controller path removed; no new coordinated recovery plan; last valid slot references retained under low-level safety control"
        ),
    }


def _run_g8_case(payload: tuple[Any, ...]) -> dict[str, Any]:
    (
        n, seed, pattern, targets, method, enable_recovery,
        edges, weights, network_parameters, scenario,
    ) = payload
    recovery_method = "S3" if method == SAFE_HOLD_METHOD else method
    metrics, _, _, _ = simulate_sequential_recovery(
        layout=str(scenario["layout"]), seed=seed, edges=edges,
        weights=weights, network_parameters=network_parameters,
        config=scenario, keep_trace=False, enable_recovery=enable_recovery,
        recovery_method=recovery_method,
        communication_cutoff_time_s=(
            None if enable_recovery else float(scenario["confirmation_time_s"])
        ),
    )
    return {
        "n": n,
        "seed": seed,
        "fault_pattern": pattern,
        "targets": f"{targets[0]},{targets[1]}",
        "method": method,
        "control_policy": (
            "coordinated_recovery" if enable_recovery
            else "safe_hold_last_valid_slots_no_new_recovery_plan"
        ),
        **metrics,
    }


def _planner_states(n: int, layout: str, spacing: float, delta: float) -> tuple[
    dict[int, VehicleState], dict[int, int], dict[int, float], set[int], tuple[float, ...]
]:
    offsets = layout_offsets(layout, n, spacing, delta)
    failed = set(scale_failure_pairs(n)["N_nonadjacent"])
    states = {
        vehicle_id: VehicleState(
            t=15.5, vehicle_id=vehicle_id, x=offsets[vehicle_id][0],
            y=offsets[vehicle_id][1], s=offsets[vehicle_id][0],
            d=offsets[vehicle_id][1], heading=0.0, v=5.0, v_s=5.0, v_d=0.0,
            active=vehicle_id not in failed, slot_rank=vehicle_id,
            coverage_width=4.0,
        )
        for vehicle_id in range(1, n + 1)
    }
    survivors = [vehicle_id for vehicle_id in range(1, n + 1) if vehicle_id not in failed]
    target_offsets = layout_offsets(layout, len(survivors), spacing, delta)
    return (
        states,
        {vehicle_id: vehicle_id for vehicle_id in range(1, n + 1)},
        {vehicle_id: offsets[vehicle_id][0] for vehicle_id in survivors},
        failed,
        tuple(target_offsets[rank][0] for rank in range(1, len(survivors) + 1)),
    )


def benchmark_plan_construction(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Benchmark only recovery-plan construction, excluding common safety screening.

    B10 and S3 execute the same vehicle-side S3 construction rule, so the
    benchmark reports the two algorithmically distinct constructions: B11's
    centralized assignment and S3's rank-compression construction.
    """
    raw = []
    summary = []
    warmup = int(config["planner_warmup_calls"])
    batches = int(config["planner_batches"])
    calls = int(config["planner_calls_per_batch"])
    for n in config["node_counts"]:
        states, pre_ranks, current_offsets, failed, target_offsets = _planner_states(
            int(n), str(config["layout"]), float(config["lateral_spacing_m"]),
            float(config["longitudinal_delta_m"]),
        )
        arguments = dict(
            t_confirm=float(config["confirmation_time_s"]), states=states,
            pre_ranks=pre_ranks, failed_ids=failed, current_s_offsets=current_offsets,
            rank_offsets_s=target_offsets, required_overlap=float(config["required_overlap_m"]),
            v_rec_max=float(config["recovery_max_offset_speed_mps"]),
            a_rec_max=float(config["recovery_max_offset_acceleration_mps2"]),
            min_duration=float(config["recovery_minimum_duration_s"]), center=0.0,
        )
        for method, planner_method in (("B11", "B11"), ("B10/S3", "S3")):
            def planned_call():
                return build_recovery_plan(method=planner_method, **arguments)
            for _ in range(warmup):
                planned_call()
            samples = []
            for batch in range(1, batches + 1):
                started = perf_counter()
                for _ in range(calls):
                    planned_call()
                elapsed_us = (perf_counter() - started) * 1.0e6 / calls
                samples.append(elapsed_us)
                raw.append({"n": n, "method": method, "batch": batch, "construction_time_us": elapsed_us})
            stats = _bootstrap(samples, int(config["bootstrap_samples"]), 20268000 + int(n) * 10 + len(summary))
            summary.append({
                "n": n, "method": method, "batches": batches, "calls_per_batch": calls,
                "median_construction_time_us": float(np.median(samples)),
                "mean_construction_time_us": stats["mean"],
                "mean_ci95_low_us": stats["ci95_low"],
                "mean_ci95_high_us": stats["ci95_high"],
            })
    return raw, summary


def run_v21_g8(
    config_path: Path, output_root: Path, project_root: Path = PROJECT_ROOT,
) -> Path:
    project_root = project_root.resolve()
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    profiles = json.loads(
        (project_root / config["network_profile_file"]).read_text(encoding="utf-8")
    )
    network_parameters = profiles[str(config["network_regime"])]
    offline_designs: dict[str, Any] | None = None
    if config.get("offline_design_file"):
        offline_path = (project_root / str(config["offline_design_file"])).resolve()
        offline_designs = json.loads(offline_path.read_text(encoding="utf-8"))
    seeds = range(int(config["seed_start"]), int(config["seed_start"]) + int(config["seed_count"]))
    records: list[dict[str, Any]] = []
    topology_records = []
    tasks = []

    for n_value in config["node_counts"]:
        n = int(n_value)
        if offline_designs is None:
            edges, weights, lambda2, lambda_max = scale_topology(
                n, float(config["spectral_upper_bound"])
            )
            topology_source = "minimum-degree seed with uniformly scaled weights"
        else:
            edges, weights, lambda2, lambda_max = certified_scale_topology(
                n, offline_designs
            )
            topology_source = "offline Stage-1/Stage-2 scale design"
        topology_records.append({
            "n": n, "edge_count": len(edges),
            "lambda2": lambda2, "lambda_max": lambda_max,
            "topology_source": topology_source,
            "edges": json.dumps(edges),
            "weights": json.dumps(weights, sort_keys=True),
        })
        for seed in seeds:
            for pattern, targets in scale_failure_pairs(n).items():
                scenario = deepcopy(config)
                scenario["n"] = n
                scenario["failure_events"] = [{
                    "vehicle_ids": list(targets),
                    "failure_time_s": float(config["failure_time_s"]),
                    "confirmation_time_s": float(config["confirmation_time_s"]),
                }]
                for method in config["methods"]:
                    tasks.append((
                        n, int(seed), str(pattern), targets, str(method), True, edges, weights,
                        network_parameters, scenario,
                    ))
                tasks.append((
                    n, int(seed), str(pattern), targets, SAFE_HOLD_METHOD, False,
                    edges, weights, network_parameters, scenario,
                ))

    workers = int(config.get("parallel_workers", min(12, os.cpu_count() or 1)))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(_run_g8_case, tasks, chunksize=1))

    paired_gain_rows = []
    resilience_summary = []
    for index, n_value in enumerate(config["node_counts"]):
        n = int(n_value)
        gains = []
        absolute = []
        for seed in seeds:
            means = {}
            for method in config["methods"]:
                values = [
                    float(row["event1_R_Te"]) for row in records
                    if int(row["n"]) == n and int(row["seed"]) == seed and row["method"] == method
                ]
                means[str(method)] = float(np.mean(values))
            gain = 100.0 * (means["S3"] - means["B9"]) / means["B9"]
            gains.append(gain)
            absolute.append(means["S3"] - means["B9"])
            paired_gain_rows.append({
                "n": n, "seed": seed, "B9_R_Te": means["B9"], "S3_R_Te": means["S3"],
                "absolute_gain": means["S3"] - means["B9"], "relative_gain_percent": gain,
            })
        gain_stats = _bootstrap(gains, int(config["bootstrap_samples"]), 20268100 + index)
        absolute_stats = _bootstrap(absolute, int(config["bootstrap_samples"]), 20268200 + index)
        resilience_summary.append({
            "n": n, "paired_samples": len(gains),
            "relative_gain_mean_percent": gain_stats["mean"],
            "relative_gain_ci95_low_percent": gain_stats["ci95_low"],
            "relative_gain_ci95_high_percent": gain_stats["ci95_high"],
            "absolute_gain_mean": absolute_stats["mean"],
            "absolute_gain_ci95_low": absolute_stats["ci95_low"],
            "absolute_gain_ci95_high": absolute_stats["ci95_high"],
        })

    planner_raw, planner_summary = benchmark_plan_construction(config)
    communication = [
        row for n in config["node_counts"] for row in communication_overhead(
            int(n), duration_s=float(config["duration_s"]),
            failure_time_s=float(config["failure_time_s"]),
            message_period_s=float(config["message_period_s"]),
            packet_size_byte=int(config["packet_size_byte"]),
        )
    ]
    architecture_rows = []
    architecture_summary = []
    for n_value in config["node_counts"]:
        n = int(n_value)
        for seed in seeds:
            method_means: dict[str, float] = {
                method: float(np.mean([
                    float(row["event1_R_Te"]) for row in records
                    if int(row["n"]) == n and int(row["seed"]) == seed
                    and row["method"] == method
                ]))
                for method in ("S3", SAFE_HOLD_METHOD)
            }
            for architecture in ("B11", "B10", "S3"):
                architecture_rows.append({
                    "n": n, "seed": seed, "architecture": architecture,
                    **architecture_path_interruption(method_means, architecture),
                })
        for architecture in ("B11", "B10", "S3"):
            subset = [
                row for row in architecture_rows
                if int(row["n"]) == n and row["architecture"] == architecture
            ]
            normal_stats = _bootstrap(
                [float(row["normal_R_Te"]) for row in subset],
                int(config["bootstrap_samples"]), 20268300 + n * 10 + len(architecture_summary),
            )
            interruption_stats = _bootstrap(
                [float(row["path_interruption_R_Te"]) for row in subset],
                int(config["bootstrap_samples"]), 20268400 + n * 10 + len(architecture_summary),
            )
            retention_stats = _bootstrap(
                [float(row["path_interruption_retention_percent"]) for row in subset],
                int(config["bootstrap_samples"]), 20268500 + n * 10 + len(architecture_summary),
            )
            architecture_summary.append({
                "n": n, "architecture": architecture,
                "normal_R_Te_mean": normal_stats["mean"],
                "path_interruption_R_Te_mean": interruption_stats["mean"],
                "path_interruption_R_Te_ci95_low": interruption_stats["ci95_low"],
                "path_interruption_R_Te_ci95_high": interruption_stats["ci95_high"],
                "path_interruption_retention_mean_percent": retention_stats["mean"],
                "path_interruption_retention_ci95_low_percent": retention_stats["ci95_low"],
                "path_interruption_retention_ci95_high_percent": retention_stats["ci95_high"],
            })
    paired_network = all(
        len({
            (round(float(row["pdr"]), 14), round(float(row["delay_p95_s"]), 14), round(float(row["aoi_p95_s"]), 14))
            for row in records
            if int(row["n"]) == int(n) and int(row["seed"]) == int(seed)
            and row["fault_pattern"] == pattern
            and row["method"] in config["methods"]
        }) == 1
        for n in config["node_counts"] for seed in seeds for pattern in scale_failure_pairs(int(n))
    )
    expected = len(config["node_counts"]) * len(list(seeds)) * 2 * (len(config["methods"]) + 1)
    acceptance = {
        "run_count_pass": len(records) == expected,
        "all_runs_safe_pass": all(bool(row["safety_ok"]) for row in records),
        "paired_network_process_pass": paired_network,
        "minimum_edge_topology_pass": all(
            int(row["edge_count"]) == (3 * int(row["n"]) + 1) // 2
            for row in topology_records
        ),
        "spectral_upper_bound_pass": all(
            float(row["lambda_max"]) <= float(config["spectral_upper_bound"]) + 5.0e-5
            for row in topology_records
        ),
        "communication_accounting_pass": all(
            next(row["wireless_bytes"] for row in communication if row["n"] == n and row["method"] == "B11")
            == 2 * next(row["wireless_bytes"] for row in communication if row["n"] == n and row["method"] == "S3")
            == next(row["wireless_bytes"] for row in communication if row["n"] == n and row["method"] == "B10")
            for n in config["node_counts"]
        ),
        "safe_hold_policy_pass": all(
            not bool(row["recovery_enabled"])
            and float(row["event1_plan_duration_s"]) == 0.0
            and float(row["event1_planner_time_s"]) == 0.0
            and abs(
                float(row["communication_cutoff_time_s"])
                - float(config["confirmation_time_s"])
            ) <= 1.0e-12
            for row in records if row["method"] == SAFE_HOLD_METHOD
        ),
        "architecture_path_interruption_emulation_pass": all(
            (
                abs(float(row["normal_R_Te"]) - float(row["path_interruption_R_Te"])) <= 1.0e-12
                if row["architecture"] == "S3"
                else float(row["path_interruption_R_Te"]) < float(row["normal_R_Te"])
            )
            for row in architecture_rows
        ),
    }
    summary = {
        "gate": config["gate"], "diagnostic_only": bool(config.get("diagnostic_only", False)),
        "passed": all(acceptance.values()), "acceptance": acceptance,
        "run_count": len(records), "topologies": topology_records,
        "resilience": resilience_summary, "planner": planner_summary,
        "communication": communication, "architecture": architecture_summary,
        "claim_boundary": config["claim_boundary"],
    }
    store = ArtifactStore(output_root.resolve(), str(config["run_name"]))
    store.write_json("protocol.json", config)
    store.write_json("summary.json", summary)
    store.write_csv("run_records.csv", records, list(records[0]))
    store.write_csv("paired_resilience_gain.csv", paired_gain_rows, list(paired_gain_rows[0]))
    store.write_csv("resilience_summary.csv", resilience_summary, list(resilience_summary[0]))
    store.write_csv("plan_construction_raw.csv", planner_raw, list(planner_raw[0]))
    store.write_csv("plan_construction_summary.csv", planner_summary, list(planner_summary[0]))
    store.write_csv("communication_summary.csv", communication, list(communication[0]))
    store.write_csv("architecture_path_interruption_records.csv", architecture_rows, list(architecture_rows[0]))
    store.write_csv("architecture_path_interruption_summary.csv", architecture_summary, list(architecture_summary[0]))
    store.write_csv("topology_summary.csv", topology_records, list(topology_records[0]))
    return store.finalize()
