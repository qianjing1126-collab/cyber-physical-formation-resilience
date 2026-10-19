from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from math import sqrt
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import ArtifactStore
from .v21_g6 import simulate_sequential_recovery


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def balanced_failure_pair_schedule(n: int, count: int) -> dict[str, list[tuple[int, int]]]:
    if n < 4 or count < 1:
        raise ValueError("G7 pair schedule requires n>=4 and at least one repetition")
    pools = {
        "A_adjacent": [(first, first + 1) for first in range(1, n)],
        "N_nonadjacent": [
            (first, second)
            for first in range(1, n + 1)
            for second in range(first + 2, n + 1)
        ],
    }
    return {
        pattern: [pool[index % len(pool)] for index in range(count)]
        for pattern, pool in pools.items()
    }


def _wilson(successes: int, count: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    half = z * sqrt(
        proportion * (1.0 - proportion) / count + z * z / (4.0 * count**2)
    ) / denominator
    return center - half, center + half


def _bootstrap(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[index].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _resolved_schedule(config: dict[str, Any]) -> dict[str, list[tuple[int, int]]]:
    repetitions = int(config["seed_count"])
    if config.get("failure_pair_mode") == "balanced_all_pairs":
        return balanced_failure_pair_schedule(int(config["n"]), repetitions)
    return {
        pattern: [tuple(int(value) for value in targets)] * repetitions
        for pattern, targets in config["failure_patterns"].items()
    }


def _run_g7_case(payload: tuple[Any, ...]) -> dict[str, Any]:
    (
        repetition, layout, fault_pattern, targets, method, seed, edges,
        weights, network_parameters, scenario_config,
    ) = payload
    metrics, _, _, _ = simulate_sequential_recovery(
        layout=layout, seed=seed, edges=edges, weights=weights,
        network_parameters=network_parameters, config=scenario_config,
        keep_trace=False, recovery_method=method,
    )
    return {
        "repetition": repetition,
        "layout": layout,
        "fault_pattern": fault_pattern,
        "targets": ",".join(str(value) for value in targets),
        "method": method,
        **metrics,
    }


def run_v21_g7(
    config_path: Path,
    output_root: Path,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    project_root = project_root.resolve()
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    g3 = json.loads(
        (project_root / config["g3_source"] / "summary.json").read_text(encoding="utf-8")
    )
    profiles = json.loads(
        (project_root / config["network_profile_file"]).read_text(encoding="utf-8")
    )
    weight_record = next(
        item for item in g3["methods"] if item["method"] == config["weight_method"]
    )
    edges = tuple(tuple(int(value) for value in edge) for edge in g3["topology"]["edges"])
    network_parameters = profiles[str(config["network_regime"])]
    seeds = list(range(
        int(config["seed_start"]),
        int(config["seed_start"]) + int(config["seed_count"]),
    ))
    schedule = _resolved_schedule(config)
    records: list[dict[str, Any]] = []

    tasks = []
    for repetition, seed in enumerate(seeds, start=1):
        for layout in config["layouts"]:
            for fault_pattern, pairs in schedule.items():
                targets = pairs[repetition - 1]
                scenario_config = deepcopy(config)
                scenario_config["failure_events"] = [{
                    "vehicle_ids": list(targets),
                    "failure_time_s": float(config["failure_time_s"]),
                    "confirmation_time_s": float(config["confirmation_time_s"]),
                }]
                for method in config["methods"]:
                    tasks.append((
                        repetition, str(layout), str(fault_pattern), targets, str(method),
                        seed, edges, weight_record["weights"], network_parameters,
                        scenario_config,
                    ))
    workers = int(config.get("parallel_workers", min(12, os.cpu_count() or 1)))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(_run_g7_case, tasks, chunksize=1))

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["layout"]), str(row["fault_pattern"]), str(row["method"]))].append(row)
    groups = []
    for index, (key, rows) in enumerate(sorted(grouped.items())):
        layout, fault_pattern, method = key
        successes = sum(bool(row["all_recoveries_successful"]) for row in rows)
        success_low, success_high = _wilson(successes, len(rows))
        rte = [float(row["event1_R_Te"]) for row in rows]
        rte_low, rte_high = _bootstrap(rte, int(config["bootstrap_samples"]), 20261000 + index)
        q_min = [float(row["event1_Q_min"]) for row in rows]
        qmin_low, qmin_high = _bootstrap(q_min, int(config["bootstrap_samples"]), 20261500 + index)
        recovery_times = [float(row["event1_recovery_time_s"]) for row in rows]
        time_low, time_high = _bootstrap(
            recovery_times, int(config["bootstrap_samples"]), 20262000 + index
        )
        groups.append({
            "layout": layout,
            "fault_pattern": fault_pattern,
            "method": method,
            "n": len(rows),
            "recovery_success_count": successes,
            "recovery_success_rate": successes / len(rows),
            "success_ci_low": success_low,
            "success_ci_high": success_high,
            "R_Te_mean": float(np.mean(rte)),
            "R_Te_ci_low": rte_low,
            "R_Te_ci_high": rte_high,
            "Q_min_mean": float(np.mean(q_min)),
            "Q_min_ci_low": qmin_low,
            "Q_min_ci_high": qmin_high,
            "recovery_time_RMST_mean_s": float(np.mean(recovery_times)),
            "recovery_time_ci_low_s": time_low,
            "recovery_time_ci_high_s": time_high,
            "minimum_clearance_worst_m": min(float(row["minimum_clearance_m"]) for row in rows),
            "peak_abs_ax_worst_mps2": max(float(row["peak_abs_ax_mps2"]) for row in rows),
            "peak_abs_ay_worst_mps2": max(float(row["peak_abs_ay_mps2"]) for row in rows),
            "planner_time_median_us": float(np.median([
                float(row["event1_planner_time_s"]) * 1.0e6 for row in rows
            ])),
        })

    patterns = tuple(schedule)
    scenario_spreads = {}
    for layout in config["layouts"]:
        for fault_pattern in patterns:
            values = [
                float(row["recovery_success_rate"])
                for row in groups
                if row["layout"] == layout and row["fault_pattern"] == fault_pattern
            ]
            scenario_spreads[f"{layout}/{fault_pattern}"] = max(values) - min(values)
    success_informative = max(scenario_spreads.values()) >= 0.20

    pair_counts = {
        pattern: {
            f"{first}-{second}": count
            for (first, second), count in sorted(Counter(pairs).items())
        }
        for pattern, pairs in schedule.items()
    }
    pair_balance = all(
        max(counts.values()) - min(counts.values()) <= 1
        for counts in pair_counts.values()
    )
    paired_network = all(
        len({
            (
                round(float(row["pdr"]), 14),
                round(float(row["delay_p95_s"]), 14),
                round(float(row["aoi_p95_s"]), 14),
            )
            for row in records
            if int(row["repetition"]) == repetition
            and row["layout"] == layout
            and row["fault_pattern"] == fault_pattern
        }) == 1
        for repetition in range(1, len(seeds) + 1)
        for layout in config["layouts"]
        for fault_pattern in patterns
    )
    b11_s3_parity = all(
        abs(float(b11[metric]) - float(s3[metric])) <= 1e-12
        for repetition in range(1, len(seeds) + 1)
        for layout in config["layouts"]
        for fault_pattern in patterns
        for metric in ("event1_R_Te", "event1_recovery_time_s", "event1_Q_min")
        for b11 in [next(
            row for row in records
            if int(row["repetition"]) == repetition and row["layout"] == layout
            and row["fault_pattern"] == fault_pattern and row["method"] == "B11"
        )]
        for s3 in [next(
            row for row in records
            if int(row["repetition"]) == repetition and row["layout"] == layout
            and row["fault_pattern"] == fault_pattern and row["method"] == "S3"
        )]
    )
    expected = len(seeds) * len(config["layouts"]) * len(patterns) * len(config["methods"])
    acceptance = {
        "run_count_pass": len(records) == expected,
        "failure_pair_balance_pass": pair_balance,
        "all_runs_safe_pass": all(bool(row["safety_ok"]) for row in records),
        "paired_network_process_pass": paired_network,
        "b11_s3_physical_recovery_parity_pass": b11_s3_parity,
    }
    diagnostic = bool(config.get("diagnostic_only", False))
    summary = {
        "gate": config["gate"],
        "diagnostic_only": diagnostic,
        "passed": all(acceptance.values()),
        "acceptance": acceptance,
        "run_count": len(records),
        "failure_pair_counts": pair_counts,
        "success_rate_is_informative": success_informative,
        "recommended_first_panel": (
            "recovery_success_rate" if success_informative else "recovery_time_s"
        ),
        "success_rate_spread_by_scenario": scenario_spreads,
        "groups": groups,
        "claim_boundary": config["claim_boundary"],
    }
    run_name = str(config.get(
        "run_name",
        "V21_G7_pilot_simultaneous_N8" if diagnostic else "V21_G7_formal_simultaneous_N8",
    ))
    store = ArtifactStore(output_root.resolve(), run_name)
    store.write_json("protocol.json", config)
    store.write_json("failure_pair_schedule.json", {
        pattern: [list(pair) for pair in pairs] for pattern, pairs in schedule.items()
    })
    store.write_json("summary.json", summary)
    record_name = "pilot_records.csv" if diagnostic else "run_records.csv"
    group_name = "pilot_group_summary.csv" if diagnostic else "group_summary.csv"
    store.write_csv(record_name, records, list(records[0]))
    store.write_csv(group_name, groups, list(groups[0]))
    return store.finalize()


def run_v21_g7_pilot(
    config_path: Path,
    output_root: Path,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    return run_v21_g7(config_path, output_root, project_root)
