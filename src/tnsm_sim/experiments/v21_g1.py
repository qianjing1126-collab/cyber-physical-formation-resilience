from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from tnsm_sim.optimization import enumerate_exact_max_z
from tnsm_sim.topology import (
    certify_topology,
    circulant_regular_edges,
    graph_sha256,
    greedy_edge_removal,
    random_connected_edges,
    topology_record,
)

from .artifacts import ArtifactStore


TOPOLOGY_FIELDS = [
    "scope", "topology_id", "method", "graph_seed", "graph_sha256", "n", "f", "edge_count",
    "edges", "degree_sequence", "minimum_degree", "degree_imbalance", "articulation_points",
    "articulation_count", "necessary_edge_bound", "necessary_count_met", "scenario_count",
    "connected_scenarios", "connectivity_rate", "worst_lambda2", "worst_failure", "runtime_s",
    "lambda2_min", "lambda2_p05", "lambda2_q25", "lambda2_median", "lambda2_q75",
    "lambda2_p95", "lambda2_max", "S_c_min", "S_c_p05", "S_c_q25", "S_c_median",
    "S_c_q75", "S_c_p95", "S_c_max", "connectivity_rate_f0", "S_c_min_f0",
    "lambda2_min_f0", "connectivity_rate_f1", "S_c_min_f1", "lambda2_min_f1",
    "connectivity_rate_f2", "S_c_min_f2", "lambda2_min_f2",
]

RESIDUAL_FIELDS = [
    "scope", "topology_id", "method", "graph_seed", "failure_count", "failed_nodes",
    "surviving_nodes", "surviving_edges", "connected", "lambda2", "lambda_max", "S_c",
]


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol") != "experiment_protocol_v2.1":
        raise ValueError("V2.1-G1 requires protocol=experiment_protocol_v2.1")
    return config


def _seed_range(block: dict) -> range:
    first, last = int(block["random_seed_start"]), int(block["random_seed_end"])
    if last < first:
        raise ValueError("random_seed_end must not precede random_seed_start")
    return range(first, last + 1)


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "mean": float(np.mean(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _append_topology(
    topology_writer: csv.DictWriter,
    residual_writer: csv.DictWriter,
    *,
    scope: str,
    method: str,
    seed: int | None,
    n: int,
    f: int,
    edges,
    runtime_s: float,
) -> dict:
    certificate = certify_topology(n, edges, f)
    record, scenarios = topology_record(
        scope=scope, method=method, seed=seed, n=n, edges=edges,
        certificate=certificate, runtime_s=runtime_s,
    )
    topology_writer.writerow(record)
    residual_writer.writerows(scenarios)
    return record


def _fixed_anchor_methods(n: int, f: int, edge_count: int) -> list[tuple[str, tuple, float]]:
    methods: list[tuple[str, tuple, float]] = []
    degree = 2 * edge_count / n
    if degree.is_integer() and 2 <= int(degree) < n:
        started = perf_counter()
        edges = circulant_regular_edges(n, int(degree))
        methods.append(("B2_circulant_regular", edges, perf_counter() - started))
    started = perf_counter()
    nominal, _ = greedy_edge_removal(n, edge_count, f, robust=False)
    methods.append(("B3_nominal_lambda2_greedy", nominal, perf_counter() - started))
    started = perf_counter()
    robust, _ = greedy_edge_removal(n, edge_count, f, robust=True)
    methods.append(("S1_worst_residual_greedy", robust, perf_counter() - started))
    return methods


def run_v21_g1(config_path: Path, output_root: Path) -> Path:
    config = _load_config(config_path)
    anchor = config["anchor"]
    n, f = int(anchor["n"]), int(anchor["f"])
    edge_counts = [int(value) for value in anchor["edge_counts"]]
    seeds = list(_seed_range(anchor))
    store = ArtifactStore(output_root, f"V21_G1_stage1_structural_N{n}_f{f}")
    store.write_json("canonical_config.json", config)

    anchor_random: dict[int, list[dict]] = defaultdict(list)
    fixed_records: list[dict] = []
    scale_records: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    selected_for_stress: dict[str, tuple] = {}
    top_path = store.path / "topology_records.csv"
    residual_path = store.path / "residual_scenarios.csv"
    with top_path.open("w", encoding="utf-8-sig", newline="") as top_handle, residual_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as residual_handle:
        topology_writer = csv.DictWriter(top_handle, fieldnames=TOPOLOGY_FIELDS, extrasaction="ignore")
        residual_writer = csv.DictWriter(residual_handle, fieldnames=RESIDUAL_FIELDS, extrasaction="ignore")
        topology_writer.writeheader()
        residual_writer.writeheader()

        for edge_count in edge_counts:
            for seed in seeds:
                started = perf_counter()
                edges = random_connected_edges(n, edge_count, seed)
                record = _append_topology(
                    topology_writer, residual_writer, scope="anchor", method="B1_random_connected",
                    seed=seed, n=n, f=f, edges=edges, runtime_s=perf_counter() - started,
                )
                anchor_random[edge_count].append(record)
                if edge_count == int(anchor["exact_edge_count"]) and seed == seeds[0]:
                    selected_for_stress["B1_random_seed_first"] = edges
            for method, edges, elapsed in _fixed_anchor_methods(n, f, edge_count):
                record = _append_topology(
                    topology_writer, residual_writer, scope="anchor", method=method,
                    seed=None, n=n, f=f, edges=edges, runtime_s=elapsed,
                )
                fixed_records.append(record)
                if edge_count == int(anchor["exact_edge_count"]):
                    selected_for_stress[method] = edges

        scale = config.get("scale_extension", {})
        if bool(scale.get("enabled", False)):
            for scale_n in [int(value) for value in scale["nodes"]]:
                for degree in [int(value) for value in scale["degrees"]]:
                    edge_count = scale_n * degree // 2
                    if scale_n * degree % 2:
                        continue
                    for scale_f in [int(value) for value in scale["failure_budgets"]]:
                        for seed in _seed_range(scale):
                            started = perf_counter()
                            edges = random_connected_edges(scale_n, edge_count, seed)
                            record = _append_topology(
                                topology_writer, residual_writer, scope="scale", method="B1_random_connected",
                                seed=seed, n=scale_n, f=scale_f, edges=edges,
                                runtime_s=perf_counter() - started,
                            )
                            scale_records[(scale_n, degree, scale_f)].append(record)
                        started = perf_counter()
                        edges = circulant_regular_edges(scale_n, degree)
                        record = _append_topology(
                            topology_writer, residual_writer, scope="scale", method="B2_circulant_regular",
                            seed=None, n=scale_n, f=scale_f, edges=edges,
                            runtime_s=perf_counter() - started,
                        )
                        fixed_records.append(record)

    exact = None
    if bool(anchor.get("run_exact_enumeration", True)):
        exact = enumerate_exact_max_z(n, f, int(anchor["exact_edge_count"]))
        exact_edges = tuple(tuple(edge) for edge in exact["edges"])
        exact["graph_sha256"] = graph_sha256(n, exact_edges)
        expected = int(anchor["exact_expected_topologies"])
        exact["expected_topologies"] = expected
        exact["count_gate_pass"] = int(exact["evaluated_topologies"]) == expected
        selected_for_stress["S1_exact_optimum"] = exact_edges
        store.write_json("exact_enumeration.json", exact)

    stress_rows: list[dict] = []
    if bool(anchor.get("run_f3_stress", True)):
        for method, edges in selected_for_stress.items():
            certificate = certify_topology(n, edges, 3)
            _, scenarios = topology_record(
                scope="anchor_f3_stress", method=method, seed=None, n=n, edges=edges,
                certificate=certificate, runtime_s=0.0,
            )
            stress_rows.extend(row for row in scenarios if int(row["failure_count"]) == 3)
        store.write_csv("f3_stress_scenarios.csv", stress_rows, RESIDUAL_FIELDS)

    random_summary = {}
    for edge_count, records in sorted(anchor_random.items()):
        random_summary[str(edge_count)] = {
            "samples": len(records),
            "unique_graphs": len({str(record["graph_sha256"]) for record in records}),
            "all_scenario_connected_rate": sum(float(record["connectivity_rate"]) == 1.0 for record in records) / len(records),
            "worst_lambda2": _distribution([float(record["worst_lambda2"]) for record in records]),
            "connectivity_rate": _distribution([float(record["connectivity_rate"]) for record in records]),
            "S_c_min": _distribution([float(record["S_c_min"]) for record in records]),
        }
    scale_summary = {}
    for (scale_n, degree, scale_f), records in sorted(scale_records.items()):
        key = f"N{scale_n}_d{degree}_f{scale_f}"
        scale_summary[key] = {
            "samples": len(records),
            "unique_graphs": len({str(record["graph_sha256"]) for record in records}),
            "all_scenario_connected_rate": sum(float(record["connectivity_rate"]) == 1.0 for record in records) / len(records),
            "worst_lambda2": _distribution([float(record["worst_lambda2"]) for record in records]),
            "S_c_min": _distribution([float(record["S_c_min"]) for record in records]),
        }
    summary = {
        "gate": "V2.1-G1",
        "experiment": "E1_stage1_structural_monte_carlo_and_exact_enumeration",
        "anchor_random": random_summary,
        "fixed_method_records": fixed_records,
        "scale_random": scale_summary,
        "exact_enumeration": exact,
        "acceptance": {
            "anchor_seed_count_pass": all(item["samples"] == len(seeds) for item in random_summary.values()),
            "anchor_edge_budget_count_pass": len(random_summary) == len(edge_counts),
            "exact_count_pass": exact is None or bool(exact["count_gate_pass"]),
            "f3_scenario_count_per_method": 0 if not selected_for_stress else len(stress_rows) // len(selected_for_stress),
            "immutable_manifest_written": True,
        },
        "claim_boundary": (
            "Exact optimality is asserted only for the frozen N=8, f=2, E=12 domain. "
            "The scale block is a random/regular structural extension; it does not claim a globally optimal S1 solution."
        ),
    }
    summary["passed"] = all(bool(value) for key, value in summary["acceptance"].items() if key.endswith("_pass"))
    store.write_json("summary.json", summary)
    return store.finalize()
