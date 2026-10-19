from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from tnsm_sim.topology import (
    certify_topology,
    circulant_regular_edges,
    greedy_edge_removal,
    random_connected_edges,
)
from tnsm_sim.plotting.g2 import plot_p01, plot_p02
from tnsm_sim.weights.delay_margin import modal_delay_margin
from tnsm_sim.weights.design import pilot_weight_designs
from tnsm_sim.weights.root_verification import empirical_delay_margin, pade_delay_margin

from .artifacts import ArtifactStore


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _record(method: str, seed: int | None, certificate, elapsed: float, z_req: float) -> dict:
    result = certificate.to_dict(include_scenarios=False)
    result.update({
        "method": method,
        "seed": seed,
        "runtime_s": elapsed,
        "z_req": z_req,
        "constraint_pass": certificate.connectivity_rate == 1.0 and certificate.worst_lambda2 >= z_req,
    })
    return result


def run_p01(config_path: Path, output_root: Path) -> Path:
    config = _load(config_path)
    n, f = int(config["n"]), int(config["f"])
    edge_count, z_req = int(config["edge_count"]), float(config["z_req"])
    records: list[dict] = []
    scenario_records: list[dict] = []

    methods = []
    start = perf_counter()
    regular = circulant_regular_edges(n, int(round(2 * edge_count / n)))
    methods.append(("B2_circulant_regular", None, regular, perf_counter() - start))
    start = perf_counter()
    nominal, _ = greedy_edge_removal(n, edge_count, f, robust=False)
    methods.append(("B3_nominal_lambda2_greedy", None, nominal, perf_counter() - start))
    start = perf_counter()
    robust, _ = greedy_edge_removal(n, edge_count, f, robust=True)
    methods.append(("Stage1_robust_greedy_pilot", None, robust, perf_counter() - start))
    for seed in config["random_seeds"]:
        start = perf_counter()
        random_edges = random_connected_edges(n, edge_count, int(seed))
        methods.append(("B1_random_connected", int(seed), random_edges, perf_counter() - start))

    topology_edges: dict[str, list[list[int]]] = {}
    for method, seed, edges, elapsed in methods:
        certificate = certify_topology(n, edges, f)
        records.append(_record(method, seed, certificate, elapsed, z_req))
        topology_edges[f"{method}_seed{seed}" if seed is not None else method] = [list(edge) for edge in edges]
        for scenario in certificate.scenarios:
            item = asdict(scenario)
            item.update({"method": method, "seed": seed})
            scenario_records.append(item)

    fixed = [record for record in records if record["seed"] is None]
    random_records = [record for record in records if record["method"] == "B1_random_connected"]
    summary = {
        "experiment": "G2_P0_1_stage1_structural_pilot",
        "scope": "N=8 pilot; Stage1 method is a recertified heuristic, not an MISDP optimum",
        "config": config,
        "fixed_methods": fixed,
        "random_connected": {
            "samples": len(random_records),
            "all_scenario_pass_rate": sum(item["constraint_pass"] for item in random_records) / len(random_records),
            "mean_worst_lambda2": sum(item["worst_lambda2"] for item in random_records) / len(random_records),
        },
    }
    store = ArtifactStore(output_root, f"G2_p01_stage1_N{n}_f{f}")
    store.write_json("canonical_config.json", config)
    store.write_json("summary.json", summary)
    store.write_json("topologies.json", topology_edges)
    store.write_json("method_records.json", records)
    store.write_json("residual_scenarios.json", scenario_records)
    plot_p01(store.path)
    return store.finalize()


def run_p02(config_path: Path, output_root: Path) -> Path:
    config = _load(config_path)
    n, f = int(config["n"]), int(config["f"])
    edges, topology_certificate = greedy_edge_removal(n, int(config["edge_count"]), f, robust=True)
    k_p, k_v = float(config["k_p"]), float(config["k_v"])
    lambda_cap = float(config["lambda_cap"])
    designs = pilot_weight_designs(
        n, f, edges, float(config["total_weight_budget"]), lambda_cap,
        int(config["search_samples"]), int(config["seed"]),
    )
    records = []
    weight_data = {}
    for design in designs:
        theory = modal_delay_margin(design.worst_lambda_max, k_p, k_v)
        pade = pade_delay_margin(design.worst_lambda_max, k_p, k_v, int(config["pade_order"]))
        empirical = empirical_delay_margin(
            design.worst_lambda_max, k_p, k_v,
            float(config["time_step"]), float(config["duration"]),
        )
        records.append({
            "method": design.name,
            "worst_lambda2": design.worst_lambda2,
            "worst_lambda_max": design.worst_lambda_max,
            "lambda_cap": lambda_cap,
            "spectral_upper_bound_pass": design.worst_lambda_max <= lambda_cap + 1e-8,
            "weight_scale": design.scale,
            "tau_theory_s": theory,
            "tau_pade_s": pade,
            "tau_empirical_s": empirical,
            "pade_relative_error": abs(pade - theory) / theory,
            "empirical_relative_error": abs(empirical - theory) / theory,
            "five_percent_gate": max(abs(pade - theory), abs(empirical - theory)) / theory <= 0.05,
        })
        weight_data[design.name] = {f"{i}-{j}": value for (i, j), value in design.weights.items()}
    summary = {
        "experiment": "G2_P0_2_stage2_delay_pilot",
        "scope": "N=8 pilot; random-search implementation checks the Stage2 constraint logic without claiming SDP optimality",
        "config": config,
        "topology_certificate": topology_certificate.to_dict(include_scenarios=False),
        "records": records,
        "all_delay_checks_within_five_percent": all(record["five_percent_gate"] for record in records),
    }
    store = ArtifactStore(output_root, f"G2_p02_stage2_N{n}_f{f}")
    store.write_json("canonical_config.json", config)
    store.write_json("summary.json", summary)
    store.write_json("topology.json", {"edges": [list(edge) for edge in edges]})
    store.write_json("weights.json", weight_data)
    store.write_json("delay_records.json", records)
    plot_p02(store.path)
    return store.finalize()
