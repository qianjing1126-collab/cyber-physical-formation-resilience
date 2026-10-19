from __future__ import annotations

from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from tnsm_sim.config import load_config
from tnsm_sim.io import verify_manifest
from tnsm_sim.runner import build_run_id, run_simulation
from tnsm_sim.topology import (
    algebraic_connectivity,
    graph_sha256,
    greedy_edge_removal,
    largest_component_ratio,
    random_connected_edges,
    residual_edges,
)

from .artifacts import ArtifactStore


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_VERSION = "experiment_protocol_v2.1"
METHODS = ("B1", "B3", "S1")
OFFICIAL_VARIANT = "v5"


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_frozen(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"frozen V2.1-G2 configuration changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def balanced_failure_pairs(n: int = 8, cases: int = 100, seed: int = 20260810) -> tuple[tuple[int, int], ...]:
    pairs = list(combinations(range(1, n + 1), 2))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    repeated = (pairs * ((cases + len(pairs) - 1) // len(pairs)))[:cases]
    return tuple(repeated)


def _components(nodes: set[int], edges: tuple[tuple[int, int], ...]) -> list[set[int]]:
    adjacency = {node: set() for node in nodes}
    for i, j in edges:
        if i in adjacency and j in adjacency:
            adjacency[i].add(j)
            adjacency[j].add(i)
    unseen = set(nodes)
    result = []
    while unseen:
        component = {unseen.pop()}
        queue = deque(component)
        while queue:
            node = queue.popleft()
            new = adjacency[node] & unseen
            unseen -= new
            component |= new
            queue.extend(new)
        result.append(component)
    return result


def network_service_metrics(n: int, edges: tuple[tuple[int, int], ...], failed: tuple[int, ...]) -> dict[str, float | int]:
    active = set(range(1, n + 1)) - set(failed)
    remaining = residual_edges(edges, active)
    components = _components(active, remaining)
    ordered_total = len(active) * max(0, len(active) - 1)
    ordered_reachable = sum(len(component) * (len(component) - 1) for component in components)
    hop_lengths: list[int] = []
    adjacency = {node: set() for node in active}
    for i, j in remaining:
        adjacency[i].add(j)
        adjacency[j].add(i)
    for source in sorted(active):
        distances = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        hop_lengths.extend(distance for target, distance in distances.items() if target != source)
    return {
        "residual_edge_count": len(remaining),
        "residual_lambda2": algebraic_connectivity(sorted(active), remaining),
        "residual_S_c": largest_component_ratio(sorted(active), remaining),
        "component_count": len(components),
        "e2e_reachability": ordered_reachable / ordered_total if ordered_total else 1.0,
        "mean_shortest_hops": float(np.mean(hop_lengths)) if hop_lengths else float("nan"),
        "p95_shortest_hops": float(np.quantile(hop_lengths, 0.95)) if hop_lengths else float("nan"),
    }


def _configuration(template: dict[str, Any], case_id: int, method: str, base_seed: int,
                   graph_seed: int | None, failed: tuple[int, int], edges: tuple[tuple[int, int], ...]) -> dict[str, Any]:
    cfg = deepcopy(template)
    cfg["meta"].update({
        "protocol_version": PROTOCOL_VERSION,
        "architecture_version": "sim_arch_v1.1",
        "module": f"V21G2{OFFICIAL_VARIANT.upper()}",
        "scenario": f"e2_case{case_id:04d}_fail{failed[0]:02d}-{failed[1]:02d}",
        "method": method,
        "base_seed": base_seed,
        "case_id": case_id,
        "graph_seed": graph_seed,
    })
    cfg["failure"].update({"type": "P-2X", "targets": list(failed), "t_d": 15.0, "t_c": 15.5})
    cfg["network"].pop("two_state_profile_file", None)
    cfg["network"].pop("two_state_regime", None)
    cfg["network"]["logical_edges"] = [list(edge) for edge in edges]
    cfg["control"]["logical_topology"] = method
    cfg["control"]["edge_weight"] = 1.0
    cfg["recovery"]["enabled"] = True
    cfg["recovery"]["method"] = "S3"
    cfg["recovery"]["require_connected_coordination"] = True
    return cfg


def build_v21_g2_matrix(g1_run_dir: Path, project_root: Path = PROJECT_ROOT) -> Path:
    project_root = project_root.resolve()
    g1_run_dir = g1_run_dir.resolve()
    exact = json.loads((g1_run_dir / "exact_enumeration.json").read_text(encoding="utf-8"))
    s1_edges = tuple(tuple(int(value) for value in edge) for edge in exact["edges"])
    b3_edges, _ = greedy_edge_removal(8, 12, 2, robust=False)
    base = load_config(project_root / "configs" / "scenarios" / "v2_1_g2_base.yaml")
    template = json.loads(base.canonical_json)
    pairs = balanced_failure_pairs()
    speed_reference = float(template["vehicle"]["v_ref"])
    expected_speed = float(OFFICIAL_VARIANT.removeprefix("v"))
    if abs(speed_reference - expected_speed) > 1.0e-12:
        raise RuntimeError(
            f"G2 variant {OFFICIAL_VARIANT} requires {expected_speed:g} m/s, "
            f"but the base configuration contains {speed_reference:g} m/s"
        )
    config_root = project_root / "runs" / "v2_1" / "g2" / f"configs_{OFFICIAL_VARIANT}"
    rows: list[dict[str, Any]] = []
    for offset, failed in enumerate(pairs):
        case_id = offset + 1
        base_seed = 101 + offset
        graph_seed = 1001 + offset
        method_edges = {
            "B1": random_connected_edges(8, 12, graph_seed),
            "B3": b3_edges,
            "S1": s1_edges,
        }
        for method in METHODS:
            edges = method_edges[method]
            cfg = _configuration(
                template, case_id, method, base_seed,
                graph_seed if method == "B1" else None, failed, edges,
            )
            relative = Path(f"case{case_id:04d}") / f"{method}.json"
            config_path = config_root / relative
            text = json.dumps(cfg, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
            _write_frozen(config_path, text)
            frozen = load_config(config_path)
            service = network_service_metrics(8, edges, failed)
            rows.append({
                "case_id": case_id,
                "method": method,
                "base_seed": base_seed,
                "graph_seed": graph_seed if method == "B1" else None,
                "failure_targets": list(failed),
                "topology_sha256": graph_sha256(8, edges),
                "logical_edges": [list(edge) for edge in edges],
                **service,
                "config_path": str(config_path),
                "config_hash": frozen.config_hash,
                "run_id": build_run_id(frozen),
            })
    counts = Counter(tuple(row["failure_targets"]) for row in rows if row["method"] == "S1")
    if len(rows) != 300 or len({row["run_id"] for row in rows}) != 300:
        raise RuntimeError("V2.1-G2 matrix must contain 300 unique runs")
    if set(counts.values()) - {3, 4} or sum(value == 4 for value in counts.values()) != 16:
        raise RuntimeError(f"failure pairs are not balanced: {counts}")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "gate": "V2.1-G2",
        "g1_source": str(g1_run_dir),
        "case_count": 100,
        "run_count": 300,
        "methods": list(METHODS),
        "speed_reference_mps": speed_reference,
        "failure_pair_counts": {f"{i}-{j}": count for (i, j), count in sorted(counts.items())},
        "rows_sha256": sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "rows": rows,
    }
    index_path = project_root / "runs" / "v2_1" / "g2" / f"V21_G2_matrix_{OFFICIAL_VARIANT}.json"
    if index_path.exists():
        prior = json.loads(index_path.read_text(encoding="utf-8"))
        if prior != payload:
            raise RuntimeError("frozen V2.1-G2 matrix changed")
    else:
        _atomic_json(index_path, payload)
    return index_path


def _run_one(row: dict[str, Any], project_root: str) -> dict[str, Any]:
    root = Path(project_root)
    expected = root / "runs" / "raw" / row["run_id"]
    if expected.exists():
        audit = verify_manifest(expected)
        if not audit["valid"]:
            raise RuntimeError(f"existing V2.1-G2 run failed manifest audit: {expected}")
        metrics = json.loads((expected / "metrics.json").read_text(encoding="utf-8"))
        if metrics["config_hash"] != row["config_hash"]:
            raise RuntimeError(f"existing V2.1-G2 run has a different config hash: {expected}")
        return {"run_id": row["run_id"], "state": "SKIPPED_VALID", "scientific_status": metrics["scientific_status"]}
    config = load_config(row["config_path"])
    run_dir, metrics = run_simulation(config, root)
    if run_dir != expected:
        raise RuntimeError(f"unexpected retry directory for V2.1-G2 run: {run_dir}")
    audit = verify_manifest(run_dir)
    if not audit["valid"]:
        raise RuntimeError(f"new V2.1-G2 run failed manifest audit: {run_dir}")
    return {"run_id": row["run_id"], "state": "COMPLETE", "scientific_status": metrics["scientific_status"]}


def run_v21_g2_batch(index_path: Path, workers: int = 4, project_root: Path = PROJECT_ROOT) -> Path:
    project_root = project_root.resolve()
    matrix = json.loads(index_path.resolve().read_text(encoding="utf-8"))
    variant = str(matrix.get("speed_reference_mps", "")).replace(".0", "")
    progress_path = project_root / "runs" / "v2_1" / "g2" / f"V21_G2_progress_v{variant}.json"
    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, row, str(project_root)): row for row in matrix["rows"]}
        for future in as_completed(futures):
            row = futures[future]
            try:
                results[row["run_id"]] = future.result()
            except Exception as exc:
                failures[row["run_id"]] = f"{type(exc).__name__}: {exc}"
            _atomic_json(progress_path, {
                "protocol_version": PROTOCOL_VERSION,
                "expected": len(matrix["rows"]),
                "finished": len(results) + len(failures),
                "valid": len(results),
                "failed_processes": len(failures),
                "failures": failures,
            })
    if failures:
        raise RuntimeError(f"{len(failures)} V2.1-G2 runs failed; inspect {progress_path}")
    _atomic_json(progress_path, {
        "protocol_version": PROTOCOL_VERSION,
        "expected": len(matrix["rows"]),
        "finished": len(results),
        "valid": len(results),
        "failed_processes": 0,
        "complete": True,
        "scientific_status_counts": dict(Counter(item["scientific_status"] for item in results.values())),
    })
    return progress_path


def _bootstrap_delta(rows: list[dict[str, Any]], metric: str, baseline: str, samples: int = 5000) -> dict[str, float]:
    by_case: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_case[int(row["case_id"])][str(row["method"])] = float(row[metric])
    deltas = np.asarray([values["S1"] - values[baseline] for values in by_case.values()], dtype=float)
    rng = np.random.default_rng(20260810)
    indices = rng.integers(0, len(deltas), size=(samples, len(deltas)))
    means = deltas[indices].mean(axis=1)
    return {
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def analyze_v21_g2(index_path: Path, project_root: Path = PROJECT_ROOT) -> Path:
    project_root = project_root.resolve()
    matrix = json.loads(index_path.resolve().read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for row in matrix["rows"]:
        run_dir = project_root / "runs" / "raw" / row["run_id"]
        audit = verify_manifest(run_dir)
        if not audit["valid"]:
            raise RuntimeError(f"cannot analyze invalid run: {run_dir}")
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        reachability = float(row["e2e_reachability"])
        link_aoi = float(metrics["AoI_p95"])
        e2e_aoi = (
            15.0 if reachability < 0.95
            else link_aoi * float(row["p95_shortest_hops"])
        )
        tr_penalized = 15.0 if metrics.get("T_r") is None else float(metrics["T_r"])
        plan_issued = int(metrics.get("stage3_duration") is not None)
        records.append({
            "case_id": row["case_id"], "method": row["method"], "base_seed": row["base_seed"],
            "graph_seed": "" if row["graph_seed"] is None else row["graph_seed"],
            "failure_targets": json.dumps(row["failure_targets"], separators=(",", ":")),
            "topology_sha256": row["topology_sha256"],
            "residual_edge_count": row["residual_edge_count"],
            "residual_lambda2": row["residual_lambda2"], "residual_S_c": row["residual_S_c"],
            "component_count": row["component_count"], "e2e_reachability": reachability,
            "mean_shortest_hops": row["mean_shortest_hops"], "p95_shortest_hops": row["p95_shortest_hops"],
            "PDR": metrics["PDR"], "effective_e2e_PDR": float(metrics["PDR"]) * reachability,
            "AoI_p95_link": link_aoi, "AoI_p95_e2e_censored": e2e_aoi,
            "position_rmse": metrics["position_tracking_rmse"],
            "velocity_rmse": metrics["velocity_tracking_rmse"],
            "Q_min": metrics["Q_min"], "R_Te": metrics["R_Te"], "R_mission": metrics["R_mission"],
            "T_r": "" if metrics.get("T_r") is None else metrics["T_r"], "T_r_penalized": tr_penalized,
            "coordination_plan_issued": plan_issued,
            "recovery_success": int(bool(metrics["recovery_success"])),
            "safety_ok": int(bool(metrics["safety_ok"])),
            "scientific_status": metrics["scientific_status"], "run_id": row["run_id"],
        })
    by_method = {}
    numeric = [
        "residual_lambda2", "residual_S_c", "e2e_reachability", "PDR", "effective_e2e_PDR",
        "AoI_p95_e2e_censored", "position_rmse", "velocity_rmse", "Q_min", "R_Te",
        "R_mission", "T_r_penalized", "coordination_plan_issued", "recovery_success", "safety_ok",
    ]
    for method in METHODS:
        subset = [row for row in records if row["method"] == method]
        by_method[method] = {
            metric: {
                "mean": float(np.mean([float(row[metric]) for row in subset])),
                "median": float(np.median([float(row[metric]) for row in subset])),
                "p05": float(np.quantile([float(row[metric]) for row in subset], 0.05)),
                "p95": float(np.quantile([float(row[metric]) for row in subset], 0.95)),
            }
            for metric in numeric
        }
        by_method[method]["status_counts"] = dict(Counter(str(row["scientific_status"]) for row in subset))
    effects = {
        f"S1_minus_{baseline}": {
            metric: _bootstrap_delta(records, metric, baseline)
            for metric in ("e2e_reachability", "effective_e2e_PDR", "AoI_p95_e2e_censored",
                           "coordination_plan_issued", "position_rmse", "velocity_rmse", "R_Te", "R_mission",
                           "T_r_penalized", "recovery_success")
        }
        for baseline in ("B1", "B3")
    }
    correlations = {}
    for name, first, second in (
        ("structure_to_network", "residual_lambda2", "effective_e2e_PDR"),
        ("network_to_coordination", "effective_e2e_PDR", "coordination_plan_issued"),
        ("coordination_to_task", "coordination_plan_issued", "R_Te"),
        ("tracking_transient_to_task", "position_rmse", "R_Te"),
    ):
        coefficient, pvalue = spearmanr(
            [float(row[first]) for row in records], [float(row[second]) for row in records]
        )
        correlations[name] = {"first": first, "second": second, "spearman_rho": float(coefficient), "p_value": float(pvalue)}
    case_methods = defaultdict(set)
    for row in records:
        case_methods[int(row["case_id"])].add(str(row["method"]))
    acceptance = {
        "run_count_pass": len(records) == 300,
        "paired_case_count_pass": len(case_methods) == 100 and all(methods == set(METHODS) for methods in case_methods.values()),
        "safety_pass": all(int(row["safety_ok"]) == 1 for row in records),
        "manifest_pass": True,
    }
    summary = {
        "gate": "V2.1-G2",
        "passed": all(acceptance.values()),
        "acceptance": acceptance,
        "method_summary": by_method,
        "paired_effects": effects,
        "cross_layer_correlations": correlations,
        "metric_boundary": (
            "effective_e2e_PDR combines measured one-hop PDR with residual all-pairs reachability. "
            "AoI_p95_e2e_censored is an offline routing-service metric; unreachable pairs are censored at T_e=15 s. "
            "Tracking RMSE is reported but is not treated as a monotone benefit metric: successful recovery creates "
            "a nonzero maneuver transient, whereas communication-infeasible cases issue no recovery trajectory."
        ),
    }
    variant = str(matrix.get("speed_reference_mps", "")).replace(".0", "")
    store = ArtifactStore(project_root / "runs" / "v2_1" / "g2", f"V21_G2_analysis_v{variant}")
    store.write_json("summary.json", summary)
    fieldnames = list(records[0])
    store.write_csv("paired_records.csv", records, fieldnames)
    store.write_json("source_matrix.json", {"path": str(index_path.resolve()), "rows_sha256": matrix["rows_sha256"]})
    return store.finalize()
