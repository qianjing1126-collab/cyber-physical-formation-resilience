from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tnsm_sim.experiments.artifacts import ArtifactStore
from tnsm_sim.experiments.g3_network import TraceRun, aggregate_metrics, discover_trace_runs
from tnsm_sim.experiments.g3_network_two_state import (
    fit_two_state_adapter,
    structural_metrics,
    synthesize_two_state_run,
)


def probe_interarrival_metrics(
    runs: Iterable[TraceRun], nominal_interval_s: float
) -> dict[str, float | int]:
    intervals: list[float] = []
    for run in runs:
        by_sender: dict[int, list[dict[str, float | int]]] = {}
        for row in run.transmissions:
            by_sender.setdefault(int(row["tx_id"]), []).append(row)
        for transmissions in by_sender.values():
            times = np.asarray([
                float(row["t_gen_s"])
                for row in sorted(transmissions, key=lambda item: int(item["seq"]))
            ])
            if len(times) > 1:
                intervals.extend(np.diff(times).tolist())
    values = np.asarray(intervals, dtype=float)
    if not values.size:
        raise ValueError("interarrival audit requires at least two transmissions per sender")
    mean = float(np.mean(values))
    return {
        "samples": int(values.size),
        "mean_s": mean,
        "std_s": float(np.std(values)),
        "min_s": float(np.min(values)),
        "max_s": float(np.max(values)),
        "mean_relative_error": abs(mean - nominal_interval_s) / nominal_interval_s,
    }


def per_seed_contention_rate_std(runs: Iterable[TraceRun], threshold_s: float) -> tuple[float, list[dict[str, float | int]]]:
    rows = []
    for run in sorted(runs, key=lambda item: int(item.metadata["seed"])):
        metrics = structural_metrics([run], threshold_s)
        rows.append({
            "seed": int(run.metadata["seed"]),
            "contention_event_rate": float(metrics["contention_event_rate"]),
        })
    values = np.asarray([row["contention_event_rate"] for row in rows], dtype=float)
    return (float(np.std(values)) if values.size else float("nan")), rows


def _relative_error(candidate: float, reference: float) -> float:
    return abs(candidate - reference) / max(abs(reference), 1e-12)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_reference_process(runs: list[TraceRun], config: dict[str, Any]) -> None:
    traffic = config["probe_traffic"]
    expected_tx = int(traffic["vehicle_count"] * round(traffic["measurement_s"] * traffic["rate_hz"]))
    for run in runs:
        if len(run.transmissions) != expected_tx:
            raise ValueError(f"probe transmission count mismatch in {run.path}: {len(run.transmissions)} != {expected_tx}")
        if run.metadata.get("generation_schedule") != traffic["generation_schedule"]:
            raise ValueError(f"generation schedule mismatch in {run.path}")
        jitter = float(run.metadata.get("generation_jitter_fraction", -1.0))
        if abs(jitter - float(traffic["generation_jitter_fraction"])) > 1e-12:
            raise ValueError(f"generation jitter mismatch in {run.path}")


def run_g3b3_calibration(config_path: Path, trace_root: Path, output_root: Path) -> Path:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8-sig"))
    runs = discover_trace_runs(trace_root, config["protocol_version"])
    regime_ids = [item["id"] for item in config["regimes"]]
    expected_cases = {
        (regime, int(seed))
        for regime in regime_ids
        for seed in config["pilot"]["seeds"]
    }
    actual_cases = {(str(run.metadata["regime"]), int(run.metadata["seed"])) for run in runs}
    if actual_cases != expected_cases:
        raise ValueError(
            f"G3-B3 trace matrix mismatch; missing={sorted(expected_cases - actual_cases)}, "
            f"extra={sorted(actual_cases - expected_cases)}"
        )
    _validate_reference_process(runs, config)

    development = {int(seed) for seed in config["pilot"]["fit_seeds"]}
    holdout = {int(seed) for seed in config["pilot"]["holdout_seeds"]}
    sample_period = float(config["metrics"]["aoi_sampling_period_s"])
    nominal_interval = 1.0 / float(config["probe_traffic"]["rate_hz"])
    acceptance = config["acceptance"]
    store = ArtifactStore(output_root, "G3B3_stationary_calibration_v1")
    store.write_json("config_frozen.json", config)

    fitted: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for regime in regime_ids:
        regime_runs = [run for run in runs if run.metadata["regime"] == regime]
        fit_runs = [run for run in regime_runs if int(run.metadata["seed"]) in development]
        holdout_runs = [run for run in regime_runs if int(run.metadata["seed"]) in holdout]
        parameters = fit_two_state_adapter(fit_runs, config)
        fitted[regime] = parameters
        adapter_runs = [synthesize_two_state_run(run, parameters) for run in holdout_runs]
        reference_metrics = aggregate_metrics(holdout_runs, sample_period)
        adapter_metrics = aggregate_metrics(adapter_runs, sample_period)
        reference_structure = structural_metrics(holdout_runs, parameters["contention_threshold_s"])
        adapter_structure = structural_metrics(adapter_runs, parameters["contention_threshold_s"])
        reference_std, per_seed = per_seed_contention_rate_std(
            holdout_runs, parameters["contention_threshold_s"]
        )
        for row in per_seed:
            seed_rows.append({"regime": regime, **row})
        interarrival = probe_interarrival_metrics(holdout_runs, nominal_interval)
        errors = {
            "pdr_absolute_error": abs(float(adapter_metrics["pdr"]) - float(reference_metrics["pdr"])),
            "delay_median_relative_error": _relative_error(float(adapter_metrics["delay_p50_s"]), float(reference_metrics["delay_p50_s"])),
            "delay_p95_relative_error": _relative_error(float(adapter_metrics["delay_p95_s"]), float(reference_metrics["delay_p95_s"])),
            "aoi_mean_relative_error": _relative_error(float(adapter_metrics["aoi_mean_s"]), float(reference_metrics["aoi_mean_s"])),
            "aoi_p95_relative_error": _relative_error(float(adapter_metrics["aoi_p95_s"]), float(reference_metrics["aoi_p95_s"])),
            "contention_event_rate_absolute_error": abs(float(adapter_structure["contention_event_rate"]) - float(reference_structure["contention_event_rate"])),
            "state_persistence_absolute_error": abs(float(adapter_structure["state_persistence"]) - float(reference_structure["state_persistence"])),
        }
        gates = {
            name: value <= float(acceptance[f"{name}_max"])
            for name, value in errors.items()
        }
        gates["holdout_per_seed_contention_rate_std"] = reference_std <= float(
            acceptance["holdout_per_seed_contention_rate_std_max"]
        )
        gates["probe_interarrival_mean_relative_error"] = float(interarrival["mean_relative_error"]) <= float(
            acceptance["probe_interarrival_mean_relative_error_max"]
        )
        comparison_rows.append({
            "regime": regime,
            **{f"reference_{key}": value for key, value in reference_metrics.items()},
            **{f"adapter_{key}": value for key, value in adapter_metrics.items()},
            **{f"reference_{key}": value for key, value in reference_structure.items()},
            **{f"adapter_{key}": value for key, value in adapter_structure.items()},
            "reference_holdout_per_seed_contention_rate_std": reference_std,
            **{f"reference_probe_interarrival_{key}": value for key, value in interarrival.items()},
            **errors,
            **{f"gate_{name}_passed": passed for name, passed in gates.items()},
            "passed": all(gates.values()),
        })

    _write_csv(store.path / "holdout_comparison.csv", comparison_rows)
    _write_csv(store.path / "holdout_seed_contention_rates.csv", seed_rows)
    store.write_json("fitted_two_state_parameters.json", fitted)
    summary = {
        "model_protocol_version": config["model_protocol_version"],
        "trace_protocol_version": config["protocol_version"],
        "trace_root": str(trace_root.resolve()),
        "config_sha256": sha256(config_bytes).hexdigest(),
        "development_seeds": sorted(development),
        "untouched_holdout_seeds": sorted(holdout),
        "regimes": comparison_rows,
        "all_holdout_gates_passed": all(bool(row["passed"]) for row in comparison_rows),
        "claim_boundary": config["claim_boundary"],
    }
    store.write_json("summary.json", summary)

    from tnsm_sim.plotting.g3_network_b3 import plot_g3b3_calibration

    plot_g3b3_calibration(store.path)
    return store.finalize()
