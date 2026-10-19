from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tnsm_sim.experiments.artifacts import ArtifactStore


@dataclass(frozen=True)
class TraceRun:
    path: Path
    metadata: dict[str, Any]
    transmissions: tuple[dict[str, float | int], ...]
    receptions: tuple[dict[str, float | int], ...]


def _read_csv(path: Path, integer_fields: set[str]) -> tuple[dict[str, float | int], ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = []
        for source in csv.DictReader(handle):
            rows.append({
                key: int(value) if key in integer_fields else float(value)
                for key, value in source.items()
            })
    return tuple(rows)


def load_trace_run(path: Path) -> TraceRun:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8-sig"))
    tx = _read_csv(path / "tx.csv", {"uid", "tx_id", "seq", "size_bytes"})
    rx = _read_csv(path / "rx.csv", {"uid", "tx_id", "rx_id", "seq", "size_bytes"})
    unique_tx = {(row["tx_id"], row["seq"]) for row in tx}
    if len(unique_tx) != len(tx):
        raise ValueError(f"duplicate transmitter/sequence keys in {path}")
    unique_rx = {(row["tx_id"], row["rx_id"], row["seq"]) for row in rx}
    if len(unique_rx) != len(rx):
        raise ValueError(f"duplicate receptions in {path}")
    if any((row["tx_id"], row["seq"]) not in unique_tx for row in rx):
        raise ValueError(f"orphan reception in {path}")
    return TraceRun(path, metadata, tx, rx)


def discover_trace_runs(trace_root: Path, protocol_version: str) -> list[TraceRun]:
    runs = []
    for metadata_path in sorted(trace_root.glob("*/metadata.json")):
        run = load_trace_run(metadata_path.parent)
        if run.metadata.get("protocol_version") != protocol_version:
            raise ValueError(f"protocol mismatch in {run.path}")
        runs.append(run)
    if not runs:
        raise ValueError(f"no ns-3 traces found under {trace_root}")
    return runs


def _aoi_samples(run: TraceRun, sample_period: float) -> np.ndarray:
    n = int(run.metadata["probe_nodes"])
    start = float(run.metadata["warmup_s"])
    stop = start + float(run.metadata["measurement_s"])
    grids = np.arange(start, stop + 0.5 * sample_period, sample_period)
    grouped: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for row in run.receptions:
        key = (int(row["tx_id"]), int(row["rx_id"]))
        grouped.setdefault(key, []).append((float(row["t_rx_s"]), float(row["t_gen_s"])))
    samples: list[float] = []
    for tx_id in range(n):
        for rx_id in range(n):
            if tx_id == rx_id:
                continue
            events = sorted(grouped.get((tx_id, rx_id), ()))
            event_index = 0
            last_generation: float | None = None
            for t in grids:
                while event_index < len(events) and events[event_index][0] <= t + 1e-12:
                    last_generation = events[event_index][1]
                    event_index += 1
                if last_generation is not None:
                    samples.append(max(0.0, float(t) - last_generation))
    return np.asarray(samples, dtype=float)


def aggregate_metrics(runs: Iterable[TraceRun], sample_period: float) -> dict[str, float | int]:
    run_list = list(runs)
    if not run_list:
        raise ValueError("at least one trace run is required")
    expected = 0
    received = 0
    delays: list[np.ndarray] = []
    aois: list[np.ndarray] = []
    for run in run_list:
        n = int(run.metadata["probe_nodes"])
        expected += len(run.transmissions) * (n - 1)
        received += len(run.receptions)
        delays.append(np.asarray([float(row["delay_s"]) for row in run.receptions], dtype=float))
        aois.append(_aoi_samples(run, sample_period))
    delay = np.concatenate([values for values in delays if values.size]) if received else np.asarray([])
    nonempty_aoi = [values for values in aois if values.size]
    aoi = np.concatenate(nonempty_aoi) if nonempty_aoi else np.asarray([])
    return {
        "runs": len(run_list),
        "expected_receptions": expected,
        "received_receptions": received,
        "pdr": received / expected if expected else 0.0,
        "delay_mean_s": float(np.mean(delay)) if delay.size else float("nan"),
        "delay_std_s": float(np.std(delay)) if delay.size else float("nan"),
        "delay_p50_s": float(np.quantile(delay, 0.50)) if delay.size else float("nan"),
        "delay_p95_s": float(np.quantile(delay, 0.95)) if delay.size else float("nan"),
        "delay_p99_s": float(np.quantile(delay, 0.99)) if delay.size else float("nan"),
        "aoi_mean_s": float(np.mean(aoi)) if aoi.size else float("nan"),
        "aoi_p95_s": float(np.quantile(aoi, 0.95)) if aoi.size else float("nan"),
        "aoi_samples": int(aoi.size),
    }


def fit_packet_event_adapter(runs: Iterable[TraceRun]) -> dict[str, Any]:
    run_list = list(runs)
    metrics = aggregate_metrics(run_list, sample_period=0.01)
    delays = np.asarray([
        float(row["delay_s"])
        for run in run_list
        for row in run.receptions
    ])
    if not delays.size:
        raise ValueError("cannot fit a delay model without received packets")
    center = float(np.mean(delays))
    probabilities = np.asarray([0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
    quantiles = np.quantile(delays, probabilities)
    return {
        "loss_probability": 1.0 - float(metrics["pdr"]),
        "access_delay_s": 0.5 * center,
        "propagation_delay_s": 0.5 * center,
        "jitter_s": 0.0,
        "fit_delay_mean_s": center,
        "fit_delay_std_s": float(np.std(delays)),
        "delay_model": "piecewise_empirical_inverse_cdf",
        "delay_probabilities": probabilities.tolist(),
        "delay_quantiles_s": quantiles.tolist(),
    }


def synthesize_adapter_run(reference: TraceRun, parameters: dict[str, Any], seed_offset: int = 900000) -> TraceRun:
    seed = int(reference.metadata["seed"])
    regime_code = sum(ord(character) for character in str(reference.metadata["regime"]))
    rng = np.random.default_rng(seed_offset + 1000 * regime_code + seed)
    n = int(reference.metadata["probe_nodes"])
    receptions: list[dict[str, float | int]] = []
    for tx in reference.transmissions:
        for rx_id in range(n):
            if rx_id == int(tx["tx_id"]):
                continue
            if rng.random() < parameters["loss_probability"]:
                continue
            if parameters.get("delay_model") == "piecewise_empirical_inverse_cdf":
                delay = float(np.interp(
                    rng.random(),
                    np.asarray(parameters["delay_probabilities"], dtype=float),
                    np.asarray(parameters["delay_quantiles_s"], dtype=float),
                ))
            else:
                access = max(0.0, parameters["access_delay_s"] + rng.uniform(-parameters["jitter_s"], parameters["jitter_s"]))
                propagation = max(0.0, parameters["propagation_delay_s"] + rng.uniform(-parameters["jitter_s"], parameters["jitter_s"]))
                delay = access + propagation
            receptions.append({
                "uid": int(tx["uid"]),
                "tx_id": int(tx["tx_id"]),
                "rx_id": rx_id,
                "seq": int(tx["seq"]),
                "t_gen_s": float(tx["t_gen_s"]),
                "t_rx_s": float(tx["t_gen_s"]) + delay,
                "delay_s": delay,
                "size_bytes": int(tx["size_bytes"]),
            })
    return TraceRun(reference.path, dict(reference.metadata), reference.transmissions, tuple(receptions))


def _relative_error(candidate: float, reference: float) -> float:
    return abs(candidate - reference) / max(abs(reference), 1e-12)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_g3_network_calibration(config_path: Path, trace_root: Path, output_root: Path) -> Path:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8-sig"))
    runs = discover_trace_runs(trace_root, config["protocol_version"])
    expected_cases = {
        (regime["id"], int(seed))
        for regime in config["regimes"]
        for seed in config["pilot"]["seeds"]
    }
    actual_cases = {(str(run.metadata["regime"]), int(run.metadata["seed"])) for run in runs}
    if actual_cases != expected_cases:
        missing, extra = sorted(expected_cases - actual_cases), sorted(actual_cases - expected_cases)
        raise ValueError(f"trace matrix mismatch; missing={missing}, extra={extra}")

    fit_seeds = {int(seed) for seed in config["pilot"]["fit_seeds"]}
    holdout_seeds = {int(seed) for seed in config["pilot"]["holdout_seeds"]}
    sample_period = float(config["metrics"]["aoi_sampling_period_s"])
    acceptance = config["acceptance"]
    store = ArtifactStore(output_root, "G3B_network_calibration_v1")
    store.write_json("config_frozen.json", config)

    fitted: dict[str, dict[str, Any]] = {}
    comparison_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for run in runs:
        metrics = aggregate_metrics([run], sample_period)
        run_rows.append({"regime": run.metadata["regime"], "seed": run.metadata["seed"], **metrics})

    for regime in [item["id"] for item in config["regimes"]]:
        regime_runs = [run for run in runs if run.metadata["regime"] == regime]
        fit_runs = [run for run in regime_runs if int(run.metadata["seed"]) in fit_seeds]
        holdout_runs = [run for run in regime_runs if int(run.metadata["seed"]) in holdout_seeds]
        parameters = fit_packet_event_adapter(fit_runs)
        fitted[regime] = parameters
        reference_metrics = aggregate_metrics(holdout_runs, sample_period)
        adapter_metrics = aggregate_metrics(
            [synthesize_adapter_run(run, parameters) for run in holdout_runs], sample_period
        )
        errors = {
            "pdr_absolute_error": abs(float(adapter_metrics["pdr"]) - float(reference_metrics["pdr"])),
            "delay_median_relative_error": _relative_error(float(adapter_metrics["delay_p50_s"]), float(reference_metrics["delay_p50_s"])),
            "delay_p95_relative_error": _relative_error(float(adapter_metrics["delay_p95_s"]), float(reference_metrics["delay_p95_s"])),
            "aoi_mean_relative_error": _relative_error(float(adapter_metrics["aoi_mean_s"]), float(reference_metrics["aoi_mean_s"])),
            "aoi_p95_relative_error": _relative_error(float(adapter_metrics["aoi_p95_s"]), float(reference_metrics["aoi_p95_s"])),
        }
        gates = {
            name: value <= float(acceptance[f"{name}_max"])
            for name, value in errors.items()
        }
        comparison_rows.append({
            "regime": regime,
            **{f"reference_{key}": value for key, value in reference_metrics.items()},
            **{f"adapter_{key}": value for key, value in adapter_metrics.items()},
            **errors,
            "passed": all(gates.values()),
        })

    _write_csv(store.path / "run_metrics.csv", run_rows)
    _write_csv(store.path / "holdout_comparison.csv", comparison_rows)
    store.write_json("fitted_parameters.json", fitted)
    summary = {
        "protocol_version": config["protocol_version"],
        "trace_root": str(trace_root.resolve()),
        "trace_config_sha256": sha256(config_bytes).hexdigest(),
        "fit_seeds": sorted(fit_seeds),
        "holdout_seeds": sorted(holdout_seeds),
        "regimes": comparison_rows,
        "all_holdout_gates_passed": all(bool(row["passed"]) for row in comparison_rows),
        "claim_boundary": config["claim_boundary"],
    }
    store.write_json("summary.json", summary)

    from tnsm_sim.plotting.g3_network import plot_g3_network_calibration

    plot_g3_network_calibration(store.path)
    return store.finalize()
