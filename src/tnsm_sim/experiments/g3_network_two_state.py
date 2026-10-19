from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tnsm_sim.experiments.artifacts import ArtifactStore
from tnsm_sim.experiments.g3_network import TraceRun, aggregate_metrics, discover_trace_runs


@dataclass(frozen=True)
class BroadcastEvent:
    sender: int
    sequence: int
    generation_time: float
    transmission: dict[str, float | int]
    receptions: tuple[dict[str, float | int], ...]
    intended_receivers: int


def broadcast_events(run: TraceRun) -> list[BroadcastEvent]:
    by_key: dict[tuple[int, int], list[dict[str, float | int]]] = {}
    for row in run.receptions:
        by_key.setdefault((int(row["tx_id"]), int(row["seq"])), []).append(row)
    intended = int(run.metadata["probe_nodes"]) - 1
    return [
        BroadcastEvent(
            sender=int(tx["tx_id"]),
            sequence=int(tx["seq"]),
            generation_time=float(tx["t_gen_s"]),
            transmission=tx,
            receptions=tuple(by_key.get((int(tx["tx_id"]), int(tx["seq"])), ())),
            intended_receivers=intended,
        )
        for tx in run.transmissions
    ]


def event_state(event: BroadcastEvent, threshold_s: float) -> int:
    if len(event.receptions) < event.intended_receivers:
        return 1
    delays = [float(row["delay_s"]) for row in event.receptions]
    return int(bool(delays) and float(np.median(delays)) > threshold_s)


def fit_two_state_adapter(runs: Iterable[TraceRun], config: dict[str, Any]) -> dict[str, Any]:
    run_list = list(runs)
    delays = np.asarray([
        float(row["delay_s"])
        for run in run_list
        for row in run.receptions
    ])
    if not delays.size:
        raise ValueError("two-state fitting requires successful receptions")
    offset = float(config["state_definition"]["minimum_tail_offset_s"])
    threshold = max(float(np.quantile(delays, 0.75)), float(np.quantile(delays, 0.50)) + offset)
    transition_alpha = float(config["state_definition"]["transition_smoothing"])
    loss_alpha = float(config["state_definition"]["loss_smoothing"])
    probabilities = np.asarray(config["delay_model"]["probabilities"], dtype=float)

    initial = np.full(2, transition_alpha, dtype=float)
    transitions = np.full((2, 2), transition_alpha, dtype=float)
    expected = np.full(2, 2.0 * loss_alpha, dtype=float)
    lost = np.full(2, loss_alpha, dtype=float)
    state_delays: list[list[float]] = [[], []]

    for run in run_list:
        grouped: dict[int, list[BroadcastEvent]] = {}
        for event in broadcast_events(run):
            grouped.setdefault(event.sender, []).append(event)
        for sender_events in grouped.values():
            ordered = sorted(sender_events, key=lambda event: event.sequence)
            states = [event_state(event, threshold) for event in ordered]
            initial[states[0]] += 1
            for previous, current in zip(states, states[1:]):
                transitions[previous, current] += 1
            for event, state in zip(ordered, states):
                expected[state] += event.intended_receivers
                lost[state] += event.intended_receivers - len(event.receptions)
                state_delays[state].extend(float(row["delay_s"]) for row in event.receptions)

    parameters: dict[str, Any] = {
        "model": "two_state_markov_shared_broadcast",
        "contention_threshold_s": threshold,
        "initial_contended_probability": float(initial[1] / np.sum(initial)),
        "transition_matrix": (transitions / transitions.sum(axis=1, keepdims=True)).tolist(),
        "states": {},
    }
    for state, name in enumerate(("clear", "contended")):
        values = np.asarray(state_delays[state] or delays.tolist(), dtype=float)
        parameters["states"][name] = {
            "loss_probability": float(lost[state] / expected[state]),
            "delay_probabilities": probabilities.tolist(),
            "delay_quantiles_s": np.quantile(values, probabilities).tolist(),
            "fitted_receptions": int(len(state_delays[state])),
        }
    return parameters


def _sample_delay(rng: np.random.Generator, state: dict[str, Any]) -> float:
    return float(np.interp(
        rng.random(),
        np.asarray(state["delay_probabilities"], dtype=float),
        np.asarray(state["delay_quantiles_s"], dtype=float),
    ))


def synthesize_two_state_run(reference: TraceRun, parameters: dict[str, Any], seed_offset: int = 1200000) -> TraceRun:
    seed = int(reference.metadata["seed"])
    regime_code = sum(ord(character) for character in str(reference.metadata["regime"]))
    rng = np.random.default_rng(seed_offset + 1000 * regime_code + seed)
    n = int(reference.metadata["probe_nodes"])
    by_sender: dict[int, list[dict[str, float | int]]] = {}
    for tx in reference.transmissions:
        by_sender.setdefault(int(tx["tx_id"]), []).append(tx)
    receptions: list[dict[str, float | int]] = []
    transition = np.asarray(parameters["transition_matrix"], dtype=float)
    for sender, transmissions in sorted(by_sender.items()):
        state = int(rng.random() < float(parameters["initial_contended_probability"]))
        for index, tx in enumerate(sorted(transmissions, key=lambda row: int(row["seq"]))):
            state_name = "contended" if state else "clear"
            state_parameters = parameters["states"][state_name]
            common_delay = _sample_delay(rng, state_parameters)
            for rx_id in range(n):
                if rx_id == sender or rng.random() < float(state_parameters["loss_probability"]):
                    continue
                receptions.append({
                    "uid": int(tx["uid"]),
                    "tx_id": sender,
                    "rx_id": rx_id,
                    "seq": int(tx["seq"]),
                    "t_gen_s": float(tx["t_gen_s"]),
                    "t_rx_s": float(tx["t_gen_s"]) + common_delay,
                    "delay_s": common_delay,
                    "size_bytes": int(tx["size_bytes"]),
                })
            if index + 1 < len(transmissions):
                state = int(rng.random() < transition[state, 1])
    return TraceRun(reference.path, dict(reference.metadata), reference.transmissions, tuple(receptions))


def structural_metrics(runs: Iterable[TraceRun], threshold_s: float) -> dict[str, float | int]:
    total = 0
    contended = 0
    same = 0
    transitions = 0
    for run in runs:
        grouped: dict[int, list[BroadcastEvent]] = {}
        for event in broadcast_events(run):
            grouped.setdefault(event.sender, []).append(event)
        for sender_events in grouped.values():
            states = [
                event_state(event, threshold_s)
                for event in sorted(sender_events, key=lambda event: event.sequence)
            ]
            total += len(states)
            contended += sum(states)
            transitions += max(0, len(states) - 1)
            same += sum(previous == current for previous, current in zip(states, states[1:]))
    return {
        "broadcast_events": total,
        "contention_event_rate": contended / total if total else 0.0,
        "state_persistence": same / transitions if transitions else 1.0,
    }


def _relative_error(candidate: float, reference: float) -> float:
    return abs(candidate - reference) / max(abs(reference), 1e-12)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_g3b2_calibration(config_path: Path, trace_root: Path, output_root: Path) -> Path:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8-sig"))
    runs = discover_trace_runs(trace_root, config["trace_protocol_version"])
    expected_cases = {
        (regime, int(seed))
        for regime in config["regimes"]
        for seed in config["seeds"]["all"]
    }
    actual_cases = {(str(run.metadata["regime"]), int(run.metadata["seed"])) for run in runs}
    if actual_cases != expected_cases:
        raise ValueError(
            f"G3-B2 trace matrix mismatch; missing={sorted(expected_cases - actual_cases)}, "
            f"extra={sorted(actual_cases - expected_cases)}"
        )
    development = {int(seed) for seed in config["seeds"]["development"]}
    holdout = {int(seed) for seed in config["seeds"]["untouched_holdout"]}
    sample_period = float(config["metrics"]["aoi_sampling_period_s"])
    acceptance = config["acceptance"]
    store = ArtifactStore(output_root, "G3B2_two_state_calibration_v1")
    store.write_json("config_frozen.json", config)

    fitted: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for regime in config["regimes"]:
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
        rows.append({
            "regime": regime,
            **{f"reference_{key}": value for key, value in reference_metrics.items()},
            **{f"adapter_{key}": value for key, value in adapter_metrics.items()},
            **{f"reference_{key}": value for key, value in reference_structure.items()},
            **{f"adapter_{key}": value for key, value in adapter_structure.items()},
            **errors,
            "passed": all(gates.values()),
        })

    _write_csv(store.path / "holdout_comparison.csv", rows)
    store.write_json("fitted_two_state_parameters.json", fitted)
    summary = {
        "model_protocol_version": config["model_protocol_version"],
        "trace_root": str(trace_root.resolve()),
        "config_sha256": sha256(config_bytes).hexdigest(),
        "development_seeds": sorted(development),
        "untouched_holdout_seeds": sorted(holdout),
        "regimes": rows,
        "all_holdout_gates_passed": all(bool(row["passed"]) for row in rows),
        "claim_boundary": config["claim_boundary"],
    }
    store.write_json("summary.json", summary)

    from tnsm_sim.plotting.g3_network_two_state import plot_g3b2_calibration

    plot_g3b2_calibration(store.path)
    return store.finalize()
