from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from tnsm_sim.network import PacketEventNetwork
from tnsm_sim.seeds import SeedStreams
from tnsm_sim.types import VehicleState

from .artifacts import ArtifactStore


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _edge_key(first: int, second: int) -> str:
    return f"{min(first, second)}-{max(first, second)}"


def _states(time: float, position: np.ndarray, velocity: np.ndarray) -> dict[int, VehicleState]:
    return {
        index + 1: VehicleState(
            t=time, vehicle_id=index + 1, x=float(position[index]), y=0.0,
            s=float(position[index]), d=0.0, heading=0.0, v=float(velocity[index]),
            v_s=float(velocity[index]), v_d=0.0,
        )
        for index in range(len(position))
    }


def _settling_time(time: np.ndarray, signal: np.ndarray, threshold: float, hold: float) -> float | None:
    dt = float(time[1] - time[0])
    count = max(1, int(round(hold / dt)))
    inside = signal <= threshold
    run = 0
    for index, value in enumerate(inside):
        run = run + 1 if value else 0
        if run >= count:
            return float(time[index - count + 1])
    return None


def simulate_packet_closed_loop(
    *, n: int, edges: tuple[tuple[int, int], ...], weights: dict[str, float],
    network_parameters: dict[str, Any], seed: int, config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    dt = float(config["dt_s"])
    duration = float(config["duration_s"])
    steps = int(round(duration / dt)) + 1
    ctrl_steps = int(round(float(config["control_period_s"]) / dt))
    msg_steps = int(round(float(config["message_period_s"]) / dt))
    if min(ctrl_steps, msg_steps) < 1:
        raise ValueError("control and message periods must be at least one integration step")
    time = np.arange(steps, dtype=float) * dt
    base_position = np.asarray([0.80, 0.55, 0.35, 0.15, -0.25, -0.45, -0.65, -0.50], dtype=float)
    if n != len(base_position):
        raise ValueError("the frozen G5 initial condition is defined for N=8")
    position = np.zeros((steps, n), dtype=float)
    velocity = np.zeros((steps, n), dtype=float)
    acceleration = np.zeros(n, dtype=float)
    position[0] = base_position - base_position.mean()
    force_pattern = np.asarray([1.0, -0.8, 0.6, -0.4, 0.3, -0.2, 0.1, -0.6], dtype=float)
    force_pattern -= force_pattern.mean()
    force_pattern /= np.sqrt(np.mean(force_pattern**2))

    network = PacketEventNetwork(
        logical_edges=edges, packet_size=int(config["packet_size_byte"]), seeds=SeedStreams(seed),
        access_delay=0.0, propagation_delay=0.0, jitter=0.0, loss_probability=0.0,
        two_state_parameters=network_parameters,
    )
    active = set(range(1, n + 1))
    aoi_samples: list[float] = []
    missing = 0
    expected = 0
    trace: list[dict[str, float]] = []
    peak_control = 0.0
    control_energy = 0.0

    for index, current_time in enumerate(time):
        network.deliver_until(float(current_time))
        current_states = _states(float(current_time), position[index], velocity[index])
        if index % msg_steps == 0:
            network.broadcast(float(current_time), current_states)
        if index % ctrl_steps == 0:
            new_acceleration = np.zeros(n, dtype=float)
            for receiver in range(1, n + 1):
                for sender in network.active_neighbors(receiver, active):
                    expected += 1
                    sample = network.cache.get(receiver, sender)
                    if sample is None:
                        missing += 1
                        continue
                    sample_index = min(index, max(0, int(round(sample.state_timestamp / dt))))
                    weight = float(weights[_edge_key(receiver, sender)])
                    position_error = position[sample_index, receiver - 1] - float(sample.s)
                    velocity_error = velocity[sample_index, receiver - 1] - float(sample.v_s)
                    new_acceleration[receiver - 1] -= weight * (
                        float(config["k_p"]) * position_error + float(config["k_v"]) * velocity_error
                    )
                    aoi_samples.append(float(current_time) - float(sample.state_timestamp))
            acceleration = new_acceleration
            peak_control = max(peak_control, float(np.max(np.abs(acceleration))))
        disturbance = np.zeros(n, dtype=float)
        disturb_end = float(config["disturbance_duration_s"])
        if current_time <= disturb_end + 1e-12:
            disturbance = (
                float(config["disturbance_amplitude_mps2"]) * force_pattern
                * np.sin(np.pi * current_time / disturb_end)
            )
        total_acceleration = acceleration + disturbance
        control_energy += float(np.mean(acceleration**2)) * dt
        position_rms = float(np.sqrt(np.mean((position[index] - position[index].mean())**2)))
        velocity_rms = float(np.sqrt(np.mean((velocity[index] - velocity[index].mean())**2)))
        trace.append({"time_s": float(current_time), "position_rms_m": position_rms,
                      "velocity_rms_mps": velocity_rms})
        if index == steps - 1:
            break
        position[index + 1] = position[index] + velocity[index] * dt + 0.5 * total_acceleration * dt**2
        velocity[index + 1] = velocity[index] + total_acceleration * dt

    network.finalize(duration)
    # Packets still in flight at the fixed simulation horizon are right-censored,
    # not channel losses.  Exclude them from the PDR/overwrite denominator while
    # reporting their count explicitly for reproducibility.
    eligible_records = [
        record for record in network.records if record.drop_reason != "simulation_end"
    ]
    censored_records = [
        record for record in network.records if record.drop_reason == "simulation_end"
    ]
    received = [record for record in eligible_records if record.status == "received"]
    delays = np.asarray([float(record.t_rx) - float(record.t_gen) for record in received], dtype=float)
    position_rms = np.asarray([row["position_rms_m"] for row in trace], dtype=float)
    velocity_rms = np.asarray([row["velocity_rms_mps"] for row in trace], dtype=float)
    aoi = np.asarray(aoi_samples, dtype=float)
    settling = _settling_time(
        time, position_rms, float(config["settling_threshold_m"]), float(config["settling_hold_s"])
    )
    queue_overwrites = sum(record.drop_reason == "queue_overwrite" for record in eligible_records)
    metrics = {
        "seed": seed,
        "packet_count": len(eligible_records),
        "simulation_end_censored_packets": len(censored_records),
        "pdr": len(received) / len(eligible_records) if eligible_records else 1.0,
        "delay_p95_s": float(np.quantile(delays, 0.95)) if delays.size else float("nan"),
        "delay_p99_s": float(np.quantile(delays, 0.99)) if delays.size else float("nan"),
        "aoi_p95_s": float(np.quantile(aoi, 0.95)) if aoi.size else float("nan"),
        "aoi_max_s": float(np.max(aoi)) if aoi.size else float("nan"),
        "missing_neighbor_fraction": missing / expected if expected else 0.0,
        "queue_overwrite_rate": queue_overwrites / len(eligible_records) if eligible_records else 0.0,
        "peak_position_rms_m": float(np.max(position_rms)),
        "final_position_rms_m": float(position_rms[-1]),
        "position_iae_m_s": float(np.trapezoid(position_rms, time)),
        "velocity_iae_m": float(np.trapezoid(velocity_rms, time)),
        "settling_time_s": settling,
        "settling_time_penalized_s": duration if settling is None else settling,
        "peak_control_acceleration_mps2": peak_control,
        "control_energy_m2ps3": control_energy,
        "bounded": bool(float(np.max(position_rms)) <= float(config["bounded_position_rms_m"])),
    }
    return metrics, trace


def _bootstrap(values: list[float], samples: int, seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "ci95_low": float(np.quantile(means, 0.025)), "ci95_high": float(np.quantile(means, 0.975)),
        "p05": float(np.quantile(array, 0.05)), "p95": float(np.quantile(array, 0.95)),
    }


def run_v21_g5(config_path: Path, output_root: Path, project_root: Path = PROJECT_ROOT) -> Path:
    project_root = project_root.resolve()
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    g3 = json.loads((project_root / config["g3_source"] / "summary.json").read_text(encoding="utf-8"))
    profiles = json.loads((project_root / config["network_profile_file"]).read_text(encoding="utf-8"))
    methods = {item["method"]: item for item in g3["methods"]}
    edges = tuple(tuple(int(value) for value in edge) for edge in g3["topology"]["edges"])
    seeds = range(int(config["seed_start"]), int(config["seed_start"]) + int(config["seed_count"]))
    records: list[dict[str, Any]] = []
    representative: list[dict[str, Any]] = []
    for regime in config["network_regimes"]:
        for seed in seeds:
            for method in config["methods"]:
                metrics, trace = simulate_packet_closed_loop(
                    n=int(config["n"]), edges=edges, weights=methods[method]["weights"],
                    network_parameters=profiles[regime], seed=seed, config=config,
                )
                records.append({"regime": regime, "method": method, **metrics})
                if seed == int(config["representative_seed"]):
                    representative.extend({"regime": regime, "method": method, **row} for row in trace[::5])

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["regime"]), str(row["method"]))].append(row)
    metric_names = (
        "pdr", "delay_p95_s", "aoi_p95_s", "missing_neighbor_fraction", "queue_overwrite_rate",
        "position_iae_m_s", "velocity_iae_m", "settling_time_penalized_s",
        "final_position_rms_m", "peak_control_acceleration_mps2", "control_energy_m2ps3",
    )
    summary_groups = {
        f"{regime}/{method}": {
            metric: _bootstrap(
                [float(row[metric]) for row in rows], int(config["bootstrap_samples"]),
                20260812 + index,
            )
            for index, metric in enumerate(metric_names)
        }
        for (regime, method), rows in grouped.items()
    }
    fairness = True
    for regime in config["network_regimes"]:
        for seed in seeds:
            subset = [row for row in records if row["regime"] == regime and row["seed"] == seed]
            signatures = {
                (round(float(row["pdr"]), 14), round(float(row["delay_p95_s"]), 14),
                 round(float(row["aoi_p95_s"]), 14), round(float(row["queue_overwrite_rate"]), 14))
                for row in subset
            }
            fairness &= len(signatures) == 1
    network_order = []
    for regime in config["network_regimes"]:
        network_order.append(summary_groups[f"{regime}/{config['methods'][0]}"]["pdr"]["mean"])
    paired: dict[str, Any] = {}
    paired_metrics = ("position_iae_m_s", "settling_time_penalized_s", "control_energy_m2ps3")
    for regime in config["network_regimes"]:
        by_seed = defaultdict(dict)
        for row in records:
            if row["regime"] == regime:
                by_seed[int(row["seed"])][str(row["method"])] = row
        for baseline in ("B7_uniform", "B8_convergence_priority"):
            paired[f"{regime}/S2_minus_{baseline}"] = {
                metric: _bootstrap(
                    [float(values["S2_delay_constrained"][metric]) - float(values[baseline][metric])
                     for values in by_seed.values()],
                    int(config["bootstrap_samples"]), 20260830 + index,
                )
                for index, metric in enumerate(paired_metrics)
            }
    acceptance = {
        "run_count_pass": len(records) == len(config["network_regimes"]) * len(config["methods"]) * int(config["seed_count"]),
        "paired_network_process_pass": fairness,
        "all_runs_bounded_pass": all(bool(row["bounded"]) for row in records),
        "pdr_severity_order_pass": all(first >= second for first, second in zip(network_order, network_order[1:])),
    }
    summary = {
        "gate": "V2.1-G5", "passed": all(acceptance.values()), "acceptance": acceptance,
        "g3_source": config["g3_source"], "topology_hash": g3["topology"]["graph_sha256"],
        "run_count": len(records), "group_summary": summary_groups,
        "paired_effects": paired,
        "claim_boundary": config["claim_boundary"],
    }
    store = ArtifactStore(output_root.resolve(), "V21_G5_packet_closed_loop_N8")
    store.write_json("protocol.json", config)
    store.write_json("summary.json", summary)
    store.write_csv("run_records.csv", records, list(records[0]))
    store.write_csv("representative_traces.csv", representative, list(representative[0]))
    return store.finalize()
