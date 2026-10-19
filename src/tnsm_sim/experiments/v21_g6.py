from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from math import sqrt
import os
from pathlib import Path
from typing import Any

import numpy as np

from tnsm_sim.dynamics import integrate_vehicle
from tnsm_sim.metrics import mission_snapshot
from tnsm_sim.metrics.resilience import fixed_window_area, mission_area, minimum_q, recovery_time
from tnsm_sim.network import PacketEventNetwork
from tnsm_sim.path import SmoothTransitionPath
from tnsm_sim.recovery import build_recovery_plan
from tnsm_sim.recovery.safety import (
    minimum_clearance,
    planned_minimum_clearance,
    enforce_recovery_plan_clearance,
)
from tnsm_sim.recovery.trajectory import QuinticOffset, PiecewiseQuinticOffset, quintic_phi
from tnsm_sim.seeds import SeedStreams
from tnsm_sim.types import ControlCommand, ReferenceState, VehicleState

from .artifacts import ArtifactStore


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _edge_key(first: int, second: int) -> str:
    return f"{min(first, second)}-{max(first, second)}"


def layout_offsets(
    layout: str, n: int, lateral_spacing: float, longitudinal_delta: float
) -> dict[int, tuple[float, float]]:
    """Return rank-indexed (longitudinal, lateral) ordered coverage offsets."""
    if n < 1 or lateral_spacing <= 0 or longitudinal_delta <= 0:
        raise ValueError("invalid G6 layout dimensions")
    midpoint = (n + 1) / 2.0
    result: dict[int, tuple[float, float]] = {}
    for rank in range(1, n + 1):
        lateral = (rank - midpoint) * lateral_spacing
        if layout == "F1_parallel":
            longitudinal = 0.0
        elif layout == "F2_oblique":
            longitudinal = (rank - midpoint) * longitudinal_delta
        elif layout == "FV_vshape":
            # For N=8 this gives exactly [-3,-2,-1,0,-1,-2,-3,-4] delta.
            leader_rank = max(1, n // 2)
            longitudinal = -abs(rank - leader_rank) * longitudinal_delta
        else:
            raise ValueError(f"unsupported G6 layout {layout}")
        result[rank] = (float(longitudinal), float(lateral))
    return result


def _base_reference(t: float, initial_s: float, speed: float, ramp_duration: float) -> tuple[float, float, float]:
    if t <= 0:
        return initial_s, 0.0, 0.0
    if t >= ramp_duration:
        return initial_s + 0.5 * speed * ramp_duration + speed * (t - ramp_duration), speed, 0.0
    xi = t / ramp_duration
    phi, dphi, _ = quintic_phi(xi)
    integral_phi = 2.5 * xi**4 - 3.0 * xi**5 + xi**6
    return (
        initial_s + speed * ramp_duration * integral_phi,
        speed * phi,
        speed * dphi / ramp_duration,
    )


def _clip(value: float, limit: float) -> tuple[float, bool]:
    result = max(-limit, min(limit, value))
    return result, abs(result - value) > 1e-12


def _clip_interval(value: float, lower: float, upper: float) -> tuple[float, bool]:
    result = max(lower, min(upper, value))
    return result, abs(result - value) > 1e-12


def _bootstrap(values: list[float], samples: int, seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def simulate_sequential_recovery(
    *, layout: str, seed: int, edges: tuple[tuple[int, int], ...],
    weights: dict[str, float], network_parameters: dict[str, Any],
    config: dict[str, Any], keep_trace: bool = False, enable_recovery: bool = True,
    recovery_method: str = "S3", communication_cutoff_time_s: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    n = int(config["n"])
    dt = float(config["dt_s"])
    duration = float(config["duration_s"])
    steps = int(round(duration / dt)) + 1
    ctrl_steps = int(round(float(config["control_period_s"]) / dt))
    msg_steps = int(round(float(config["message_period_s"]) / dt))
    metric_steps = int(round(float(config["metric_period_s"]) / dt))
    state_steps = int(round(float(config["state_output_period_s"]) / dt))
    if min(ctrl_steps, msg_steps, metric_steps, state_steps) < 1:
        raise ValueError("G6 periods must be at least one dynamics step")

    spacing = float(config["lateral_spacing_m"])
    delta = float(
        config.get("vshape_longitudinal_delta_m", config["longitudinal_delta_m"])
        if layout == "FV_vshape" else config["longitudinal_delta_m"]
    )
    initial_offsets = layout_offsets(layout, n, spacing, delta)
    initial_s = float(config["initial_centerline_s_m"])
    path = SmoothTransitionPath(
        length=float(config["path_length_m"]),
        transition_start=float(config["path_transition_start_m"]),
        transition_length=float(config["path_transition_length_m"]),
        lateral_shift=float(config["path_lateral_shift_m"]),
    )
    states: dict[int, VehicleState] = {}
    for vehicle_id in range(1, n + 1):
        s_offset, d_offset = initial_offsets[vehicle_id]
        x, y = path.world(initial_s + s_offset, d_offset)
        states[vehicle_id] = VehicleState(
            t=0.0, vehicle_id=vehicle_id, x=x, y=y, s=initial_s + s_offset,
            d=d_offset, heading=0.0, v=0.0, v_s=0.0, v_d=0.0,
            active=True, slot_rank=vehicle_id, coverage_width=float(config["plow_width_m"]),
        )

    network = PacketEventNetwork(
        logical_edges=edges, packet_size=int(config["packet_size_byte"]), seeds=SeedStreams(seed),
        access_delay=0.0, propagation_delay=0.0, jitter=0.0, loss_probability=0.0,
        two_state_parameters=network_parameters,
    )
    failures = sorted(config["failure_events"], key=lambda item: float(item["failure_time_s"]))
    failed_ids: set[int] = set()
    transitions: dict[
        int,
        tuple[QuinticOffset | PiecewiseQuinticOffset, QuinticOffset | PiecewiseQuinticOffset],
    ] = {}
    plan_records: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    current_acceleration = {vehicle_id: (0.0, 0.0) for vehicle_id in states}
    state_history = np.zeros((steps, n, 4), dtype=float)
    reference_history = np.zeros((steps, n, 4), dtype=float)
    metric_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    aoi_samples: list[float] = []
    minimum_clearance_all = float("inf")
    peak_ax = 0.0
    peak_ay = 0.0
    saturated_updates = 0
    control_updates = 0
    path_violation = False
    pre_ranks = {vehicle_id: vehicle_id for vehicle_id in states}

    def event_targets(event: dict[str, Any]) -> tuple[int, ...]:
        if "vehicle_ids" in event:
            return tuple(int(value) for value in event["vehicle_ids"])
        return (int(event["vehicle_id"]),)

    def offset_reference(vehicle_id: int, t: float) -> tuple[float, float, float, float, float, float]:
        if vehicle_id in transitions:
            longitudinal, lateral = transitions[vehicle_id]
            s_value, s_velocity, s_acceleration = longitudinal.sample(t)
            d_value, d_velocity, d_acceleration = lateral.sample(t)
            return s_value, d_value, s_velocity, d_velocity, s_acceleration, d_acceleration
        s_value, d_value = initial_offsets[vehicle_id]
        return s_value, d_value, 0.0, 0.0, 0.0, 0.0

    def full_reference(vehicle_id: int, t: float) -> ReferenceState:
        base_s, base_v, base_a = _base_reference(
            t, initial_s, float(config["reference_speed_mps"]), float(config["speed_ramp_duration_s"])
        )
        s_off, d_off, vs_off, vd_off, as_off, ad_off = offset_reference(vehicle_id, t)
        return ReferenceState(base_s + s_off, d_off, base_v + vs_off, vd_off, base_a + as_off, ad_off)

    def create_plan(t: float, failure_index: int) -> None:
        active_order = sorted(vehicle_id for vehicle_id, state in states.items() if state.active)
        target_rank_offsets = layout_offsets(layout, len(active_order), spacing, delta)
        base_s, _, _ = _base_reference(
            t, initial_s, float(config["reference_speed_mps"]),
            float(config["speed_ramp_duration_s"]),
        )
        current_s_offsets = {
            vehicle_id: states[vehicle_id].s - base_s for vehicle_id in active_order
        }
        plan = build_recovery_plan(
            method=recovery_method,
            t_confirm=t,
            states=states,
            pre_ranks=pre_ranks,
            failed_ids=set(failed_ids),
            current_s_offsets=current_s_offsets,
            rank_offsets_s=tuple(
                target_rank_offsets[rank][0] for rank in range(1, len(active_order) + 1)
            ),
            required_overlap=float(config["required_overlap_m"]),
            v_rec_max=float(config["recovery_max_offset_speed_mps"]),
            a_rec_max=float(config["recovery_max_offset_acceleration_mps2"]),
            min_duration=float(config["recovery_minimum_duration_s"]),
        )
        plan = enforce_recovery_plan_clearance(
            plan,
            body_length=float(config["vehicle_body_length_m"]),
            body_width=float(config["vehicle_body_width_m"]),
            required_clearance=float(config["safety_clearance_m"]),
            path=path,
            base_s_at_confirm=base_s,
            reference_speed=float(config["reference_speed_mps"]),
        )
        planned_clearance = planned_minimum_clearance(
            plan,
            float(config["vehicle_body_length_m"]),
            float(config["vehicle_body_width_m"]),
            path=path,
            base_s_at_confirm=base_s,
            reference_speed=float(config["reference_speed_mps"]),
        )
        transitions.clear()
        for vehicle_id in active_order:
            transitions[vehicle_id] = (
                plan.s_trajectories[vehicle_id], plan.trajectories[vehicle_id]
            )
            states[vehicle_id].slot_rank = plan.ranks[vehicle_id]
        plan_records.append({
            "failure_index": failure_index + 1,
            "confirmation_time_s": t,
            "duration_s": plan.duration,
            "completion_time_s": t + plan.duration,
            "active_ids": ",".join(str(value) for value in active_order),
            "planner_time_s": plan.planner_time_s,
            "planned_oriented_clearance_m": planned_clearance,
            "target_offsets": json.dumps(
                {
                    vehicle_id: {
                        "s": plan.targets[vehicle_id].s_offset,
                        "d": plan.targets[vehicle_id].d,
                    }
                    for vehicle_id in active_order
                },
                sort_keys=True,
            ),
        })

    for index in range(steps):
        t = round(index * dt, 12)
        coordination_available = (
            communication_cutoff_time_s is None
            or t < communication_cutoff_time_s - 1.0e-12
        )
        for failure_index, event in enumerate(failures):
            if abs(t - float(event["failure_time_s"])) <= dt / 2:
                for vehicle_id in event_targets(event):
                    if states[vehicle_id].active:
                        states[vehicle_id].active = False
                        failed_ids.add(vehicle_id)
                        event_rows.append({
                            "event": "physical_dropout", "failure_index": failure_index + 1,
                            "vehicle_id": vehicle_id, "time_s": t,
                        })
            if abs(t - float(event["confirmation_time_s"])) <= dt / 2:
                if enable_recovery:
                    create_plan(t, failure_index)
                    event_name = "stage3_recovery_started"
                else:
                    event_name = "recovery_disabled_counterfactual"
                event_rows.append({
                    "event": event_name, "failure_index": failure_index + 1,
                    "vehicle_id": ",".join(str(value) for value in event_targets(event)),
                    "time_s": t,
                })

        references = {vehicle_id: full_reference(vehicle_id, t) for vehicle_id in states}
        for vehicle_id, state in states.items():
            state.t = t
            state_history[index, vehicle_id - 1] = (state.s, state.d, state.v_s, state.v_d)
            ref = references[vehicle_id]
            reference_history[index, vehicle_id - 1] = (ref.s, ref.d, ref.v_s, ref.v_d)

        network.deliver_until(t)
        if index % msg_steps == 0 and coordination_available:
            network.broadcast(t, states)
        active_ids = {vehicle_id for vehicle_id, state in states.items() if state.active}

        if index % ctrl_steps == 0:
            for receiver in sorted(active_ids):
                state = states[receiver]
                ref = references[receiver]
                coupling_s = 0.0
                coupling_d = 0.0
                if coordination_available:
                    for sender in network.active_neighbors(receiver, active_ids):
                        sample = network.cache.get(receiver, sender)
                        if sample is None:
                            continue
                        sample_index = min(index, max(0, int(round(float(sample.state_timestamp) / dt))))
                        own_past = state_history[sample_index, receiver - 1]
                        own_ref = reference_history[sample_index, receiver - 1]
                        neighbor_ref = reference_history[sample_index, sender - 1]
                        weight = float(weights[_edge_key(receiver, sender)])
                        relative_s = (own_past[0] - float(sample.s)) - (own_ref[0] - neighbor_ref[0])
                        relative_vs = (own_past[2] - float(sample.v_s)) - (own_ref[2] - neighbor_ref[2])
                        relative_d = (own_past[1] - float(sample.d)) - (own_ref[1] - neighbor_ref[1])
                        relative_vd = (own_past[3] - float(sample.v_d)) - (own_ref[3] - neighbor_ref[3])
                        coupling_s -= weight * (
                            float(config["k_p_s"]) * relative_s + float(config["k_v_s"]) * relative_vs
                        )
                        coupling_d -= weight * (
                            float(config["k_p_d"]) * relative_d + float(config["k_v_d"]) * relative_vd
                        )
                        aoi_samples.append(t - float(sample.state_timestamp))
                gain = float(config["coupling_gain"])
                command_s = (
                    ref.a_s + float(config["k_p_s"]) * (ref.s - state.s)
                    + float(config["k_v_s"]) * (ref.v_s - state.v_s) + gain * coupling_s
                )
                command_d = (
                    ref.a_d + float(config["k_p_d"]) * (ref.d - state.d)
                    + float(config["k_v_d"]) * (ref.v_d - state.v_d) + gain * coupling_d
                )
                applied_s, saturated_s = _clip(command_s, float(config["max_longitudinal_acceleration_mps2"]))
                lateral_limit = float(config["max_lateral_acceleration_mps2"])
                curvature = float(path.curvature(state.s))
                frenet_scale = max(1e-6, 1.0 - curvature * state.d)
                path_lateral_acceleration = curvature * state.v_s**2 / frenet_scale
                applied_d, saturated_d = _clip_interval(
                    command_d,
                    -lateral_limit - path_lateral_acceleration,
                    lateral_limit - path_lateral_acceleration,
                )
                current_acceleration[receiver] = (applied_s, applied_d)
                saturated_updates += int(saturated_s or saturated_d)
                control_updates += 1

        clearance = minimum_clearance(
            states, float(config["vehicle_body_length_m"]), float(config["vehicle_body_width_m"]),
            ordered_only=True,
        )
        minimum_clearance_all = min(minimum_clearance_all, clearance)
        peak_ax = max(peak_ax, *(abs(current_acceleration[i][0]) for i in active_ids))
        peak_ay = max(peak_ay, *(abs(states[i].a_y) for i in active_ids))
        path_violation = path_violation or any(not path.inside(states[i].s) for i in active_ids)

        if index % metric_steps == 0:
            epsilon_parallel = (
                float(config["parallel_longitudinal_tolerance_m"])
                if layout == "F1_parallel" else float(config["shaped_longitudinal_tolerance_m"])
            )
            snap = mission_snapshot(
                t=t, states=states, n_nominal=n, plow_width=float(config["plow_width_m"]),
                spacing=spacing, v_ref=float(config["reference_speed_mps"]), epsilon_perp=0.0,
                epsilon_parallel=epsilon_parallel,
                required_overlap=float(config["required_overlap_m"]),
            )
            row = {"layout": layout, "seed": seed, **snap.row(), "min_clearance_m": clearance}
            metric_rows.append(row)
            if keep_trace:
                trace_rows.append(row)

        if keep_trace and index % state_steps == 0:
            for vehicle_id, state in states.items():
                state_rows.append({"layout": layout, "seed": seed, **state.row()})

        if index == steps - 1:
            break
        t_next = round((index + 1) * dt, 12)
        for vehicle_id in sorted(states):
            state = states[vehicle_id]
            applied_s, applied_d = current_acceleration[vehicle_id] if state.active else (0.0, 0.0)
            ref = references[vehicle_id]
            command = ControlCommand(
                t=t, vehicle_id=vehicle_id, ref_s=ref.s, ref_d=ref.d,
                u_s_cmd=applied_s, u_d_cmd=applied_d,
                u_s_applied=applied_s, u_d_applied=applied_d,
                saturated=False, neighbor_count=0,
            )
            states[vehicle_id] = integrate_vehicle(state, command, dt, t_next, path=path)

    network.finalize(duration)
    eligible = [record for record in network.records if record.drop_reason != "simulation_end"]
    received = [record for record in eligible if record.status == "received"]
    delays = np.asarray([float(record.t_rx) - float(record.t_gen) for record in received], dtype=float)
    q_samples = [(float(row["t"]), float(row["Q"])) for row in metric_rows]
    qrec_samples = [(float(row["t"]), float(row["Q_rec"])) for row in metric_rows]
    safety_ok = (
        minimum_clearance_all + 1e-9 >= float(config["safety_clearance_m"])
        and peak_ax <= float(config["max_longitudinal_acceleration_mps2"]) + 1e-9
        and peak_ay <= float(config["max_lateral_acceleration_mps2"]) + 1e-9
        and not path_violation
    )
    metrics: dict[str, Any] = {
        "layout": layout,
        "seed": seed,
        "recovery_enabled": enable_recovery,
        "recovery_method": recovery_method,
        "communication_cutoff_time_s": communication_cutoff_time_s,
        "pdr": len(received) / len(eligible) if eligible else 1.0,
        "delay_p95_s": float(np.quantile(delays, 0.95)) if delays.size else 0.0,
        "aoi_p95_s": float(np.quantile(aoi_samples, 0.95)) if aoi_samples else 0.0,
        "minimum_clearance_m": minimum_clearance_all,
        "peak_abs_ax_mps2": peak_ax,
        "peak_abs_ay_mps2": peak_ay,
        "saturation_fraction": saturated_updates / control_updates if control_updates else 0.0,
        "path_violation": path_violation,
        "safety_ok": safety_ok,
        "R_mission": mission_area(q_samples),
        "final_Q": q_samples[-1][1],
        "final_Q_rec": qrec_samples[-1][1],
    }
    all_recovered = True
    for failure_index, event in enumerate(failures):
        start = float(event["failure_time_s"])
        end = float(failures[failure_index + 1]["failure_time_s"]) if failure_index + 1 < len(failures) else duration
        window = end - start
        recovery, censored = recovery_time(
            qrec_samples, start, window, float(config["recovery_holding_time_s"]),
            float(config["recovery_target_ratio"]),
        )
        metrics.update({
            f"event{failure_index + 1}_Q_min": minimum_q(q_samples, start, window),
            f"event{failure_index + 1}_R_Te": fixed_window_area(q_samples, start, window),
            f"event{failure_index + 1}_recovery_time_s": window if recovery is None else recovery,
            f"event{failure_index + 1}_recovery_censored": censored,
            f"event{failure_index + 1}_plan_duration_s": (
                plan_records[failure_index]["duration_s"] if enable_recovery else 0.0
            ),
            f"event{failure_index + 1}_planner_time_s": (
                plan_records[failure_index].get("planner_time_s", 0.0)
                if enable_recovery else 0.0
            ),
        })
        all_recovered &= not censored
    metrics["all_recoveries_successful"] = bool(enable_recovery and all_recovered and safety_ok)
    return metrics, trace_rows, state_rows, event_rows + plan_records


def _run_g6_case(payload: tuple[Any, ...]) -> tuple[Any, ...]:
    layout, seed, edges, weights, network_parameters, config, keep_trace = payload
    metrics, trace, states, events = simulate_sequential_recovery(
        layout=layout, seed=seed, edges=edges, weights=weights,
        network_parameters=network_parameters, config=config,
        keep_trace=keep_trace,
    )
    counter_metrics, counter_trace, _, _ = simulate_sequential_recovery(
        layout=layout, seed=seed, edges=edges, weights=weights,
        network_parameters=network_parameters, config=config,
        keep_trace=keep_trace, enable_recovery=False,
    )
    return layout, seed, metrics, trace, states, events, counter_metrics, counter_trace


def run_v21_g6(config_path: Path, output_root: Path, project_root: Path = PROJECT_ROOT) -> Path:
    project_root = project_root.resolve()
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    g3 = json.loads((project_root / config["g3_source"] / "summary.json").read_text(encoding="utf-8"))
    profiles = json.loads((project_root / config["network_profile_file"]).read_text(encoding="utf-8"))
    method = next(item for item in g3["methods"] if item["method"] == config["weight_method"])
    edges = tuple(tuple(int(value) for value in edge) for edge in g3["topology"]["edges"])
    network_parameters = profiles[str(config["network_regime"])]
    seeds = range(int(config["seed_start"]), int(config["seed_start"]) + int(config["seed_count"]))
    records: list[dict[str, Any]] = []
    counterfactual_records: list[dict[str, Any]] = []
    representative_trace: list[dict[str, Any]] = []
    representative_states: list[dict[str, Any]] = []
    representative_events: list[dict[str, Any]] = []
    tasks = [
        (
            str(layout), seed, edges, method["weights"], network_parameters, config,
            seed == int(config["representative_seed"]),
        )
        for seed in seeds for layout in config["layouts"]
    ]
    workers = int(config.get("parallel_workers", min(12, os.cpu_count() or 1)))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_run_g6_case, tasks, chunksize=1))
    for layout, seed, metrics, trace, states, events, counter_metrics, counter_trace in results:
        metrics["R_mission_no_recovery"] = counter_metrics["R_mission"]
        metrics["R_mission_gain"] = float(metrics["R_mission"]) - float(counter_metrics["R_mission"])
        metrics["service_gain_area_s"] = float(config["duration_s"]) * float(metrics["R_mission_gain"])
        records.append(metrics)
        counterfactual_records.append(counter_metrics)
        if seed == int(config["representative_seed"]):
            representative_trace.extend({"scenario": "S3_recovery", **row} for row in trace)
            representative_trace.extend({"scenario": "no_recovery", **row} for row in counter_trace)
            representative_states.extend({"scenario": "S3_recovery", **row} for row in states)
            representative_events.extend({"layout": layout, "seed": seed, **row} for row in events)

    metric_names = (
        "pdr", "delay_p95_s", "aoi_p95_s", "minimum_clearance_m",
        "peak_abs_ax_mps2", "peak_abs_ay_mps2", "saturation_fraction", "R_mission",
        "event1_Q_min", "event1_R_Te", "event1_recovery_time_s",
        "event2_Q_min", "event2_R_Te", "event2_recovery_time_s",
        "R_mission_no_recovery", "R_mission_gain", "service_gain_area_s",
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["layout"])].append(row)
    summary_groups = {
        layout: {
            metric: _bootstrap(
                [float(row[metric]) for row in rows], int(config["bootstrap_samples"]),
                20260901 + metric_index,
            )
            for metric_index, metric in enumerate(metric_names)
        }
        for layout, rows in grouped.items()
    }
    paired_network = True
    counterfactual_pairing = True
    for seed in seeds:
        signatures = {
            (
                round(float(row["pdr"]), 14), round(float(row["delay_p95_s"]), 14),
                round(float(row["aoi_p95_s"]), 14),
            )
            for row in records if int(row["seed"]) == seed
        }
        paired_network &= len(signatures) == 1
    for main, counter in zip(records, counterfactual_records, strict=True):
        counterfactual_pairing &= all(
            abs(float(main[name]) - float(counter[name])) <= 1e-14
            for name in ("pdr", "delay_p95_s", "aoi_p95_s")
        )
    expected = len(config["layouts"]) * int(config["seed_count"])
    acceptance = {
        "run_count_pass": len(records) == expected,
        "paired_network_process_pass": paired_network,
        "counterfactual_network_pairing_pass": counterfactual_pairing,
        "all_runs_safe_pass": all(bool(row["safety_ok"]) for row in records),
        "all_sequential_recoveries_pass": all(bool(row["all_recoveries_successful"]) for row in records),
        "final_residual_capacity_pass": all(
            abs(float(row["final_Q_rec"]) - 1.0) <= 0.02 for row in records
        ),
    }
    summary = {
        "gate": "V2.1-G6", "passed": all(acceptance.values()), "acceptance": acceptance,
        "run_count": len(records), "g3_source": config["g3_source"],
        "topology_hash": g3["topology"]["graph_sha256"],
        "weight_method": config["weight_method"], "group_summary": summary_groups,
        "claim_boundary": config["claim_boundary"],
    }
    store = ArtifactStore(output_root.resolve(), "V21_G6_sequential_recovery_N8")
    store.write_json("protocol.json", config)
    store.write_json("summary.json", summary)
    store.write_csv("run_records.csv", records, list(records[0]))
    store.write_csv("counterfactual_records.csv", counterfactual_records, list(counterfactual_records[0]))
    store.write_csv("representative_metrics.csv", representative_trace, list(representative_trace[0]))
    store.write_csv("representative_states.csv", representative_states, list(representative_states[0]))
    store.write_csv("representative_events.csv", representative_events, list(representative_events[0]))
    return store.finalize()
