from __future__ import annotations

from dataclasses import asdict
import json
from math import isfinite, sqrt
from pathlib import Path
from typing import Any

from tnsm_sim.clock import SimulationClock
from tnsm_sim.config import FrozenConfig
from tnsm_sim.control import DistributedController
from tnsm_sim.dynamics import integrate_vehicle
from tnsm_sim.events import EventSchedule
from tnsm_sim.formation import ordered_layout_slots
from tnsm_sim.io import RunStore
from tnsm_sim.metrics.mission import mission_snapshot
from tnsm_sim.metrics.resilience import fixed_window_area, minimum_q, mission_area, recovery_time
from tnsm_sim.network import IdealNetwork, PacketEventNetwork
from tnsm_sim.network.aoi import age_of_information
from tnsm_sim.path import CircularPath, SmoothTransitionPath, StraightPath
from tnsm_sim.recovery import RecoveryPlan, build_recovery_plan
from tnsm_sim.recovery.safety import (
    minimum_clearance,
    planned_minimum_clearance,
    enforce_recovery_plan_clearance,
    target_minimum_clearance,
)
from tnsm_sim.seeds import SeedStreams
from tnsm_sim.topology.graph import algebraic_connectivity, canonical_edges, largest_component_ratio, residual_edges, ring_edges
from tnsm_sim.types import ControlCommand, ReferenceState, SimEvent, VehicleState


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(val) for key, val in value.items()}
    if hasattr(value, "items"):
        return {key: _plain(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def build_run_id(config: FrozenConfig) -> str:
    meta = config.section("meta")
    n = int(config.section("vehicle")["N"])
    seed = int(meta["base_seed"])
    return f"{meta['module']}_{meta['scenario']}_{meta['method']}_N{n}_seed{seed:04d}"


def _build_path(config: FrozenConfig):
    path_cfg = config.section("path")
    path_id = str(path_cfg["path_id"])
    if path_id == "P1":
        return StraightPath(float(path_cfg["length"]))
    if path_id == "P2":
        return CircularPath(float(path_cfg["length"]), float(path_cfg["radius"]))
    if path_id == "P3":
        return SmoothTransitionPath(
            length=float(path_cfg["length"]),
            transition_start=float(path_cfg["transition_start"]),
            transition_length=float(path_cfg["transition_length"]),
            lateral_shift=float(path_cfg["lateral_shift"]),
        )
    raise ValueError(f"unsupported path {path_id}")


def _initial_states(
    config: FrozenConfig,
    t0: float,
    path,
    seeds: SeedStreams,
    s0: float = 10.0,
):
    vehicle = config.section("vehicle")
    formation = config.section("formation")
    n = int(vehicle["N"])
    layout_id = str(formation["layout_id"])
    offsets = tuple(float(value) for value in formation.get("longitudinal_offsets", ()))
    heterogeneity_min = float(formation.get("heterogeneity_min", 0.9))
    heterogeneity_max = float(formation.get("heterogeneity_max", 1.1))
    slots = ordered_layout_slots(
        n=n,
        layout_id=layout_id,
        coverage_width=float(vehicle["plow_width"]),
        required_overlap=float(formation["required_overlap"]),
        longitudinal_offsets=offsets,
        seed_uniform=lambda vehicle_id, role: seeds.keyed_uniform("vehicle", layout_id, vehicle_id, role),
        heterogeneity_min=heterogeneity_min,
        heterogeneity_max=heterogeneity_max,
        minimum_center_spacing=float(vehicle["body_width"]) + float(config.section("recovery")["safety_clearance"]),
    )
    states: dict[int, VehicleState] = {}
    for vehicle_id, slot in slots.items():
        initial_s = s0 + slot.s_offset
        x, y = path.world(initial_s, slot.d)
        states[vehicle_id] = VehicleState(
            t=t0,
            vehicle_id=vehicle_id,
            x=x,
            y=y,
            s=initial_s,
            d=slot.d,
            heading=path.heading(initial_s),
            v=float(vehicle["v_ref"]),
            v_s=float(vehicle["v_ref"]),
            v_d=0.0,
            active=True,
            slot_rank=slot.rank,
            coverage_width=slot.coverage_width,
            recovery_speed_factor=slot.recovery_speed_factor,
            recovery_acceleration_factor=slot.recovery_acceleration_factor,
        )
    return states, slots


def _curvature_acceleration_bound(path, slots, v_ref: float, samples: int = 2001) -> float:
    """Conservative nominal path-following lateral acceleration over all slots."""
    length = float(path.length)
    offsets = [slot.d for slot in slots.values()]
    peak = 0.0
    for index in range(samples):
        s = length * index / (samples - 1)
        curvature = float(path.curvature(s))
        for d in offsets:
            denominator = 1.0 - curvature * d
            if denominator <= 0:
                raise ValueError("formation offset crosses the local path-curvature center")
            peak = max(peak, abs(curvature * v_ref**2 / denominator))
    return peak


def run_simulation(config: FrozenConfig, project_root: Path | None = None) -> tuple[Path, dict[str, Any]]:
    project_root = (project_root or PROJECT_ROOT).resolve()
    cfg = config.data
    meta = config.section("meta")
    time_cfg = config.section("time")
    vehicle_cfg = config.section("vehicle")
    formation_cfg = config.section("formation")
    network_cfg = config.section("network")
    control_cfg = config.section("control")
    failure_cfg = config.section("failure")
    recovery_cfg = config.section("recovery")
    output_cfg = config.section("output")

    clock = SimulationClock(float(time_cfg["t_start"]), float(time_cfg["t_end"]), float(time_cfg["dt_dyn"]))
    ctrl_ticks = clock.period_ticks(float(time_cfg["T_ctrl"]))
    msg_ticks = clock.period_ticks(float(time_cfg["T_msg"]))
    state_ticks = clock.period_ticks(float(output_cfg["state_period"]))
    metric_ticks = clock.period_ticks(float(output_cfg["metric_period"]))
    failure_tick = None if failure_cfg["t_d"] is None else clock.tick_of(float(failure_cfg["t_d"]))
    confirmation_tick = None if failure_cfg["t_c"] is None else clock.tick_of(float(failure_cfg["t_c"]))
    schedule = EventSchedule(failure_tick, confirmation_tick, tuple(int(x) for x in failure_cfg["targets"]))
    seeds = SeedStreams(int(meta["base_seed"]))

    path = _build_path(config)
    states, initial_slots = _initial_states(config, clock.t_start, path, seeds)
    pre_ranks = {vehicle_id: state.slot_rank for vehicle_id, state in states.items()}
    configured_edges = network_cfg.get("logical_edges")
    logical_edges = (
        canonical_edges([tuple(int(value) for value in edge) for edge in configured_edges])
        if configured_edges is not None else ring_edges(int(vehicle_cfg["N"]))
    )
    if network_cfg["backend"] == "ideal":
        network = IdealNetwork(float(network_cfg["fixed_delay"]), int(network_cfg["packet_size"]), logical_edges)
    else:
        network = PacketEventNetwork(
            logical_edges=logical_edges,
            packet_size=int(network_cfg["packet_size"]),
            seeds=seeds,
            access_delay=float(network_cfg["access_delay"]),
            propagation_delay=float(network_cfg["propagation_delay"]),
            jitter=float(network_cfg["jitter"]),
            loss_probability=float(network_cfg["loss_probability"]),
            forced_drop_sequences=tuple(int(x) for x in network_cfg.get("forced_drop_sequences", ())),
            delay_probabilities=tuple(float(x) for x in network_cfg.get("delay_cdf", {}).get("probabilities", ())),
            delay_quantiles=tuple(float(x) for x in network_cfg.get("delay_cdf", {}).get("quantiles_s", ())),
            two_state_parameters=network_cfg.get("two_state"),
        )
    v_ref = float(vehicle_cfg["v_ref"])
    nominal_path_ay = _curvature_acceleration_bound(path, initial_slots, v_ref)
    lateral_control_margin = float(vehicle_cfg["a_y_max"]) - nominal_path_ay
    if lateral_control_margin <= 0:
        raise ValueError("path curvature leaves no lateral-acceleration margin for formation control")
    controller = DistributedController(
        kp_s=float(control_cfg["kp_s"]), kv_s=float(control_cfg["kv_s"]),
        kp_d=float(control_cfg["kp_d"]), kv_d=float(control_cfg["kv_d"]),
        coupling_gain=float(control_cfg["coupling_gain"]), edge_weight=float(control_cfg["edge_weight"]),
        a_x_max=float(vehicle_cfg["a_x_max"]), a_y_max=lateral_control_margin,
    )

    store = RunStore(project_root, build_run_id(config))
    store.write_text("canonical_config.yaml", json.dumps(_plain(cfg), indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    store.write_environment()
    store.write_json("solver.json", {"status": "not_used_in_G1", "threads": 1})

    state_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    history: dict[tuple[int, float], VehicleState] = {}
    current_commands: dict[int, ControlCommand] = {}
    plan: RecoveryPlan | None = None
    attempted_plan: RecoveryPlan | None = None
    feasibility_status: str | None = None
    failed_ids: set[int] = set()
    minimum_clearance_all = float("inf")
    peak_ax = 0.0
    peak_ay = 0.0
    path_violation = False
    aoi_samples_all: list[float] = []

    s0 = sum(
        states[vehicle_id].s - initial_slots[vehicle_id].s_offset
        for vehicle_id in states
    ) / len(states)
    recovery_method = str(recovery_cfg.get("method", "S3")).upper()

    def reference_at(vehicle_id: int, t: float) -> ReferenceState:
        base_s = s0 + v_ref * (t - clock.t_start)
        if plan is not None and vehicle_id in plan.trajectories and t >= plan.t_confirm - 1e-12:
            d_ref, v_d_ref, a_d_ref = plan.lateral_reference(vehicle_id, t)
            s_offset, v_s_offset, a_s_ref = plan.longitudinal_reference(vehicle_id, t)
        else:
            d_ref, v_d_ref, a_d_ref = initial_slots[vehicle_id].d, 0.0, 0.0
            s_offset, v_s_offset, a_s_ref = initial_slots[vehicle_id].s_offset, 0.0, 0.0
        return ReferenceState(base_s + s_offset, d_ref, v_ref + v_s_offset, v_d_ref, a_s_ref, a_d_ref)

    def state_at(vehicle_id: int, t: float) -> VehicleState:
        key = (vehicle_id, round(t, 12))
        if key not in history:
            raise KeyError(f"missing self-history for vehicle {vehicle_id} at {t}")
        return history[key]

    # Exact initial command before the first control update.
    for vehicle_id, state in states.items():
        ref = reference_at(vehicle_id, clock.t_start)
        current_commands[vehicle_id] = ControlCommand(clock.t_start, vehicle_id, ref.s, ref.d, 0, 0, 0, 0, False, 0)

    for tick in clock.ticks():
        t = clock.time(tick)

        # Priority 1: physical failure occurrence. It changes the physical set,
        # but does not yet authorize rank compression.
        if schedule.failure_due(tick):
            for vehicle_id in schedule.targets:
                if states[vehicle_id].active:
                    states[vehicle_id].active = False
                    failed_ids.add(vehicle_id)
                    event_rows.append(SimEvent(
                        event_id=f"physical_dropout_{vehicle_id}", kind="physical_dropout",
                        target=str(vehicle_id), t_occurrence=t,
                        t_confirm=float(failure_cfg["t_c"]), t_effective=t,
                    ).row())

        # Priority 2: confirmation gate and Stage 3 planning.
        if schedule.confirmation_due(tick):
            event_rows.append(SimEvent(
                event_id="dropout_confirmation", kind="physical_dropout_confirmed",
                target=",".join(map(str, sorted(failed_ids))),
                t_occurrence=float(failure_cfg["t_d"]), t_confirm=t, t_effective=t,
            ).row())
            coordination_active = {vehicle_id for vehicle_id, state in states.items() if state.active}
            coordination_edges = residual_edges(logical_edges, coordination_active)
            coordination_infeasible = (
                bool(recovery_cfg.get("require_connected_coordination", False))
                and largest_component_ratio(sorted(coordination_active), coordination_edges) < 1.0 - 1e-12
            )
            if coordination_infeasible:
                feasibility_status = "COMMUNICATION_INFEASIBLE"
                event_rows.append(SimEvent(
                    event_id="recovery_infeasible", kind="communication_infeasible",
                    target=",".join(map(str, sorted(coordination_active))),
                    t_occurrence=t, t_confirm=t, t_effective=t,
                    payload={
                        "method": recovery_method,
                        "largest_component_ratio": largest_component_ratio(
                            sorted(coordination_active), coordination_edges
                        ),
                        "residual_edges": [list(edge) for edge in coordination_edges],
                    },
                ).row())
            elif bool(recovery_cfg["enabled"]) and recovery_method != "B0":
                active_order = sorted(
                    (vehicle_id for vehicle_id in states if vehicle_id not in failed_ids),
                    key=lambda vehicle_id: pre_ranks[vehicle_id],
                )
                current_s_offsets = {
                    vehicle_id: states[vehicle_id].s - (s0 + v_ref * (t - clock.t_start))
                    for vehicle_id in active_order
                }
                configured_offsets = tuple(float(value) for value in formation_cfg.get("longitudinal_offsets", ()))
                rank_offsets_s = (
                    configured_offsets[:len(active_order)]
                    if str(formation_cfg["layout_id"]) == "F2"
                    else tuple(0.0 for _ in active_order)
                )
                recovery_acceleration_limit = min(
                    float(recovery_cfg["a_rec_max"]), lateral_control_margin
                )
                attempted_plan = build_recovery_plan(
                    method=recovery_method,
                    t_confirm=t, states=states, pre_ranks=pre_ranks, failed_ids=failed_ids,
                    current_s_offsets=current_s_offsets,
                    rank_offsets_s=rank_offsets_s,
                    required_overlap=float(formation_cfg["required_overlap"]),
                    v_rec_max=float(recovery_cfg["v_rec_max"]),
                    a_rec_max=recovery_acceleration_limit,
                    min_duration=float(recovery_cfg["min_adjustment_time"]),
                )
                try:
                    attempted_plan = enforce_recovery_plan_clearance(
                        attempted_plan,
                        body_length=float(vehicle_cfg["body_length"]),
                        body_width=float(vehicle_cfg["body_width"]),
                        required_clearance=float(recovery_cfg["safety_clearance"])
                        + 2 * float(recovery_cfg["tracking_error_bound"]),
                        path=path,
                        base_s_at_confirm=s0 + v_ref * (t - clock.t_start),
                        reference_speed=v_ref,
                    )
                except ValueError:
                    feasibility_status = "PLANNER_INFEASIBLE"
                planned_clearance = planned_minimum_clearance(
                    attempted_plan,
                    float(vehicle_cfg["body_length"]),
                    float(vehicle_cfg["body_width"]),
                    path=path,
                    base_s_at_confirm=s0 + v_ref * (t - clock.t_start),
                    reference_speed=v_ref,
                )
                target_clearance = target_minimum_clearance(
                    attempted_plan,
                    float(vehicle_cfg["body_length"]),
                    float(vehicle_cfg["body_width"]),
                    path=path,
                    base_s_at_confirm=s0 + v_ref * (t - clock.t_start),
                    reference_speed=v_ref,
                )
                required_clearance = float(recovery_cfg["safety_clearance"]) + 2 * float(recovery_cfg["tracking_error_bound"])
                if target_clearance + 1e-9 < required_clearance:
                    feasibility_status = "GEOMETRY_INFEASIBLE"
                elif planned_clearance + 1e-9 < required_clearance:
                    feasibility_status = "PLANNER_INFEASIBLE"
                if feasibility_status is None:
                    plan = attempted_plan
                    for vehicle_id, rank in plan.ranks.items():
                        states[vehicle_id].slot_rank = rank
                    event_rows.append(SimEvent(
                        event_id="stage3_plan", kind="formation_recovery_planned",
                        target=",".join(map(str, sorted(plan.ranks))),
                        t_occurrence=t, t_confirm=t, t_effective=t,
                        payload={
                            "method": plan.method,
                            "duration": plan.duration,
                            "planner_time_s": plan.planner_time_s,
                            "planned_clearance": planned_clearance,
                            "target_clearance": target_clearance,
                            "targets": {
                                i: {
                                    "rank": plan.targets[i].rank,
                                    "s_offset": plan.targets[i].s_offset,
                                    "d": plan.targets[i].d,
                                }
                                for i in plan.targets
                            }
                        },
                    ).row())
                else:
                    event_rows.append(SimEvent(
                        event_id="recovery_infeasible",
                        kind=feasibility_status.lower(),
                        target=",".join(map(str, sorted(attempted_plan.ranks))),
                        t_occurrence=t, t_confirm=t, t_effective=t,
                        payload={
                            "method": attempted_plan.method,
                            "required_clearance": required_clearance,
                            "planned_clearance": planned_clearance,
                            "target_clearance": target_clearance,
                            "planner_time_s": attempted_plan.planner_time_s,
                        },
                    ).row())

        # Store current state before network/control so own history can be
        # retrieved at every packet generation timestamp.
        for vehicle_id, state in states.items():
            state.t = t
            history[(vehicle_id, t)] = VehicleState(**state.row())

        # Priority 3/4: deliver pending packets, then generate due messages.
        network.deliver_until(t)
        if tick % msg_ticks == 0:
            network.broadcast(t, states)

        active_ids = {i for i, state in states.items() if state.active}

        # Priority 5/6: update continuous references and controller.
        if tick % ctrl_ticks == 0:
            for vehicle_id in sorted(active_ids):
                state = states[vehicle_id]
                neighbors = network.active_neighbors(vehicle_id, active_ids)
                ref = reference_at(vehicle_id, t)
                command = controller.compute(t, state, ref, neighbors, network.cache, state_at, reference_at)
                current_commands[vehicle_id] = command
                command_row = command.row()
                command_row.update({
                    "error_s": state.s - ref.s,
                    "error_d": state.d - ref.d,
                    "error_v_s": state.v_s - ref.v_s,
                    "error_v_d": state.v_d - ref.v_d,
                })
                control_rows.append(command_row)

        # Priority 8 (at the current timestamp): write state and evaluate.
        if tick % state_ticks == 0:
            state_rows.extend(state.row() for state in states.values())

        clearance = minimum_clearance(
            states, float(vehicle_cfg["body_length"]), float(vehicle_cfg["body_width"]),
            ordered_only=True,
        )
        minimum_clearance_all = min(minimum_clearance_all, clearance)
        peak_ax = max(peak_ax, *(abs(state.a_x) for state in states.values() if state.active))
        peak_ay = max(peak_ay, *(abs(state.a_y) for state in states.values() if state.active))
        path_violation = path_violation or any(not path.inside(state.s) for state in states.values() if state.active)

        if tick % metric_ticks == 0:
            snap = mission_snapshot(
                t=t, states=states, n_nominal=int(vehicle_cfg["N"]),
                plow_width=float(vehicle_cfg["plow_width"]), spacing=float(formation_cfg["slot_spacing"]),
                v_ref=v_ref, epsilon_perp=float(formation_cfg["epsilon_perp"]),
                epsilon_parallel=float(formation_cfg["epsilon_parallel"]),
                required_overlap=float(formation_cfg["required_overlap"]),
            )
            active_edges = residual_edges(logical_edges, active_ids)
            ages = [
                age for receiver in active_ids
                for sender in network.active_neighbors(receiver, active_ids)
                if (age := age_of_information(t, network.cache.get(receiver, sender))) is not None
            ]
            aoi_samples_all.extend(float(age) for age in ages)
            row = snap.row()
            row.update({
                "S_c": largest_component_ratio(sorted(active_ids), active_edges),
                "lambda2": algebraic_connectivity(sorted(active_ids), active_edges),
                "aoi_mean": sum(ages) / len(ages) if ages else "",
                "aoi_max": max(ages) if ages else "",
                "min_clearance": clearance,
                "max_abs_ax": max((abs(state.a_x) for state in states.values() if state.active), default=0.0),
                "max_abs_ay": max((abs(state.a_y) for state in states.values() if state.active), default=0.0),
            })
            metric_rows.append(row)

        if tick == clock.final_tick:
            break

        # Priority 7: integrate all active vehicles to the next dynamics tick.
        t_next = clock.time(tick + 1)
        for vehicle_id, state in list(states.items()):
            command = current_commands[vehicle_id]
            next_state = integrate_vehicle(state, command, clock.dt, t_next, path=path)
            next_state.slot_rank = state.slot_rank
            states[vehicle_id] = next_state

    network.finalize(clock.t_end)
    q_samples = [(float(row["t"]), float(row["Q"])) for row in metric_rows]
    qrec_samples = [(float(row["t"]), float(row["Q_rec"])) for row in metric_rows]
    safety_ok = (
        minimum_clearance_all + 1e-9 >= float(recovery_cfg["safety_clearance"])
        and peak_ax <= float(vehicle_cfg["a_x_max"]) + 1e-9
        and peak_ay <= float(vehicle_cfg["a_y_max"]) + 1e-9
        and not path_violation
    )
    sorted_aoi = sorted(aoi_samples_all)
    aoi_p95 = (
        sorted_aoi[min(len(sorted_aoi) - 1, int(round(0.95 * (len(sorted_aoi) - 1))))]
        if sorted_aoi else 0.0
    )
    position_rmse = sqrt(
        sum(float(row["error_s"])**2 + float(row["error_d"])**2 for row in control_rows) / len(control_rows)
    )
    velocity_rmse = sqrt(
        sum(float(row["error_v_s"])**2 + float(row["error_v_d"])**2 for row in control_rows) / len(control_rows)
    )
    final_metrics: dict[str, Any] = {
        "protocol_version": meta["protocol_version"],
        "architecture_version": meta["architecture_version"],
        "config_hash": config.config_hash,
        "run_id": store.run_id,
        "Q_min_mission": min(q for _, q in q_samples),
        "R_mission": mission_area(q_samples),
        "minimum_clearance": minimum_clearance_all,
        "peak_abs_ax": peak_ax,
        "peak_abs_ay": peak_ay,
        "path_violation": path_violation,
        "nominal_path_peak_abs_ay": nominal_path_ay,
        "lateral_control_margin": lateral_control_margin,
        "safety_ok": safety_ok,
        "packet_count": len(network.records),
        "PDR": (
            sum(record.status == "received" for record in network.records) / len(network.records)
            if network.records else 1.0
        ),
        "AoI_max": max(
            (float(row["aoi_max"]) for row in metric_rows if row["aoi_max"] != ""),
            default=0.0,
        ),
        "AoI_p95": aoi_p95,
        "position_tracking_rmse": position_rmse,
        "velocity_tracking_rmse": velocity_rmse,
        "scientific_status": (
            "SAFETY_VIOLATION" if not safety_ok
            else feasibility_status or "COMPLETE"
        ),
        "seed_streams": {
            "base": seeds.base, "graph": seeds.graph, "net": seeds.net,
            "vehicle": seeds.vehicle, "disturb": seeds.disturb,
            "fault": seeds.fault, "stats": seeds.stats,
        },
    }
    if failure_cfg["type"] != "none":
        td = float(failure_cfg["t_d"])
        Te = float(recovery_cfg["T_e"])
        Th = float(recovery_cfg["T_h"])
        at_confirm = min(metric_rows, key=lambda row: abs(float(row["t"]) - float(failure_cfg["t_c"])))
        qcap_plus = float(at_confirm["Q_cap"])
        q_target = float(recovery_cfg["zeta_r"]) * qcap_plus
        tr, censored = recovery_time(q_samples, td, Te, Th, q_target)
        final_metrics.update({
            "Q_cap_plus": qcap_plus,
            "Q_target": q_target,
            "Q_min": minimum_q(q_samples, td, Te),
            "T_r": tr,
            "T_r_censored": censored,
            "R_Te": fixed_window_area(q_samples, td, Te),
            "R_rec": fixed_window_area(qrec_samples, td, Te),
            "recovery_success": (not censored) and safety_ok,
            "confirmation_latency": float(failure_cfg["t_c"]) - td,
            "stage3_duration": None if attempted_plan is None else attempted_plan.duration,
            "recovery_method": recovery_method,
            "planner_time_s": None if attempted_plan is None else attempted_plan.planner_time_s,
            "feasibility_status": feasibility_status,
        })
        if feasibility_status is not None:
            final_metrics["recovery_success"] = False
        elif censored and safety_ok:
            final_metrics["scientific_status"] = "RECOVERY_FAILED"

    adjacency_rows = [
        {"graph_role": "logical", "scenario": "nominal", "source": i, "target": j}
        for i, j in logical_edges
    ] + [
        {"graph_role": "residual", "scenario": "final", "source": i, "target": j}
        for i, j in residual_edges(logical_edges, {i for i, state in states.items() if state.active})
    ]
    weight_rows = [
        {"source": i, "target": j, "weight": float(control_cfg["edge_weight"]), "budget_role": "G1_fixed"}
        for i, j in logical_edges
    ]
    store.write_csv("states.csv", state_rows)
    store.write_csv("controls.csv", control_rows)
    store.write_csv("packets.csv", [record.row() for record in network.records],
                    ["tx_id", "rx_id", "seq", "t_gen", "t_tx", "t_rx", "status", "drop_reason", "size"])
    store.write_csv("events.csv", event_rows,
                    ["event_id", "kind", "target", "t_occurrence", "t_confirm", "t_effective", "payload"])
    store.write_csv("metrics_timeseries.csv", metric_rows)
    store.write_csv("adjacency.csv", adjacency_rows)
    store.write_csv("weights.csv", weight_rows)
    store.write_json("metrics.json", final_metrics)
    final_dir = store.finalize({
        "run_state": "COMPLETE",
        "scientific_status": final_metrics["scientific_status"],
        "stop_reason": "t_end_reached",
    })
    return final_dir, final_metrics
