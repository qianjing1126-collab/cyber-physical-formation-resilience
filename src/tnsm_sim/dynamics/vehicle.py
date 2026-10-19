from __future__ import annotations

from math import atan2, hypot
from typing import Any

from tnsm_sim.types import ControlCommand, VehicleState


def integrate_vehicle(
    state: VehicleState,
    command: ControlCommand,
    dt: float,
    t_next: float,
    path: Any | None = None,
) -> VehicleState:
    if not state.active:
        return VehicleState(**{**state.row(), "t": t_next, "a_x": 0.0, "a_y": 0.0})
    # Velocity-Verlet for the double-integrator P1 model.
    s_next = state.s + state.v_s * dt + 0.5 * command.u_s_applied * dt**2
    d_next = state.d + state.v_d * dt + 0.5 * command.u_d_applied * dt**2
    v_s_next = state.v_s + command.u_s_applied * dt
    v_d_next = state.v_d + command.u_d_applied * dt
    x_next, y_next = (s_next, d_next) if path is None else path.world(s_next, d_next)
    path_heading = 0.0 if path is None else path.heading(s_next)
    curvature = 0.0 if path is None else float(path.curvature(s_next))
    frenet_scale = max(1e-6, 1.0 - curvature * d_next)
    physical_lateral_acceleration = command.u_d_applied + curvature * v_s_next**2 / frenet_scale
    return VehicleState(
        t=t_next,
        vehicle_id=state.vehicle_id,
        x=x_next,
        y=y_next,
        s=s_next,
        d=d_next,
        heading=path_heading + atan2(v_d_next, v_s_next),
        v=hypot(v_s_next, v_d_next),
        v_s=v_s_next,
        v_d=v_d_next,
        a_x=command.u_s_applied,
        a_y=physical_lateral_acceleration,
        active=True,
        slot_rank=state.slot_rank,
        coverage_width=state.coverage_width,
        recovery_speed_factor=state.recovery_speed_factor,
        recovery_acceleration_factor=state.recovery_acceleration_factor,
    )
