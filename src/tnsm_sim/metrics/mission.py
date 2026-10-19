from __future__ import annotations

from dataclasses import asdict, dataclass

from tnsm_sim.metrics.coverage import coverage_components, effective_component
from tnsm_sim.types import VehicleState


@dataclass(frozen=True)
class MissionSnapshot:
    t: float
    active_count: int
    B_eff: float
    v_eff: float
    Q: float
    Q_cap: float
    Q_rec: float

    def row(self) -> dict[str, float | int]:
        return asdict(self)


def nominal_width(n: int, plow_width: float, spacing: float) -> float:
    return 0.0 if n <= 0 else plow_width + (n - 1) * spacing


def heterogeneous_nominal_width(widths: list[float], required_overlap: float) -> float:
    if not widths:
        return 0.0
    if any(width <= 0 for width in widths) or required_overlap < 0:
        raise ValueError("invalid heterogeneous coverage widths")
    return sum(widths) - (len(widths) - 1) * required_overlap


def service_values(B_eff: float, v_eff: float, B0: float, v0: float, Q_cap: float) -> tuple[float, float]:
    if B0 <= 0 or v0 <= 0:
        raise ValueError("nominal width and speed must be positive")
    q_b = min(1.0, max(0.0, B_eff / B0))
    q_v = min(1.0, max(0.0, v_eff / v0))
    q = q_b * q_v
    q_rec = min(1.0, q / Q_cap) if Q_cap > 0 else 0.0
    return q, q_rec


def mission_snapshot(
    t: float,
    states: dict[int, VehicleState],
    n_nominal: int,
    plow_width: float,
    spacing: float,
    v_ref: float,
    epsilon_perp: float,
    epsilon_parallel: float,
    required_overlap: float | None = None,
) -> MissionSnapshot:
    components = coverage_components(states, plow_width, epsilon_perp, epsilon_parallel)
    component = effective_component(components)
    B_eff = 0.0 if component is None else component.width
    v_eff = 0.0 if component is None else component.effective_speed
    active_count = sum(1 for state in states.values() if state.active)
    configured_widths = [
        state.coverage_width if state.coverage_width > 0 else plow_width
        for state in sorted(states.values(), key=lambda item: item.vehicle_id)
    ]
    if required_overlap is None:
        B0 = nominal_width(n_nominal, plow_width, spacing)
        B_cap = nominal_width(active_count, plow_width, spacing)
    else:
        B0 = heterogeneous_nominal_width(configured_widths, required_overlap)
        active_widths = [
            state.coverage_width if state.coverage_width > 0 else plow_width
            for state in states.values() if state.active
        ]
        B_cap = heterogeneous_nominal_width(active_widths, required_overlap)
    Q_cap = B_cap / B0 if B0 > 0 and active_count > 0 else 0.0
    q, q_rec = service_values(B_eff, v_eff, B0, v_ref, Q_cap)
    return MissionSnapshot(t, active_count, B_eff, v_eff, q, Q_cap, q_rec)
