from __future__ import annotations

from dataclasses import dataclass

from tnsm_sim.types import VehicleState


Interval = tuple[float, float]


def interval_measure(intervals: list[Interval]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    left, right = ordered[0]
    for nxt_left, nxt_right in ordered[1:]:
        if nxt_left <= right:
            right = max(right, nxt_right)
        else:
            total += right - left
            left, right = nxt_left, nxt_right
    return total + right - left


@dataclass(frozen=True)
class CoverageComponent:
    vehicle_ids: tuple[int, ...]
    width: float
    effective_speed: float


def coverage_components(
    states: dict[int, VehicleState],
    plow_width: float,
    epsilon_perp: float,
    epsilon_parallel: float,
) -> list[CoverageComponent]:
    active = sorted((state for state in states.values() if state.active), key=lambda state: (state.slot_rank, state.vehicle_id))
    if not active:
        return []
    groups: list[list[VehicleState]] = [[active[0]]]

    def width(state: VehicleState) -> float:
        return state.coverage_width if state.coverage_width > 0 else plow_width

    for previous, current in zip(active, active[1:]):
        previous_width = width(previous)
        current_width = width(current)
        prev_interval = (previous.d - previous_width / 2, previous.d + previous_width / 2)
        cur_interval = (current.d - current_width / 2, current.d + current_width / 2)
        lateral_gap = max(0.0, max(prev_interval[0], cur_interval[0]) - min(prev_interval[1], cur_interval[1]))
        longitudinal_gap = abs(previous.s - current.s)
        if lateral_gap <= epsilon_perp + 1e-12 and longitudinal_gap <= epsilon_parallel + 1e-12:
            groups[-1].append(current)
        else:
            groups.append([current])
    components: list[CoverageComponent] = []
    for group in groups:
        intervals = [(state.d - width(state) / 2, state.d + width(state) / 2) for state in group]
        components.append(CoverageComponent(
            vehicle_ids=tuple(state.vehicle_id for state in group),
            width=interval_measure(intervals),
            effective_speed=min(max(0.0, state.v_s) for state in group),
        ))
    return components


def effective_component(components: list[CoverageComponent]) -> CoverageComponent | None:
    if not components:
        return None
    return max(components, key=lambda component: (component.width, component.effective_speed))
