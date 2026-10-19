from __future__ import annotations

from dataclasses import replace
from math import atan2, cos, hypot, sin
from time import perf_counter
from typing import Any

from tnsm_sim.types import VehicleState
from tnsm_sim.recovery.stage3 import RecoveryPlan
from tnsm_sim.recovery.trajectory import PiecewiseQuinticOffset, QuinticOffset


Point = tuple[float, float]


def _rectangle_corners(
    x: float, y: float, heading: float, body_length: float, body_width: float,
) -> tuple[Point, Point, Point, Point]:
    forward = (cos(heading), sin(heading))
    left = (-forward[1], forward[0])
    half_length = 0.5 * body_length
    half_width = 0.5 * body_width
    return tuple(
        (
            x + longitudinal * half_length * forward[0] + lateral * half_width * left[0],
            y + longitudinal * half_length * forward[1] + lateral * half_width * left[1],
        )
        for longitudinal, lateral in ((1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0))
    )


def _dot(point: Point, axis: Point) -> float:
    return point[0] * axis[0] + point[1] * axis[1]


def _point_segment_distance(point: Point, start: Point, end: Point) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-24:
        return hypot(point[0] - start[0], point[1] - start[1])
    fraction = max(0.0, min(1.0, (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / denominator))
    nearest = (start[0] + fraction * dx, start[1] + fraction * dy)
    return hypot(point[0] - nearest[0], point[1] - nearest[1])


def _oriented_rectangle_clearance(
    first: tuple[Point, Point, Point, Point],
    second: tuple[Point, Point, Point, Point],
) -> float:
    """Signed Euclidean boundary clearance for two oriented rectangles."""
    axes: list[Point] = []
    for rectangle in (first, second):
        for index in (0, 1):
            edge = (
                rectangle[(index + 1) % 4][0] - rectangle[index][0],
                rectangle[(index + 1) % 4][1] - rectangle[index][1],
            )
            norm = hypot(edge[0], edge[1])
            axes.append((-edge[1] / norm, edge[0] / norm))

    overlaps = []
    separated = False
    for axis in axes:
        first_projection = [_dot(point, axis) for point in first]
        second_projection = [_dot(point, axis) for point in second]
        overlap = min(max(first_projection), max(second_projection)) - max(
            min(first_projection), min(second_projection)
        )
        overlaps.append(overlap)
        separated = separated or overlap < -1e-12
    if not separated:
        penetration = min(overlaps)
        return 0.0 if penetration <= 1e-12 else -penetration

    return min(
        _point_segment_distance(point, other[index], other[(index + 1) % 4])
        for rectangle, other in ((first, second), (second, first))
        for point in rectangle
        for index in range(4)
    )


def pair_clearance(a: VehicleState, b: VehicleState, body_length: float, body_width: float) -> float:
    """Physical clearance between world-oriented rectangular vehicle envelopes."""
    return _oriented_rectangle_clearance(
        _rectangle_corners(a.x, a.y, a.heading, body_length, body_width),
        _rectangle_corners(b.x, b.y, b.heading, body_length, body_width),
    )


def minimum_clearance(
    states: dict[int, VehicleState],
    body_length: float,
    body_width: float,
    *,
    ordered_only: bool = False,
) -> float:
    active = [state for state in states.values() if state.active]
    if len(active) < 2:
        return float("inf")
    corners = {
        state.vehicle_id: _rectangle_corners(
            state.x, state.y, state.heading, body_length, body_width
        )
        for state in active
    }
    if ordered_only and all(state.slot_rank > 0 for state in active) and len({
        state.slot_rank for state in active
    }) == len(active):
        ordered = sorted(active, key=lambda state: state.slot_rank)
        pairs = zip(ordered, ordered[1:])
    else:
        pairs = (
            (active[first], active[second])
            for first in range(len(active)) for second in range(first + 1, len(active))
        )
    return min(
        _oriented_rectangle_clearance(corners[first.vehicle_id], corners[second.vehicle_id])
        for first, second in pairs
    )


def _reference_states(
    plan: RecoveryPlan,
    t: float,
    *,
    path: Any | None,
    base_s_at_confirm: float,
    reference_speed: float,
) -> dict[int, VehicleState]:
    states: dict[int, VehicleState] = {}
    for vehicle_id in sorted(plan.trajectories):
        s_offset, s_velocity, _ = plan.longitudinal_reference(vehicle_id, t)
        d, d_velocity, _ = plan.lateral_reference(vehicle_id, t)
        s = base_s_at_confirm + reference_speed * (t - plan.t_confirm) + s_offset
        x, y = (s, d) if path is None else path.world(s, d)
        path_heading = 0.0 if path is None else float(path.heading(s))
        heading = path_heading + atan2(d_velocity, reference_speed + s_velocity)
        states[vehicle_id] = VehicleState(
            t=t, vehicle_id=vehicle_id, x=x, y=y, s=s, d=d,
            heading=heading, v=0.0, v_s=reference_speed + s_velocity,
            v_d=d_velocity, active=True, slot_rank=plan.ranks[vehicle_id],
        )
    return states


def planned_minimum_clearance(
    plan: RecoveryPlan,
    body_length: float,
    body_width: float,
    samples: int = 201,
    *,
    path: Any | None = None,
    base_s_at_confirm: float = 0.0,
    reference_speed: float = 5.0,
) -> float:
    """Sample world-oriented reference envelopes over the whole recovery plan."""
    if samples < 2:
        raise ValueError("samples must be at least 2")
    ids = sorted(plan.trajectories)
    if len(ids) < 2:
        return float("inf")
    minimum = float("inf")
    for k in range(samples):
        t = plan.t_confirm + plan.duration * k / (samples - 1)
        states = _reference_states(
            plan, t, path=path, base_s_at_confirm=base_s_at_confirm,
            reference_speed=reference_speed,
        )
        minimum = min(minimum, minimum_clearance(
            states, body_length, body_width, ordered_only=True
        ))
    return minimum


def _offset_clearance(
    offsets: dict[int, tuple[float, float]],
    body_length: float,
    body_width: float,
) -> float:
    ids = sorted(offsets)
    minimum = float("inf")
    for index, first in enumerate(ids):
        for second in ids[index + 1:]:
            ds = abs(offsets[first][0] - offsets[second][0]) - body_length
            dd = abs(offsets[first][1] - offsets[second][1]) - body_width
            if ds >= 0 and dd >= 0:
                clearance = hypot(ds, dd)
            elif ds >= 0:
                clearance = ds
            elif dd >= 0:
                clearance = dd
            else:
                clearance = max(ds, dd)
            minimum = min(minimum, clearance)
    return minimum


def target_minimum_clearance(
    plan: RecoveryPlan,
    body_length: float,
    body_width: float,
    *,
    path: Any | None = None,
    base_s_at_confirm: float = 0.0,
    reference_speed: float = 5.0,
) -> float:
    """World-oriented envelope clearance after every planned maneuver completes."""
    t = plan.t_confirm + plan.duration
    states = _reference_states(
        plan, t, path=path, base_s_at_confirm=base_s_at_confirm,
        reference_speed=reference_speed,
    )
    return minimum_clearance(states, body_length, body_width, ordered_only=True)


def planned_lateral_clearance(plan: RecoveryPlan, body_width: float, samples: int = 201) -> float:
    """Backward-compatible F1 check."""
    ids = sorted(plan.trajectories)
    minimum = float("inf")
    for index in range(samples):
        t = plan.t_confirm + plan.duration * index / (samples - 1)
        offsets = {vehicle_id: plan.lateral_reference(vehicle_id, t)[0] for vehicle_id in ids}
        minimum = min(minimum, *(
            abs(offsets[first] - offsets[second]) - body_width
            for position, first in enumerate(ids) for second in ids[position + 1:]
        ))
    return minimum


def _scaled_trajectory(
    trajectory: QuinticOffset | PiecewiseQuinticOffset,
    t_origin: float,
    scale: float,
) -> QuinticOffset | PiecewiseQuinticOffset:
    def scaled(segment: QuinticOffset) -> QuinticOffset:
        return QuinticOffset(
            t_origin + scale * (segment.t_start - t_origin),
            scale * segment.duration,
            segment.start,
            segment.target,
        )

    if isinstance(trajectory, PiecewiseQuinticOffset):
        return PiecewiseQuinticOffset(tuple(scaled(segment) for segment in trajectory.segments))
    return scaled(trajectory)


def _time_dilated_plan(plan: RecoveryPlan, scale: float) -> RecoveryPlan:
    return replace(
        plan,
        duration=scale * plan.duration,
        trajectories={
            vehicle_id: _scaled_trajectory(trajectory, plan.t_confirm, scale)
            for vehicle_id, trajectory in plan.trajectories.items()
        },
        s_trajectories={
            vehicle_id: _scaled_trajectory(trajectory, plan.t_confirm, scale)
            for vehicle_id, trajectory in plan.s_trajectories.items()
        },
    )


def enforce_recovery_plan_clearance(
    plan: RecoveryPlan,
    *,
    body_length: float,
    body_width: float,
    required_clearance: float,
    path: Any | None = None,
    base_s_at_confirm: float = 0.0,
    reference_speed: float = 5.0,
    samples: int = 201,
    reserve: float = 0.02,
    maximum_time_scale: float = 6.0,
) -> RecoveryPlan:
    """Time-dilate a recovery until its oriented envelopes satisfy clearance."""
    started = perf_counter()
    threshold = required_clearance + reserve
    baseline = planned_minimum_clearance(
        plan, body_length, body_width, samples=samples, path=path,
        base_s_at_confirm=base_s_at_confirm, reference_speed=reference_speed,
    )
    if baseline + 1e-9 >= threshold:
        return replace(plan, planner_time_s=plan.planner_time_s + perf_counter() - started)

    steps = int(round((maximum_time_scale - 1.0) / 0.25))
    for index in range(1, steps + 1):
        candidate = _time_dilated_plan(plan, 1.0 + 0.25 * index)
        clearance = planned_minimum_clearance(
            candidate, body_length, body_width, samples=samples, path=path,
            base_s_at_confirm=base_s_at_confirm, reference_speed=reference_speed,
        )
        if clearance + 1e-9 >= threshold:
            return replace(
                candidate,
                planner_time_s=plan.planner_time_s + perf_counter() - started,
            )
    raise ValueError(
        f"no time-dilated recovery satisfies {threshold:.3f} m oriented-envelope clearance"
    )
