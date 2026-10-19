from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Slot:
    rank: int
    d: float
    s_offset: float = 0.0
    coverage_width: float = 0.0
    recovery_speed_factor: float = 1.0
    recovery_acceleration_factor: float = 1.0


def ordered_f1_slots(n: int, spacing: float, *, center: float = 0.0) -> dict[int, Slot]:
    if n < 1 or spacing <= 0:
        raise ValueError("n and spacing must be positive")
    midpoint = (n + 1) / 2.0
    return {
        vehicle_id: Slot(rank=vehicle_id, d=center + (vehicle_id - midpoint) * spacing)
        for vehicle_id in range(1, n + 1)
    }


def target_slots_for_survivors(
    survivor_ids: list[int], spacing: float, *, center: float = 0.0
) -> dict[int, Slot]:
    ordered = sorted(survivor_ids)
    midpoint = (len(ordered) + 1) / 2.0
    return {
        vehicle_id: Slot(rank=index, d=center + (index - midpoint) * spacing)
        for index, vehicle_id in enumerate(ordered, start=1)
    }


def ordered_layout_slots(
    n: int,
    layout_id: str,
    coverage_width: float,
    required_overlap: float,
    *,
    longitudinal_offsets: tuple[float, ...] = (),
    seed_uniform=None,
    heterogeneity_min: float = 0.9,
    heterogeneity_max: float = 1.1,
    minimum_center_spacing: float = 0.0,
) -> dict[int, Slot]:
    if n < 1 or coverage_width <= 0 or not 0 <= required_overlap < coverage_width:
        raise ValueError("invalid ordered-layout dimensions")
    if layout_id not in {"F1", "F2", "F3"}:
        raise ValueError(f"unsupported ordered layout {layout_id}")
    if layout_id == "F2" and len(longitudinal_offsets) != n:
        raise ValueError("F2 requires one longitudinal offset per vehicle")
    if layout_id == "F3" and seed_uniform is None:
        raise ValueError("F3 requires a deterministic keyed-uniform source")

    widths: dict[int, float] = {}
    speed_factors: dict[int, float] = {}
    acceleration_factors: dict[int, float] = {}
    if layout_id == "F3":
        span = heterogeneity_max - heterogeneity_min
        # F3 is sampled from the frozen heterogeneity range conditional on a
        # safe nominal formation. Unsafe initial geometries are not algorithm
        # failures and therefore must not enter the recovery comparison.
        for attempt in range(1000):
            candidate = {
                vehicle_id: coverage_width * (
                    heterogeneity_min
                    + span * seed_uniform((vehicle_id, attempt), "coverage_width")
                )
                for vehicle_id in range(1, n + 1)
            }
            if all(
                0.5 * (candidate[i] + candidate[i + 1]) - required_overlap
                >= minimum_center_spacing - 1e-12
                for i in range(1, n)
            ):
                widths.update(candidate)
                break
        else:
            raise ValueError("unable to sample a nominally safe F3 layout")

    for vehicle_id in range(1, n + 1):
        if layout_id == "F3":
            speed_factors[vehicle_id] = heterogeneity_min + span * seed_uniform(vehicle_id, "recovery_speed")
            acceleration_factors[vehicle_id] = heterogeneity_min + span * seed_uniform(vehicle_id, "recovery_acceleration")
        else:
            widths[vehicle_id] = coverage_width
            speed_factors[vehicle_id] = 1.0
            acceleration_factors[vehicle_id] = 1.0

    raw_d = {1: 0.0}
    for vehicle_id in range(2, n + 1):
        previous = vehicle_id - 1
        raw_d[vehicle_id] = raw_d[previous] + 0.5 * (
            widths[previous] + widths[vehicle_id]
        ) - required_overlap
    center = 0.5 * (raw_d[1] + raw_d[n])
    return {
        vehicle_id: Slot(
            rank=vehicle_id,
            d=raw_d[vehicle_id] - center,
            s_offset=(longitudinal_offsets[vehicle_id - 1] if layout_id == "F2" else 0.0),
            coverage_width=widths[vehicle_id],
            recovery_speed_factor=speed_factors[vehicle_id],
            recovery_acceleration_factor=acceleration_factors[vehicle_id],
        )
        for vehicle_id in range(1, n + 1)
    }
