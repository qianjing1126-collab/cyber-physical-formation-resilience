from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from time import perf_counter

from scipy.optimize import linear_sum_assignment

from tnsm_sim.formation.slots import Slot
from tnsm_sim.recovery.trajectory import PiecewiseQuinticOffset, QuinticOffset, sufficient_duration
from tnsm_sim.types import VehicleState


def compress_ranks(pre_ranks: dict[int, int], failed_ids: set[int]) -> dict[int, int]:
    failed_ranks = sorted(pre_ranks[i] for i in failed_ids)
    result: dict[int, int] = {}
    for vehicle_id, rank in pre_ranks.items():
        if vehicle_id in failed_ids:
            continue
        result[vehicle_id] = rank - sum(1 for failed_rank in failed_ranks if failed_rank < rank)
    if sorted(result.values()) != list(range(1, len(result) + 1)):
        raise ValueError("rank compression did not produce contiguous ranks")
    return result


@dataclass(frozen=True)
class RecoveryPlan:
    method: str
    t_confirm: float
    duration: float
    ranks: dict[int, int]
    targets: dict[int, Slot]
    trajectories: dict[int, QuinticOffset | PiecewiseQuinticOffset]
    s_trajectories: dict[int, QuinticOffset | PiecewiseQuinticOffset]
    planner_time_s: float = 0.0

    def lateral_reference(self, vehicle_id: int, t: float) -> tuple[float, float, float]:
        return self.trajectories[vehicle_id].sample(t)

    def longitudinal_reference(self, vehicle_id: int, t: float) -> tuple[float, float, float]:
        return self.s_trajectories[vehicle_id].sample(t)


def _ordered_target_slots(
    survivors: list[int],
    states: dict[int, VehicleState],
    rank_offsets_s: tuple[float, ...],
    required_overlap: float,
    *,
    center: float = 0.0,
) -> list[Slot]:
    if len(rank_offsets_s) < len(survivors):
        raise ValueError("insufficient target longitudinal offsets")
    raw_d = [0.0]
    for left, right in zip(survivors, survivors[1:]):
        raw_d.append(raw_d[-1] + 0.5 * (states[left].coverage_width + states[right].coverage_width) - required_overlap)
    shift = center - 0.5 * (raw_d[0] + raw_d[-1]) if raw_d else center
    return [
        Slot(
            rank=rank,
            d=raw_d[rank - 1] + shift,
            s_offset=rank_offsets_s[rank - 1],
            coverage_width=states[vehicle_id].coverage_width,
            recovery_speed_factor=states[vehicle_id].recovery_speed_factor,
            recovery_acceleration_factor=states[vehicle_id].recovery_acceleration_factor,
        )
        for rank, vehicle_id in enumerate(survivors, start=1)
    ]


def _make_trajectories(
    method: str,
    t_confirm: float,
    states: dict[int, VehicleState],
    current_s_offsets: dict[int, float],
    targets: dict[int, Slot],
    v_rec_max: float,
    a_rec_max: float,
    min_duration: float,
    sequential: bool,
) -> tuple[dict[int, QuinticOffset], dict[int, QuinticOffset], float]:
    individual: dict[int, float] = {}
    for vehicle_id, target in targets.items():
        ds = target.s_offset - current_s_offsets[vehicle_id]
        dd = target.d - states[vehicle_id].d
        delta = hypot(ds, dd)
        individual[vehicle_id] = sufficient_duration(
            delta,
            v_rec_max * states[vehicle_id].recovery_speed_factor,
            a_rec_max * states[vehicle_id].recovery_acceleration_factor,
            0.0 if delta < 1e-12 else min_duration,
        )

    lateral: dict[int, QuinticOffset] = {}
    longitudinal: dict[int, QuinticOffset] = {}
    if sequential:
        cursor = t_confirm
        for vehicle_id in sorted(targets, key=lambda item: targets[item].rank):
            duration = individual[vehicle_id]
            start = cursor
            if duration > 0:
                cursor += duration
            lateral[vehicle_id] = QuinticOffset(start, duration, states[vehicle_id].d, targets[vehicle_id].d)
            longitudinal[vehicle_id] = QuinticOffset(start, duration, current_s_offsets[vehicle_id], targets[vehicle_id].s_offset)
        total = cursor - t_confirm
    else:
        common = max(individual.values(), default=0.0)
        for vehicle_id in targets:
            lateral[vehicle_id] = QuinticOffset(t_confirm, common, states[vehicle_id].d, targets[vehicle_id].d)
            longitudinal[vehicle_id] = QuinticOffset(t_confirm, common, current_s_offsets[vehicle_id], targets[vehicle_id].s_offset)
        total = common
    return lateral, longitudinal, total


def build_recovery_plan(
    method: str,
    t_confirm: float,
    states: dict[int, VehicleState],
    pre_ranks: dict[int, int],
    failed_ids: set[int],
    current_s_offsets: dict[int, float],
    rank_offsets_s: tuple[float, ...],
    required_overlap: float,
    v_rec_max: float,
    a_rec_max: float,
    min_duration: float,
    center: float = 0.0,
) -> RecoveryPlan:
    if method not in {"B9", "B11", "S3"}:
        raise ValueError(f"unsupported recovery planner {method}")
    started = perf_counter()
    compressed = compress_ranks(pre_ranks, failed_ids)
    survivors = sorted(compressed, key=lambda vehicle_id: pre_ranks[vehicle_id])
    centered_slots = _ordered_target_slots(
        survivors, states, rank_offsets_s, required_overlap, center=center
    )

    if method == "B9":
        # Local cascade is anchored at the lower-rank edge. Vacated slots are
        # filled serially instead of shifting both sides simultaneously.
        shift = states[survivors[0]].d - centered_slots[0].d
        targets = {
            vehicle_id: Slot(**{
                **centered_slots[index].__dict__,
                "d": centered_slots[index].d + shift,
            })
            for index, vehicle_id in enumerate(survivors)
        }
        ranks = compressed
        sequential = True
    elif method == "B11":
        cost = []
        for survivor_index, vehicle_id in enumerate(survivors):
            cost.append([
                hypot(
                    current_s_offsets[vehicle_id] - slot.s_offset,
                    states[vehicle_id].d - slot.d,
                )
                # An order-incompatible assignment would require two plows to
                # exchange lateral order. That is infeasible inside the
                # ordered coverage corridor, so it receives a deterministic
                # prohibitive cost before the Hungarian solve.
                + (0.0 if survivor_index == slot_index else 1e6 * abs(survivor_index - slot_index))
                for slot_index, slot in enumerate(centered_slots)
            ])
        row_index, column_index = linear_sum_assignment(cost)
        targets = {
            survivors[int(row)]: Slot(**{
                **centered_slots[int(column)].__dict__,
                "coverage_width": states[survivors[int(row)]].coverage_width,
            })
            for row, column in zip(row_index, column_index)
        }
        ranks = {vehicle_id: target.rank for vehicle_id, target in targets.items()}
        sequential = False
    else:
        targets = {vehicle_id: centered_slots[index] for index, vehicle_id in enumerate(survivors)}
        ranks = compressed
        sequential = False

    lateral, longitudinal, duration = _make_trajectories(
        method, t_confirm, states, current_s_offsets, targets,
        v_rec_max, a_rec_max, min_duration, sequential,
    )
    return RecoveryPlan(
        method=method,
        t_confirm=t_confirm,
        duration=duration,
        ranks=ranks,
        targets=targets,
        trajectories=lateral,
        s_trajectories=longitudinal,
        planner_time_s=perf_counter() - started,
    )


def build_stage3_plan(
    t_confirm: float,
    states: dict[int, VehicleState],
    pre_ranks: dict[int, int],
    failed_ids: set[int],
    spacing: float,
    v_rec_max: float,
    a_rec_max: float,
    min_duration: float,
    center: float = 0.0,
) -> RecoveryPlan:
    # Backward-compatible homogeneous/F1 wrapper used by G1 tests/configs.
    survivors = [i for i in states if i not in failed_ids]
    width = spacing + 0.4
    for vehicle_id in survivors:
        if states[vehicle_id].coverage_width <= 0:
            states[vehicle_id].coverage_width = width
    return build_recovery_plan(
        "S3", t_confirm, states, pre_ranks, failed_ids,
        current_s_offsets={vehicle_id: 0.0 for vehicle_id in survivors},
        rank_offsets_s=tuple(0.0 for _ in survivors),
        required_overlap=0.4,
        v_rec_max=v_rec_max,
        a_rec_max=a_rec_max,
        min_duration=min_duration,
        center=center,
    )
