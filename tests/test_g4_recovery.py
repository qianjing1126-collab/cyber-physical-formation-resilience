from __future__ import annotations

import unittest

from tnsm_sim.formation import ordered_layout_slots
from tnsm_sim.recovery import build_recovery_plan
from tnsm_sim.recovery.safety import (
    enforce_recovery_plan_clearance,
    pair_clearance,
    planned_minimum_clearance,
    target_minimum_clearance,
)
from tnsm_sim.seeds import SeedStreams
from tnsm_sim.types import VehicleState


def recovery_states(layout_id: str = "F1") -> tuple[dict[int, VehicleState], dict[int, object]]:
    offsets = (-0.25, 0.25, -0.25, 0.25, -0.25, 0.25, -0.25, 0.25)
    slots = ordered_layout_slots(
        8,
        layout_id,
        4.0,
        0.4,
        longitudinal_offsets=offsets if layout_id == "F2" else (),
    )
    states = {
        vehicle_id: VehicleState(
            t=5.5,
            vehicle_id=vehicle_id,
            x=54.0 + slot.s_offset,
            y=slot.d,
            s=54.0 + slot.s_offset,
            d=slot.d,
            heading=0.0,
            v=8.0,
            v_s=8.0,
            v_d=0.0,
            active=True,
            slot_rank=slot.rank,
            coverage_width=slot.coverage_width,
            recovery_speed_factor=slot.recovery_speed_factor,
            recovery_acceleration_factor=slot.recovery_acceleration_factor,
        )
        for vehicle_id, slot in slots.items()
    }
    return states, slots


def make_plan(method: str, failed: set[int], layout_id: str = "F1"):
    states, slots = recovery_states(layout_id)
    for vehicle_id in failed:
        states[vehicle_id].active = False
    survivors = [i for i in range(1, 9) if i not in failed]
    target_offsets = (
        (-0.25, 0.25, -0.25, 0.25, -0.25, 0.25)[:len(survivors)]
        if layout_id == "F2"
        else tuple(0.0 for _ in survivors)
    )
    plan = build_recovery_plan(
        method=method,
        t_confirm=5.5,
        states=states,
        pre_ranks={i: i for i in range(1, 9)},
        failed_ids=failed,
        current_s_offsets={i: slots[i].s_offset for i in survivors},
        rank_offsets_s=target_offsets,
        required_overlap=0.4,
        v_rec_max=1.0,
        a_rec_max=0.55,
        min_duration=4.0,
    )
    return enforce_recovery_plan_clearance(
        plan,
        body_length=8.0,
        body_width=3.0,
        required_clearance=0.5,
        reference_speed=8.0,
        reserve=0.0,
    )


class G4RecoveryPlannerTests(unittest.TestCase):
    def test_world_oriented_rectangle_clearance(self) -> None:
        first = VehicleState(0.0, 1, 0.0, 0.0, 100.0, 100.0, 0.0, 5.0, 5.0, 0.0)
        parallel = VehicleState(0.0, 2, 0.0, 3.6, 100.0, 100.0, 0.0, 5.0, 5.0, 0.0)
        rotated = VehicleState(0.0, 3, 0.0, 3.6, 100.0, 100.0, 0.15, 5.0, 5.0, 0.0)
        overlapping = VehicleState(0.0, 4, 0.0, 2.5, 100.0, 100.0, 0.0, 5.0, 5.0, 0.0)
        self.assertAlmostEqual(pair_clearance(first, parallel, 8.0, 3.0), 0.6, places=10)
        self.assertLess(pair_clearance(first, rotated, 8.0, 3.0), 0.6)
        self.assertLess(pair_clearance(first, overlapping, 8.0, 3.0), 0.0)

    def test_method_identity_and_contiguous_targets(self) -> None:
        for method in ("B9", "B11", "S3"):
            plan = make_plan(method, {4})
            self.assertEqual(plan.method, method)
            self.assertEqual(sorted(plan.ranks.values()), list(range(1, 8)))
            self.assertEqual(set(plan.targets), {1, 2, 3, 5, 6, 7, 8})

    def test_b9_is_anchored_and_serial_whereas_s3_is_simultaneous(self) -> None:
        b9 = make_plan("B9", {4})
        s3 = make_plan("S3", {4})
        self.assertAlmostEqual(b9.targets[1].d, -12.6)
        self.assertAlmostEqual(s3.targets[1].d, -10.8)
        moving_b9 = [trajectory for trajectory in b9.trajectories.values() if trajectory.duration > 0]
        self.assertEqual([trajectory.t_start for trajectory in moving_b9], sorted(trajectory.t_start for trajectory in moving_b9))
        self.assertGreater(len(set(trajectory.t_start for trajectory in moving_b9)), 1)
        self.assertEqual({trajectory.t_start for trajectory in s3.trajectories.values()}, {5.5})
        self.assertGreater(b9.duration, s3.duration)

    def test_b11_assignment_is_unique_deterministic_and_simultaneous(self) -> None:
        first = make_plan("B11", {4})
        second = make_plan("B11", {4})
        self.assertEqual(first.targets, second.targets)
        self.assertEqual(len({slot.rank for slot in first.targets.values()}), 7)
        self.assertEqual({trajectory.t_start for trajectory in first.trajectories.values()}, {5.5})

    def test_simultaneous_failures_compress_once_and_preserve_safety(self) -> None:
        for failed in ({4, 5}, {3, 6}):
            for method in ("B9", "B11", "S3"):
                plan = make_plan(method, failed)
                self.assertEqual(sorted(plan.ranks.values()), list(range(1, 7)))
                self.assertGreaterEqual(
                    planned_minimum_clearance(plan, 8.0, 3.0, reference_speed=8.0), 0.5
                )

    def test_f2_longitudinal_targets_and_limits_are_planned(self) -> None:
        plan = make_plan("S3", {3, 6}, "F2")
        ordered = sorted(plan.targets.values(), key=lambda slot: slot.rank)
        self.assertEqual(tuple(slot.s_offset for slot in ordered), (-0.25, 0.25, -0.25, 0.25, -0.25, 0.25))
        self.assertTrue(any(trajectory.start != trajectory.target for trajectory in plan.s_trajectories.values()))
        self.assertGreaterEqual(
            planned_minimum_clearance(plan, 8.0, 3.0, reference_speed=8.0), 0.5
        )

    def test_f3_residual_geometry_infeasibility_is_distinguishable(self) -> None:
        seeds = SeedStreams(110)
        slots = ordered_layout_slots(
            8, "F3", 4.0, 0.4,
            seed_uniform=lambda vehicle_id, role: seeds.keyed_uniform("vehicle", "F3", vehicle_id, role),
            minimum_center_spacing=3.5,
        )
        states = {
            i: VehicleState(
                5.5, i, 54.0, slot.d, 54.0, slot.d, 0.0, 8.0, 8.0, 0.0,
                active=i not in {4, 5}, slot_rank=i, coverage_width=slot.coverage_width,
                recovery_speed_factor=slot.recovery_speed_factor,
                recovery_acceleration_factor=slot.recovery_acceleration_factor,
            )
            for i, slot in slots.items()
        }
        plan = build_recovery_plan(
            "S3", 5.5, states, {i: i for i in range(1, 9)}, {4, 5},
            current_s_offsets={i: 0.0 for i in states if i not in {4, 5}},
            rank_offsets_s=(0.0,) * 6, required_overlap=0.4,
            v_rec_max=1.0, a_rec_max=0.55, min_duration=4.0,
        )
        self.assertLess(target_minimum_clearance(plan, 8.0, 3.0), 0.5)


if __name__ == "__main__":
    unittest.main()
