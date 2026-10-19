from __future__ import annotations

import unittest

import numpy as np

from tnsm_sim.events import EventSchedule
from tnsm_sim.formation import ordered_layout_slots
from tnsm_sim.metrics.mission import heterogeneous_nominal_width
from tnsm_sim.path import CircularPath, SmoothTransitionPath
from tnsm_sim.seeds import SeedStreams


class G4GeometryAndLayoutTests(unittest.TestCase):
    def test_p2_world_projection_and_curvature(self) -> None:
        path = CircularPath(length=220.0, radius=80.0)
        for s, d in ((0.0, 0.0), (40.0, -2.0), (120.0, 3.0), (220.0, 0.0)):
            x, y = path.world(s, d)
            projected_s, projected_d = path.project(x, y)
            self.assertAlmostEqual(projected_s, s, places=8)
            self.assertAlmostEqual(projected_d, d, places=8)
        self.assertAlmostEqual(path.curvature(100.0), 1.0 / 80.0)
        self.assertLess(8.0**2 * path.curvature(100.0), 1.5)

    def test_p3_is_arc_length_parameterized_and_bounded(self) -> None:
        path = SmoothTransitionPath(220.0, 40.0, 80.0, 12.0)
        self.assertAlmostEqual(path.heading(40.0), 0.0, places=10)
        self.assertAlmostEqual(path.heading(120.0), 0.0, places=10)
        self.assertAlmostEqual(path.curvature(40.0), 0.0, places=10)
        self.assertAlmostEqual(path.curvature(120.0), 0.0, places=10)
        samples = np.linspace(0.0, 220.0, 2001)
        max_nominal_ay = max(abs(path.curvature(float(s))) * 8.0**2 for s in samples)
        self.assertLess(max_nominal_ay, 1.5)
        for s, d in ((10.0, 1.0), (60.0, -1.5), (100.0, 2.0), (180.0, 0.5)):
            x, y = path.world(s, d)
            projected_s, projected_d = path.project(x, y)
            self.assertAlmostEqual(projected_s, s, delta=0.06)
            self.assertAlmostEqual(projected_d, d, delta=0.01)

    def test_f2_respects_frozen_longitudinal_tolerance(self) -> None:
        offsets = (-0.25, 0.25, -0.25, 0.25, -0.25, 0.25, -0.25, 0.25)
        slots = ordered_layout_slots(8, "F2", 4.0, 0.4, longitudinal_offsets=offsets)
        ordered = [slots[i] for i in range(1, 9)]
        self.assertLessEqual(max(slot.s_offset for slot in ordered) - min(slot.s_offset for slot in ordered), 0.5)
        self.assertTrue(all(ordered[i + 1].d > ordered[i].d for i in range(7)))

    def test_f3_is_deterministic_ordered_and_overlap_exact(self) -> None:
        seeds = SeedStreams(101)
        make = lambda: ordered_layout_slots(
            8, "F3", 4.0, 0.4,
            seed_uniform=lambda vehicle_id, role: seeds.keyed_uniform("vehicle", "F3", vehicle_id, role),
            minimum_center_spacing=3.5,
        )
        first, second = make(), make()
        self.assertEqual(first, second)
        widths = [first[i].coverage_width for i in range(1, 9)]
        self.assertTrue(all(3.6 <= width <= 4.4 for width in widths))
        for i in range(1, 8):
            spacing = first[i + 1].d - first[i].d
            self.assertAlmostEqual(spacing, 0.5 * (first[i].coverage_width + first[i + 1].coverage_width) - 0.4)
            self.assertGreaterEqual(spacing, 3.5)
        self.assertAlmostEqual(
            heterogeneous_nominal_width(widths, 0.4),
            max(first[i].d + first[i].coverage_width / 2 for i in first)
            - min(first[i].d - first[i].coverage_width / 2 for i in first),
        )

    def test_simultaneous_targets_share_one_event_tick(self) -> None:
        schedule = EventSchedule(500, 550, (4, 5))
        self.assertTrue(schedule.failure_due(500))
        self.assertTrue(schedule.confirmation_due(550))
        self.assertEqual(schedule.targets, (4, 5))


if __name__ == "__main__":
    unittest.main()
