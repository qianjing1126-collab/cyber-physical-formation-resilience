from __future__ import annotations

import json
from pathlib import Path
import unittest

from tnsm_sim.experiments.v21_g6 import layout_offsets, simulate_sequential_recovery


ROOT = Path(__file__).resolve().parents[1]


class V21G6Tests(unittest.TestCase):
    def test_vshape_matches_the_frozen_eight_vehicle_definition(self) -> None:
        delta = 2.5
        offsets = layout_offsets("FV_vshape", 8, 3.6, delta)
        self.assertEqual(
            [offsets[index][0] / delta for index in range(1, 9)],
            [-3.0, -2.0, -1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
        )
        self.assertTrue(all(
            abs(offsets[index + 1][1] - offsets[index][1] - 3.6) < 1e-12
            for index in range(1, 8)
        ))

    def test_two_sequential_failures_recover_safely(self) -> None:
        config = json.loads((ROOT / "configs" / "v2_1_g6.json").read_text(encoding="utf-8"))
        g3 = json.loads((ROOT / config["g3_source"] / "summary.json").read_text(encoding="utf-8"))
        profiles = json.loads((ROOT / config["network_profile_file"]).read_text(encoding="utf-8"))
        method = next(item for item in g3["methods"] if item["method"] == config["weight_method"])
        metrics, trace, states, events = simulate_sequential_recovery(
            layout="F1_parallel", seed=int(config["representative_seed"]),
            edges=tuple(tuple(edge) for edge in g3["topology"]["edges"]),
            weights=method["weights"], network_parameters=profiles[config["network_regime"]],
            config=config, keep_trace=True,
        )
        self.assertTrue(metrics["safety_ok"])
        self.assertTrue(metrics["all_recoveries_successful"])
        self.assertLess(metrics["event1_Q_min"], 0.8)
        self.assertLess(metrics["event2_Q_min"], 0.5)
        self.assertAlmostEqual(metrics["final_Q_rec"], 1.0, places=6)
        self.assertEqual(sum(row.get("event") == "physical_dropout" for row in events), 2)
        self.assertGreater(len(trace), 900)
        self.assertGreater(len(states), 3000)

        counter_metrics, counter_trace, _, counter_events = simulate_sequential_recovery(
            layout="F1_parallel", seed=int(config["representative_seed"]),
            edges=tuple(tuple(edge) for edge in g3["topology"]["edges"]),
            weights=method["weights"], network_parameters=profiles[config["network_regime"]],
            config=config, keep_trace=True, enable_recovery=False,
        )
        self.assertFalse(counter_metrics["all_recoveries_successful"])
        self.assertGreater(metrics["R_mission"], counter_metrics["R_mission"])
        self.assertEqual(
            sum(row.get("event") == "recovery_disabled_counterfactual" for row in counter_events), 2
        )
        self.assertEqual(len(trace), len(counter_trace))
        self.assertGreater(abs(float(states[-1]["y"])) if isinstance(states[-1], dict) else 0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
