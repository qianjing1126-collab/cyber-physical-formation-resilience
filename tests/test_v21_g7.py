from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from tnsm_sim.experiments.v21_g6 import simulate_sequential_recovery
from tnsm_sim.experiments.v21_g7 import balanced_failure_pair_schedule


ROOT = Path(__file__).resolve().parents[1]


class V21G7PilotTests(unittest.TestCase):
    def test_formal_pair_schedule_covers_and_balances_all_pairs(self) -> None:
        schedule = balanced_failure_pair_schedule(8, 30)
        expected_adjacent = {(first, first + 1) for first in range(1, 8)}
        expected_nonadjacent = {
            (first, second)
            for first in range(1, 9)
            for second in range(first + 2, 9)
        }
        self.assertEqual(set(schedule["A_adjacent"]), expected_adjacent)
        self.assertEqual(set(schedule["N_nonadjacent"]), expected_nonadjacent)
        for pairs in schedule.values():
            counts = {pair: pairs.count(pair) for pair in set(pairs)}
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_paired_methods_have_fair_recovery_authority(self) -> None:
        config = json.loads((ROOT / "configs" / "v2_1_g7_pilot.json").read_text(encoding="utf-8"))
        g3 = json.loads((ROOT / config["g3_source"] / "summary.json").read_text(encoding="utf-8"))
        profiles = json.loads((ROOT / config["network_profile_file"]).read_text(encoding="utf-8"))
        weight_record = next(item for item in g3["methods"] if item["method"] == config["weight_method"])
        scenario = deepcopy(config)
        scenario["failure_events"] = [{
            "vehicle_ids": [3, 6],
            "failure_time_s": 15.0,
            "confirmation_time_s": 15.5,
        }]
        results = {}
        for method in ("B9", "B11", "S3"):
            metrics, _, _, _ = simulate_sequential_recovery(
                layout="FV_vshape", seed=8101,
                edges=tuple(tuple(edge) for edge in g3["topology"]["edges"]),
                weights=weight_record["weights"],
                network_parameters=profiles[config["network_regime"]],
                config=scenario, recovery_method=method,
            )
            results[method] = metrics
        self.assertFalse(results["B9"]["all_recoveries_successful"])
        self.assertTrue(results["B11"]["all_recoveries_successful"])
        self.assertTrue(results["S3"]["all_recoveries_successful"])
        self.assertAlmostEqual(results["B11"]["event1_R_Te"], results["S3"]["event1_R_Te"], places=12)
        self.assertAlmostEqual(
            results["B11"]["event1_recovery_time_s"],
            results["S3"]["event1_recovery_time_s"],
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
