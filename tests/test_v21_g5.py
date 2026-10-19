from __future__ import annotations

import json
import unittest
from pathlib import Path

from tnsm_sim.experiments.v21_g5 import simulate_packet_closed_loop


ROOT = Path(__file__).resolve().parents[1]


class V21G5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT / "configs" / "v2_1_g5.json").read_text(encoding="utf-8"))
        cls.config["duration_s"] = 2.0
        cls.profiles = json.loads((ROOT / cls.config["network_profile_file"]).read_text(encoding="utf-8"))
        g3 = json.loads((ROOT / cls.config["g3_source"] / "summary.json").read_text(encoding="utf-8"))
        cls.edges = tuple(tuple(edge) for edge in g3["topology"]["edges"])
        cls.methods = {item["method"]: item for item in g3["methods"]}

    def run_method(self, method: str):
        return simulate_packet_closed_loop(
            n=8, edges=self.edges, weights=self.methods[method]["weights"],
            network_parameters=self.profiles["moderate"], seed=7301, config=self.config,
        )[0]

    def test_paired_methods_receive_the_same_packet_process(self):
        first = self.run_method("B7_uniform")
        second = self.run_method("S2_delay_constrained")
        for metric in ("pdr", "delay_p95_s", "delay_p99_s", "aoi_p95_s", "queue_overwrite_rate"):
            self.assertAlmostEqual(first[metric], second[metric], places=14)

    def test_packet_closed_loop_produces_finite_metrics(self):
        result = self.run_method("S2_delay_constrained")
        self.assertGreater(result["packet_count"], 0)
        self.assertGreaterEqual(result["pdr"], 0.0)
        self.assertLessEqual(result["pdr"], 1.0)
        self.assertGreater(result["position_iae_m_s"], 0.0)
        self.assertTrue(result["bounded"])


if __name__ == "__main__":
    unittest.main()
