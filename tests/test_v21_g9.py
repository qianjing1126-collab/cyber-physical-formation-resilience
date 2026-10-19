from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tnsm_sim.experiments.v21_g9 import run_v21_g9
from tnsm_sim.optimization import round_relaxed_topology
from tnsm_sim.topology import circulant_regular_edges, minimum_degree_seed_edges
from tnsm_sim.topology.graph import complete_edges


class V21G9Tests(unittest.TestCase):
    def test_odd_seed_attains_the_minimum_edge_bound(self) -> None:
        for n in (9, 11, 13, 15):
            edges = minimum_degree_seed_edges(n, 3)
            degrees = {node: 0 for node in range(1, n + 1)}
            for first, second in edges:
                degrees[first] += 1
                degrees[second] += 1
            self.assertEqual(len(edges), (3 * n + 1) // 2)
            self.assertEqual(sorted(degrees.values()), [3] * (n - 1) + [4])

    def test_regular_feasible_start_avoids_symmetric_relaxation_tie_bias(self) -> None:
        n, f, edge_count = 10, 2, 15
        candidates = complete_edges(n)
        edges, certificate, record = round_relaxed_topology(
            n, f, edge_count, candidates, [1.0] * len(candidates),
            random_restarts=0, max_swap_rounds=1,
            initial_edges=circulant_regular_edges(n, 3),
            preserve_degree=True,
        )
        self.assertEqual(len(edges), edge_count)
        self.assertEqual(certificate.connectivity_rate, 1.0)
        self.assertEqual(certificate.minimum_degree, 3)
        self.assertEqual(record["random_restarts"], 0)
        self.assertEqual(record["swap_type"], "degree_preserving_two_switch")

    def test_small_offline_pipeline_is_recertified(self) -> None:
        config = {
            "node_counts": [4],
            "f": 1,
            "minimum_regular_degree": 2,
            "exact_enumeration_node_counts": [4],
            "solver": "CLARABEL",
            "time_limit_s": 10.0,
            "rounding_random_restarts": 0,
            "rounding_max_swap_rounds": 2,
            "rounding_seed": 7,
            "w_max": 2.0,
            "k_p": 1.2,
            "k_v": 2.0,
            "delay_requirement": {
                "state_message_period_s": 0.10,
                "control_period_s": 0.05,
                "admitted_network_tail_s": 0.02,
                "engineering_allowance_s": 0.01,
                "target_delay_margin_s": 0.18,
            },
            "numerical_feasibility_tolerance": 5e-5,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            output = run_v21_g9(path, root / "runs")
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["rows"][0]["connectivity_rate"], 1.0)
        self.assertAlmostEqual(summary["rows"][0]["selected_to_exact_gap"], 0.0)
        self.assertGreaterEqual(summary["rows"][0]["certified_delay_margin_s"], 0.18)


if __name__ == "__main__":
    unittest.main()
