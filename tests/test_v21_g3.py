from __future__ import annotations

import unittest

import numpy as np

from tnsm_sim.experiments.v21_g3 import evaluate_weights
from tnsm_sim.weights import lambda_cap_for_delay_margin, modal_delay_margin


class V21G3Tests(unittest.TestCase):
    def test_delay_requirement_is_inverted_without_rounding_to_an_integer(self):
        cap = lambda_cap_for_delay_margin(0.18, 1.2, 2.0)
        self.assertAlmostEqual(cap, 4.1527508833819775, places=10)
        self.assertAlmostEqual(modal_delay_margin(cap, 1.2, 2.0), 0.18, places=12)

    def test_uniform_scaling_preserves_ratio_and_improves_delay_margin(self):
        edges = ((1, 2), (1, 3), (2, 3), (2, 4), (3, 4))
        full = evaluate_weights(4, 0, edges, np.ones(len(edges)), 1.2, 2.0)
        half = evaluate_weights(4, 0, edges, 0.5 * np.ones(len(edges)), 1.2, 2.0)
        self.assertAlmostEqual(half["worst_lambda2"], 0.5 * full["worst_lambda2"])
        self.assertAlmostEqual(half["worst_lambda_max"], 0.5 * full["worst_lambda_max"])
        self.assertGreater(half["delay_margin_s"], full["delay_margin_s"])

    def test_evaluation_reports_budget_statistics(self):
        edges = ((1, 2), (1, 3), (2, 3))
        result = evaluate_weights(3, 0, edges, np.asarray([0.5, 0.75, 1.0]), 1.2, 2.0)
        self.assertAlmostEqual(result["weight_sum"], 2.25)
        self.assertAlmostEqual(result["weight_max"], 1.0)
        self.assertAlmostEqual(result["weight_min"], 0.5)


if __name__ == "__main__":
    unittest.main()
