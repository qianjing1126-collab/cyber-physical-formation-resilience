from __future__ import annotations

import unittest

from tnsm_sim.optimization import (
    enumerate_exact_max_z,
    round_relaxed_topology,
    solve_stage1_relaxation,
    solve_stage2_sdp,
)


class G3OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.relaxation = solve_stage1_relaxation(6, 2, 9, "CLARABEL", 30.0)
        cls.edges, cls.certificate, _ = round_relaxed_topology(
            6, 2, 9,
            [tuple(edge) for edge in cls.relaxation["candidate_edges"]],
            cls.relaxation["edge_fractions"],
            random_restarts=8,
            seed=20260806,
        )
        cls.exact = enumerate_exact_max_z(6, 2, 9)

    def test_stage1_bound_incumbent_and_exact_order(self):
        upper = self.relaxation["relaxation_upper_bound"]
        exact = self.exact["optimal_worst_lambda2"]
        incumbent = self.certificate.worst_lambda2
        self.assertGreaterEqual(upper + 1e-7, exact)
        self.assertGreaterEqual(exact + 1e-7, incumbent)
        self.assertAlmostEqual(exact, incumbent, places=7)
        self.assertEqual(self.certificate.connectivity_rate, 1.0)

    def test_stage2_independent_solvers_and_recertification(self):
        results = [
            solve_stage2_sdp(6, 2, self.edges, 4.0, 9.0, 2.0, solver, 30.0)
            for solver in ("CLARABEL", "SCS")
        ]
        for result in results:
            self.assertIn(result["status"], {"optimal", "optimal_inaccurate"})
            recertification = result["numerical_recertification"]
            self.assertLess(recertification["upper_lmi_violation"], 5e-5)
            self.assertLess(recertification["lower_lmi_violation"], 5e-5)
        difference = abs(results[0]["gamma"] - results[1]["gamma"]) / results[0]["gamma"]
        self.assertLess(difference, 1e-3)


if __name__ == "__main__":
    unittest.main()
