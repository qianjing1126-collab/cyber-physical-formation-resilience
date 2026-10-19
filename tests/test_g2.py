from __future__ import annotations

import unittest

from tnsm_sim.topology import certify_topology, circulant_regular_edges, greedy_edge_removal
from tnsm_sim.weights import empirical_delay_margin, modal_delay_margin, pade_delay_margin


class G2VerificationTests(unittest.TestCase):
    def test_stage1_certificate_enumerates_full_failure_set(self):
        edges = circulant_regular_edges(8, 3)
        certificate = certify_topology(8, edges, 2)
        self.assertEqual(certificate.scenario_count, 37)
        self.assertEqual(certificate.edge_count, 12)
        self.assertEqual(certificate.minimum_degree, 3)
        self.assertTrue(certificate.necessary_count_met)
        self.assertEqual(certificate.connectivity_rate, 1.0)
        self.assertGreater(certificate.worst_lambda2, 0.0)

    def test_robust_greedy_is_recertified(self):
        edges, certificate = greedy_edge_removal(8, 12, 2, robust=True)
        self.assertEqual(len(edges), 12)
        self.assertEqual(certificate.connectivity_rate, 1.0)
        self.assertGreater(certificate.worst_lambda2, 0.0)

    def test_pade_root_margin_matches_theory(self):
        theory = modal_delay_margin(4.0, 1.2, 2.0)
        numeric = pade_delay_margin(4.0, 1.2, 2.0)
        self.assertLess(abs(numeric - theory) / theory, 1e-5)

    def test_time_domain_margin_matches_theory(self):
        theory = modal_delay_margin(4.0, 1.2, 2.0)
        empirical = empirical_delay_margin(4.0, 1.2, 2.0, duration=20.0)
        self.assertLess(abs(empirical - theory) / theory, 0.02)


if __name__ == "__main__":
    unittest.main()
