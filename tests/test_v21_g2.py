from __future__ import annotations

from collections import Counter
import unittest

from tnsm_sim.experiments.v21_g2 import balanced_failure_pairs, network_service_metrics


class V21G2Tests(unittest.TestCase):
    def test_failure_pairs_are_balanced_and_deterministic(self):
        first = balanced_failure_pairs()
        second = balanced_failure_pairs()
        self.assertEqual(first, second)
        counts = Counter(first)
        self.assertEqual(len(first), 100)
        self.assertEqual(len(counts), 28)
        self.assertEqual(set(counts.values()), {3, 4})
        self.assertEqual(sum(value == 4 for value in counts.values()), 16)

    def test_network_service_metrics_detect_disconnection(self):
        connected = network_service_metrics(4, ((1, 2), (2, 3), (3, 4), (1, 4)), ())
        disconnected = network_service_metrics(4, ((1, 2), (3, 4)), ())
        self.assertEqual(connected["e2e_reachability"], 1.0)
        self.assertEqual(connected["component_count"], 1)
        self.assertLess(disconnected["e2e_reachability"], 1.0)
        self.assertEqual(disconnected["component_count"], 2)


if __name__ == "__main__":
    unittest.main()
