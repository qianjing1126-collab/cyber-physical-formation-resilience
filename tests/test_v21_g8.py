from __future__ import annotations

import unittest

from tnsm_sim.experiments.v21_g8 import (
    SAFE_HOLD_METHOD,
    architecture_path_interruption,
    communication_overhead,
    scale_failure_pairs,
    scale_topology,
)


class V21G8Tests(unittest.TestCase):
    def test_scale_topologies_have_minimum_degree_three_and_are_spectrally_normalized(self) -> None:
        for n in range(8, 17):
            edges, weights, lambda2, lambda_max = scale_topology(n, 4.0)
            degrees = {node: 0 for node in range(1, n + 1)}
            for first, second in edges:
                degrees[first] += 1
                degrees[second] += 1
            self.assertEqual(min(degrees.values()), 3)
            self.assertEqual(max(degrees.values()), 3 if n % 2 == 0 else 4)
            self.assertEqual(sum(value == 4 for value in degrees.values()), n % 2)
            self.assertEqual(len(weights), (3 * n + 1) // 2)
            self.assertGreater(lambda2, 0.0)
            self.assertAlmostEqual(lambda_max, 4.0, places=10)

    def test_failure_pairs_keep_adjacent_and_nonadjacent_roles(self) -> None:
        for n in range(8, 17):
            pairs = scale_failure_pairs(n)
            self.assertEqual(pairs["A_adjacent"][1] - pairs["A_adjacent"][0], 1)
            self.assertGreaterEqual(pairs["N_nonadjacent"][1] - pairs["N_nonadjacent"][0], 2)

    def test_centralized_air_bytes_are_twice_distributed_broadcast_bytes(self) -> None:
        gaps = []
        for n in range(8, 17):
            rows = communication_overhead(
                n, duration_s=30.0, failure_time_s=15.0,
                message_period_s=0.1, packet_size_byte=200,
            )
            values = {row["method"]: row["wireless_bytes"] for row in rows}
            self.assertEqual(values["B11"], 2 * values["S3"])
            self.assertEqual(values["B10"], values["B11"])
            gaps.append(values["B11"] - values["S3"])
        self.assertTrue(all(second > first for first, second in zip(gaps, gaps[1:])))

    def test_path_interruption_uses_safe_hold_not_b9_fallback(self) -> None:
        means = {"B9": 0.60, "S3": 0.80, SAFE_HOLD_METHOD: 0.40}
        for architecture in ("B10", "B11"):
            result = architecture_path_interruption(means, architecture)
            self.assertEqual(result["path_interruption_R_Te"], 0.40)
            self.assertEqual(result["path_interruption_retention_percent"], 50.0)
            self.assertNotEqual(result["path_interruption_R_Te"], means["B9"])
        s3 = architecture_path_interruption(means, "S3")
        self.assertEqual(s3["path_interruption_R_Te"], means["S3"])
        self.assertEqual(s3["path_interruption_retention_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
