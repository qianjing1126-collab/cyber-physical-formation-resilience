from __future__ import annotations

from pathlib import Path
import unittest

from tnsm_sim.experiments.g3_network import TraceRun
from tnsm_sim.experiments.g3_network_b3 import per_seed_contention_rate_std, probe_interarrival_metrics


def _run(seed: int, times: list[float]) -> TraceRun:
    tx = tuple({
        "uid": index,
        "tx_id": 0,
        "seq": index,
        "t_gen_s": time,
        "size_bytes": 200,
    } for index, time in enumerate(times))
    rx = tuple({
        "uid": index,
        "tx_id": 0,
        "rx_id": 1,
        "seq": index,
        "t_gen_s": time,
        "t_rx_s": time + 0.001,
        "delay_s": 0.001,
        "size_bytes": 200,
    } for index, time in enumerate(times))
    return TraceRun(Path("synthetic"), {"seed": seed, "probe_nodes": 2}, tx, rx)


class G3B3AuditTests(unittest.TestCase):
    def test_interarrival_mean_is_audited_per_sender(self) -> None:
        metrics = probe_interarrival_metrics([_run(1, [0.0, 0.09, 0.20, 0.30])], 0.1)
        self.assertEqual(metrics["samples"], 3)
        self.assertAlmostEqual(float(metrics["mean_s"]), 0.1)
        self.assertAlmostEqual(float(metrics["mean_relative_error"]), 0.0)

    def test_per_seed_contention_std_is_zero_for_identical_clear_runs(self) -> None:
        std, rows = per_seed_contention_rate_std(
            [_run(1, [0.0, 0.1]), _run(2, [0.0, 0.1])], 0.01
        )
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(std, 0.0)


if __name__ == "__main__":
    unittest.main()
