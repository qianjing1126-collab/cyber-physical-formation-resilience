from __future__ import annotations

from pathlib import Path
import unittest

from tnsm_sim.experiments.g3_network import TraceRun, aggregate_metrics, fit_packet_event_adapter, synthesize_adapter_run


def _trace(seed: int = 1) -> TraceRun:
    metadata = {
        "probe_nodes": 2,
        "warmup_s": 0.0,
        "measurement_s": 0.3,
        "seed": seed,
        "regime": "light",
    }
    tx = tuple(
        {"uid": seq, "tx_id": 0, "seq": seq, "t_gen_s": 0.1 * (seq - 1), "size_bytes": 200}
        for seq in range(1, 4)
    )
    rx = tuple(
        {**row, "rx_id": 1, "t_rx_s": float(row["t_gen_s"]) + 0.01, "delay_s": 0.01}
        for row in tx
    )
    return TraceRun(Path("synthetic"), metadata, tx, rx)


class G3NetworkCalibrationTests(unittest.TestCase):
    def test_metrics_from_exact_trace(self):
        metrics = aggregate_metrics([_trace()], 0.01)
        self.assertEqual(metrics["expected_receptions"], 3)
        self.assertEqual(metrics["received_receptions"], 3)
        self.assertAlmostEqual(metrics["pdr"], 1.0)
        self.assertAlmostEqual(metrics["delay_p50_s"], 0.01)
        self.assertGreater(metrics["aoi_samples"], 0)

    def test_fit_and_synthesis_are_deterministic(self):
        trace = _trace(7)
        parameters = fit_packet_event_adapter([trace])
        self.assertAlmostEqual(parameters["loss_probability"], 0.0)
        self.assertAlmostEqual(parameters["access_delay_s"], 0.005)
        self.assertEqual(parameters["delay_model"], "piecewise_empirical_inverse_cdf")
        first = synthesize_adapter_run(trace, parameters)
        second = synthesize_adapter_run(trace, parameters)
        self.assertEqual(first.receptions, second.receptions)
        self.assertEqual(len(first.receptions), 3)


if __name__ == "__main__":
    unittest.main()
