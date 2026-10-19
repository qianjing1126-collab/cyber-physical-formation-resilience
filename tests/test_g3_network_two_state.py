from __future__ import annotations

from pathlib import Path
import unittest

from tnsm_sim.experiments.g3_network import TraceRun
from tnsm_sim.experiments.g3_network_two_state import (
    broadcast_events,
    event_state,
    fit_two_state_adapter,
    structural_metrics,
    synthesize_two_state_run,
)


def _config():
    return {
        "state_definition": {
            "minimum_tail_offset_s": 0.0001,
            "transition_smoothing": 1.0,
            "loss_smoothing": 0.5,
        },
        "delay_model": {"probabilities": [0.0, 0.5, 1.0]},
    }


def _trace(seed: int = 1) -> TraceRun:
    metadata = {"probe_nodes": 3, "seed": seed, "regime": "moderate", "warmup_s": 0.0, "measurement_s": 0.4}
    tx = tuple(
        {"uid": seq, "tx_id": 0, "seq": seq, "t_gen_s": 0.1 * seq, "size_bytes": 200}
        for seq in range(1, 5)
    )
    rx = []
    for tx_row in tx:
        delay = 0.003 if int(tx_row["seq"]) in {2, 3} else 0.001
        for receiver in (1, 2):
            rx.append({
                **tx_row, "rx_id": receiver,
                "t_rx_s": float(tx_row["t_gen_s"]) + delay,
                "delay_s": delay,
            })
    return TraceRun(Path("synthetic"), metadata, tx, tuple(rx))


class G3B2Tests(unittest.TestCase):
    def test_event_classification_and_structure(self):
        trace = _trace()
        events = broadcast_events(trace)
        states = [event_state(event, 0.002) for event in events]
        self.assertEqual(states, [0, 1, 1, 0])
        metrics = structural_metrics([trace], 0.002)
        self.assertAlmostEqual(metrics["contention_event_rate"], 0.5)
        self.assertAlmostEqual(metrics["state_persistence"], 1.0 / 3.0)

    def test_fit_and_synthesis_are_deterministic(self):
        trace = _trace(7)
        parameters = fit_two_state_adapter([trace], _config())
        self.assertEqual(parameters["model"], "two_state_markov_shared_broadcast")
        self.assertEqual(len(parameters["transition_matrix"]), 2)
        first = synthesize_two_state_run(trace, parameters)
        second = synthesize_two_state_run(trace, parameters)
        self.assertEqual(first.receptions, second.receptions)


if __name__ == "__main__":
    unittest.main()
