from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

from tnsm_sim.config import load_config
from tnsm_sim.runner import run_simulation


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class MinimumClosedLoopEvidenceTests(unittest.TestCase):
    def test_mce0_nominal_determinism(self):
        config = load_config(ROOT / "configs" / "scenarios" / "g1_mce0.yaml")
        with tempfile.TemporaryDirectory() as temp:
            run1, metrics1 = run_simulation(config, Path(temp))
            run2, metrics2 = run_simulation(config, Path(temp))
            self.assertGreaterEqual(metrics1["Q_min_mission"], 0.99)
            self.assertTrue(metrics1["safety_ok"])
            for name in ("states.csv", "controls.csv", "packets.csv", "metrics_timeseries.csv"):
                self.assertEqual(digest(run1 / name), digest(run2 / name), name)
            self.assertEqual(metrics1["Q_min_mission"], metrics2["Q_min_mission"])

    def test_mce1_single_dropout_recovery(self):
        config = load_config(ROOT / "configs" / "scenarios" / "g1_mce1.yaml")
        with tempfile.TemporaryDirectory() as temp:
            run_dir, metrics = run_simulation(config, Path(temp))
            self.assertTrue(metrics["safety_ok"])
            self.assertTrue(metrics["recovery_success"])
            self.assertFalse(metrics["T_r_censored"])
            self.assertGreater(metrics["Q_cap_plus"], 0.0)
            self.assertGreaterEqual(metrics["minimum_clearance"], 0.5)
            self.assertTrue((run_dir / "events.csv").exists())

    def test_mce2_packet_event_closed_loop(self):
        config = load_config(ROOT / "configs" / "scenarios" / "g1_mce2.yaml")
        with tempfile.TemporaryDirectory() as temp:
            _, metrics = run_simulation(config, Path(temp))
            self.assertGreaterEqual(metrics["Q_min_mission"], 0.99)
            self.assertLess(metrics["PDR"], 1.0)
            self.assertGreater(metrics["AoI_max"], 0.1)
            self.assertTrue(metrics["safety_ok"])


if __name__ == "__main__":
    unittest.main()
