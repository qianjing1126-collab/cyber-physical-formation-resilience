from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from scripts.run_v21_g4_dynamic import laplacian
from tnsm_sim.weights.root_verification import spectral_abscissa


ROOT = Path(__file__).resolve().parents[1]


class V21G4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = json.loads((ROOT / "configs" / "v2_1_g4.json").read_text(encoding="utf-8"))
        g3 = json.loads((ROOT / config["g3_source"] / "summary.json").read_text(encoding="utf-8"))
        edges = [tuple(edge) for edge in g3["topology"]["edges"]]
        cls.config = config
        cls.matrices = {
            item["method"]: laplacian(8, edges, item["weights"])
            for item in g3["methods"]
        }

    def classify(self, method: str, delay: float) -> bool:
        eigenvalues = np.linalg.eigvalsh(self.matrices[method])[1:]
        abscissa = max(
            spectral_abscissa(
                delay,
                float(value),
                float(self.config["k_p"]),
                float(self.config["k_v"]),
                int(self.config["pade_order"]),
            )
            for value in eigenvalues
        )
        return abscissa < 0

    def test_realistic_delay_keeps_all_methods_stable(self):
        delay = float(self.config["realistic_delay_s"])
        self.assertTrue(all(self.classify(method, delay) for method in self.matrices))

    def test_stress_delay_separates_s2_from_baselines(self):
        delay = float(self.config["stress_delay_s"])
        self.assertFalse(self.classify("B7_uniform", delay))
        self.assertFalse(self.classify("B8_convergence_priority", delay))
        self.assertTrue(self.classify("S2_delay_constrained", delay))

    def test_delay_beyond_s2_margin_is_unstable(self):
        self.assertTrue(all(not self.classify(method, 0.2) for method in self.matrices))


if __name__ == "__main__":
    unittest.main()
