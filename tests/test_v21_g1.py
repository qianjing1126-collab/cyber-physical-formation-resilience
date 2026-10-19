from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from tnsm_sim.experiments import run_v21_g1
from tnsm_sim.topology import articulation_points, graph_sha256, ring_edges


class V21G1Tests(unittest.TestCase):
    def test_graph_hash_is_canonical(self):
        first = [(1, 2), (3, 2), (2, 1)]
        second = [(2, 3), (2, 1)]
        self.assertEqual(graph_sha256(3, first), graph_sha256(3, second))

    def test_articulation_points(self):
        self.assertEqual(articulation_points(5, ring_edges(5)), ())
        self.assertEqual(articulation_points(4, ((1, 2), (2, 3), (3, 4))), (2, 3))

    def test_smoke_batch_writes_reproducible_tables_and_manifest(self):
        config = {
            "protocol": "experiment_protocol_v2.1",
            "gate": "V2.1-G1-smoke",
            "anchor": {
                "n": 6,
                "f": 1,
                "edge_counts": [9],
                "random_seed_start": 11,
                "random_seed_end": 13,
                "exact_edge_count": 9,
                "run_exact_enumeration": False,
                "exact_expected_topologies": 70,
                "run_f3_stress": False,
            },
            "scale_extension": {"enabled": False},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = run_v21_g1(config_path, root / "runs")
            self.assertTrue((output / "manifest.sha256").exists())
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["anchor_random"]["9"]["samples"], 3)
            with (output / "topology_records.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(sum(row["method"] == "B1_random_connected" for row in rows), 3)
            self.assertTrue(all(len(row["graph_sha256"]) == 64 for row in rows))


if __name__ == "__main__":
    unittest.main()
