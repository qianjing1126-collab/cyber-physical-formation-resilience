from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tnsm_sim.config.loader import load_config
from tnsm_sim.io import RunStore, verify_manifest
from tnsm_sim.metrics.coverage import interval_measure
from tnsm_sim.metrics.mission import service_values
from tnsm_sim.metrics.resilience import fixed_window_area, recovery_time
from tnsm_sim.network.aoi import age_of_information
from tnsm_sim.network.packet_event import PacketEventNetwork
from tnsm_sim.recovery.safety import pair_clearance
from tnsm_sim.recovery.stage3 import compress_ranks
from tnsm_sim.recovery.trajectory import QuinticOffset, quintic_phi
from tnsm_sim.seeds import SeedStreams
from tnsm_sim.topology.graph import algebraic_connectivity, residual_edges, ring_edges
from tnsm_sim.types import NeighborSample, VehicleState
from tnsm_sim.weights.delay_margin import modal_delay_margin


ROOT = Path(__file__).resolve().parents[1]


def state(vehicle_id: int, s: float, d: float) -> VehicleState:
    return VehicleState(0.0, vehicle_id, s, d, s, d, 0.0, 8.0, 8.0, 0.0, active=True, slot_rank=vehicle_id)


class FormulaAndInfrastructureTests(unittest.TestCase):
    def test_u01_interval_union(self):
        self.assertAlmostEqual(interval_measure([(-2, 2), (1, 5), (8, 9)]), 8.0)
        self.assertEqual(interval_measure([]), 0.0)

    def test_u02_q_qcap_qrec(self):
        q, qrec = service_values(8.0, 5.0, 10.0, 10.0, 0.4)
        self.assertAlmostEqual(q, 0.4)
        self.assertAlmostEqual(qrec, 1.0)
        q, qrec = service_values(0.0, 0.0, 10.0, 10.0, 0.0)
        self.assertEqual((q, qrec), (0.0, 0.0))

    def test_u03_recovery_holding_gate(self):
        samples = [(i * 0.5, 0.96 if 2 <= i <= 3 or i >= 6 else 0.8) for i in range(11)]
        tr, censored = recovery_time(samples, 0.0, 5.0, 1.0, 0.95)
        self.assertFalse(censored)
        self.assertEqual(tr, 3.0)
        tr, censored = recovery_time([(i * 0.5, 0.8) for i in range(11)], 0.0, 5.0, 1.0, 0.95)
        self.assertTrue(censored)
        self.assertIsNone(tr)

    def test_u04_fixed_window_area(self):
        samples = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
        self.assertAlmostEqual(fixed_window_area(samples, 0.0, 1.0), 0.5)

    def test_u05_aoi(self):
        sample = NeighborSample(2, 1.0, 1.2, 0, 0, 0, 0)
        self.assertAlmostEqual(age_of_information(1.5, sample), 0.5)
        self.assertIsNone(age_of_information(1.5, None))

    def test_u05_packet_event_hand_trace_and_overwrite(self):
        states = {1: state(1, 0, 0), 2: state(2, 0, 3.6)}
        net = PacketEventNetwork(((1, 2),), 200, SeedStreams(1), 0.0, 0.1,
                                 forced_drop_sequences=(2,))
        net.broadcast(0.0, states)
        net.deliver_until(0.1)
        self.assertAlmostEqual(age_of_information(0.1, net.cache.get(1, 2)), 0.1)
        net.broadcast(0.1, states)  # sequence 2 is forced to drop
        net.deliver_until(0.2)
        self.assertAlmostEqual(age_of_information(0.2, net.cache.get(1, 2)), 0.2)
        net.broadcast(0.2, states)
        net.deliver_until(0.3)
        self.assertAlmostEqual(age_of_information(0.3, net.cache.get(1, 2)), 0.1)

        queued = PacketEventNetwork(((1, 2),), 200, SeedStreams(1), 0.15, 0.0)
        queued.broadcast(0.0, states)
        queued.broadcast(0.1, states)
        self.assertTrue(any(r.drop_reason == "queue_overwrite" for r in queued.records))

    def test_u05_empirical_delay_cdf(self):
        net = PacketEventNetwork(
            ((1, 2),), 200, SeedStreams(17), 0.0, 0.0,
            delay_probabilities=(0.0, 0.5, 1.0),
            delay_quantiles=(0.001, 0.003, 0.009),
        )
        value = net._empirical_delay(1, 2, 1)
        self.assertGreaterEqual(value, 0.001)
        self.assertLessEqual(value, 0.009)

    def test_u05_two_state_delay_is_shared_per_broadcast(self):
        parameters = {
            "model": "two_state_markov_shared_broadcast",
            "initial_contended_probability": 0.0,
            "transition_matrix": [[1.0, 0.0], [0.0, 1.0]],
            "states": {
                "clear": {
                    "loss_probability": 0.0,
                    "delay_probabilities": [0.0, 1.0],
                    "delay_quantiles_s": [0.001, 0.003],
                },
                "contended": {
                    "loss_probability": 1.0,
                    "delay_probabilities": [0.0, 1.0],
                    "delay_quantiles_s": [0.010, 0.020],
                },
            },
        }
        states = {i: state(i, 0, 3.6 * i) for i in (1, 2, 3)}
        net = PacketEventNetwork(
            ((1, 2), (1, 3)), 200, SeedStreams(19), 0.0, 0.0,
            two_state_parameters=parameters,
        )
        net.broadcast(0.0, states)
        net.deliver_until(0.1)
        delays = [record.t_rx - record.t_tx for record in net.records if record.tx_id == 1 and record.status == "received"]
        self.assertEqual(len(delays), 2)
        self.assertAlmostEqual(delays[0], delays[1])

    def test_u06_graph_laplacian_and_residual(self):
        edges = ring_edges(8)
        self.assertGreater(algebraic_connectivity(list(range(1, 9)), edges), 0.0)
        active = set(range(2, 9))
        self.assertGreater(algebraic_connectivity(sorted(active), residual_edges(edges, active)), 0.0)

    def test_u07_delay_margin_monotonicity(self):
        low = modal_delay_margin(1.0, 1.2, 2.0)
        high = modal_delay_margin(4.0, 1.2, 2.0)
        self.assertGreater(low, high)

    def test_u08_rank_compression(self):
        ranks = {i: i for i in range(1, 9)}
        self.assertEqual(compress_ranks(ranks, {4})[5], 4)
        result = compress_ranks(ranks, {3, 6})
        self.assertEqual(sorted(result.values()), list(range(1, 7)))

    def test_u09_quintic_endpoints(self):
        self.assertEqual(quintic_phi(0.0), (0.0, 0.0, 0.0))
        phi, dphi, ddphi = quintic_phi(1.0)
        self.assertAlmostEqual(phi, 1.0)
        self.assertAlmostEqual(dphi, 0.0)
        self.assertAlmostEqual(ddphi, 0.0)
        traj = QuinticOffset(2.0, 4.0, -1.0, 3.0)
        self.assertEqual(traj.sample(2.0), (-1.0, 0.0, 0.0))
        self.assertEqual(traj.sample(6.0), (3.0, 0.0, 0.0))

    def test_u10_safety_clearance(self):
        self.assertAlmostEqual(pair_clearance(state(1, 0, 0), state(2, 0, 3.5), 8.0, 3.0), 0.5)
        self.assertLess(pair_clearance(state(1, 0, 0), state(2, 0, 2.5), 8.0, 3.0), 0.0)

    def test_u11_seed_substreams(self):
        a = SeedStreams(7)
        b = SeedStreams(7)
        self.assertEqual(a.net, 2007)
        self.assertEqual(a.keyed_uniform("net", 1, 2, 3), b.keyed_uniform("net", 1, 2, 3))
        self.assertNotEqual(a.keyed_uniform("net", 1), a.keyed_uniform("graph", 1))

    def test_u12_run_store_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = RunStore(root, "run")
            first.write_json("metrics.json", {"x": 1})
            first_dir = first.finalize({"run_state": "COMPLETE"})
            second = RunStore(root, "run")
            second.write_json("metrics.json", {"x": 1})
            second_dir = second.finalize({"run_state": "COMPLETE"})
            self.assertEqual(first_dir.name, "run")
            self.assertEqual(second_dir.name, "run_retry01")
            self.assertTrue((first_dir / "manifest.sha256").exists())
            audit = verify_manifest(first_dir)
            self.assertEqual(audit["hash_mismatches"], [])
            self.assertIn("states.csv", audit["missing"])

    def test_configuration_freeze(self):
        config = load_config(ROOT / "configs" / "scenarios" / "g1_mce0.yaml")
        self.assertEqual(config.data["meta"]["scenario"], "mce0_no_fault")
        self.assertEqual(len(config.config_hash), 64)
        with self.assertRaises(TypeError):
            config.data["meta"]["scenario"] = "changed"


if __name__ == "__main__":
    unittest.main()
