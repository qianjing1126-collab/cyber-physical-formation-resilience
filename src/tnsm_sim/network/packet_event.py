from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from typing import Any, Mapping

from tnsm_sim.network.ideal import NeighborCache
from tnsm_sim.seeds import SeedStreams
from tnsm_sim.types import NeighborSample, PacketRecord, VehicleState


@dataclass
class _Pending:
    tx_id: int
    rx_id: int
    seq: int
    t_gen: float
    t_tx: float
    state: VehicleState
    channel_delay: float | None = None
    channel_loss_probability: float | None = None


@dataclass
class _InFlight:
    pending: _Pending
    t_rx: float


class PacketEventNetwork:
    """Deterministic packet-event adapter for G1-W8.

    It separates generation, latest-sample access queue, transmission, loss,
    propagation and reception. Random values are keyed by packet identity so
    method iteration order cannot change the common network process.
    """

    def __init__(
        self,
        logical_edges: tuple[tuple[int, int], ...],
        packet_size: int,
        seeds: SeedStreams,
        access_delay: float,
        propagation_delay: float,
        jitter: float = 0.0,
        loss_probability: float = 0.0,
        forced_drop_sequences: tuple[int, ...] = (),
        delay_probabilities: tuple[float, ...] = (),
        delay_quantiles: tuple[float, ...] = (),
        two_state_parameters: Mapping[str, Any] | None = None,
    ) -> None:
        if min(access_delay, propagation_delay, jitter, loss_probability) < 0 or loss_probability > 1:
            raise ValueError("invalid packet-event network parameters")
        self.logical_edges = logical_edges
        self.packet_size = packet_size
        self.seeds = seeds
        self.access_delay = access_delay
        self.propagation_delay = propagation_delay
        self.jitter = jitter
        self.loss_probability = loss_probability
        self.forced_drop_sequences = set(forced_drop_sequences)
        if bool(delay_probabilities) != bool(delay_quantiles):
            raise ValueError("delay probabilities and quantiles must be supplied together")
        if delay_probabilities:
            if len(delay_probabilities) != len(delay_quantiles) or len(delay_probabilities) < 2:
                raise ValueError("invalid empirical delay CDF dimensions")
            if delay_probabilities[0] != 0.0 or delay_probabilities[-1] != 1.0:
                raise ValueError("empirical delay probabilities must span [0, 1]")
            if any(b <= a for a, b in zip(delay_probabilities, delay_probabilities[1:])):
                raise ValueError("empirical delay probabilities must be strictly increasing")
            if any(value < 0 for value in delay_quantiles) or any(
                b < a for a, b in zip(delay_quantiles, delay_quantiles[1:])
            ):
                raise ValueError("empirical delay quantiles must be nonnegative and nondecreasing")
        self.delay_probabilities = delay_probabilities
        self.delay_quantiles = delay_quantiles
        if two_state_parameters is not None and delay_probabilities:
            raise ValueError("two-state and unconditional empirical delay models are mutually exclusive")
        self.two_state_parameters = two_state_parameters
        self._two_state_current: dict[int, int] = {}
        if two_state_parameters is not None:
            if two_state_parameters.get("model") != "two_state_markov_shared_broadcast":
                raise ValueError("unsupported two-state network model")
            transition = two_state_parameters.get("transition_matrix", ())
            if len(transition) != 2 or any(len(row) != 2 for row in transition):
                raise ValueError("two-state transition_matrix must be 2x2")
            for row in transition:
                if any(float(value) < 0 for value in row) or abs(sum(float(value) for value in row) - 1.0) > 1e-9:
                    raise ValueError("two-state transition rows must be probabilities summing to one")
            initial = float(two_state_parameters.get("initial_contended_probability", -1.0))
            if not 0.0 <= initial <= 1.0:
                raise ValueError("invalid initial contended probability")
            for name in ("clear", "contended"):
                state = two_state_parameters.get("states", {}).get(name, {})
                probabilities = tuple(float(x) for x in state.get("delay_probabilities", ()))
                quantiles = tuple(float(x) for x in state.get("delay_quantiles_s", ()))
                loss = float(state.get("loss_probability", -1.0))
                if len(probabilities) != len(quantiles) or len(probabilities) < 2:
                    raise ValueError(f"invalid {name} delay CDF")
                if probabilities[0] != 0.0 or probabilities[-1] != 1.0:
                    raise ValueError(f"{name} delay probabilities must span [0, 1]")
                if any(b <= a for a, b in zip(probabilities, probabilities[1:])):
                    raise ValueError(f"{name} delay probabilities must be strictly increasing")
                if any(value < 0 for value in quantiles) or any(b < a for a, b in zip(quantiles, quantiles[1:])):
                    raise ValueError(f"invalid {name} delay quantiles")
                if not 0.0 <= loss <= 1.0:
                    raise ValueError(f"invalid {name} loss probability")
        self.cache = NeighborCache()
        self.records: list[PacketRecord] = []
        self._seq: dict[int, int] = {}
        self._pending: dict[tuple[int, int], _Pending] = {}
        self._in_flight: list[_InFlight] = []

    def active_neighbors(self, vehicle_id: int, active_ids: set[int]) -> list[int]:
        result = []
        for i, j in self.logical_edges:
            if i == vehicle_id and j in active_ids:
                result.append(j)
            elif j == vehicle_id and i in active_ids:
                result.append(i)
        return sorted(result)

    def _jitter(self, tx_id: int, rx_id: int, seq: int, role: str) -> float:
        if self.jitter == 0:
            return 0.0
        u = self.seeds.keyed_uniform("net", tx_id, rx_id, seq, role)
        return self.jitter * (2 * u - 1)

    def _empirical_delay(self, tx_id: int, rx_id: int, seq: int) -> float:
        u = self.seeds.keyed_uniform("net", tx_id, rx_id, seq, "empirical_delay")
        index = min(bisect_right(self.delay_probabilities, u), len(self.delay_probabilities) - 1)
        lower = index - 1
        p0, p1 = self.delay_probabilities[lower], self.delay_probabilities[index]
        q0, q1 = self.delay_quantiles[lower], self.delay_quantiles[index]
        return q0 + (u - p0) / (p1 - p0) * (q1 - q0)

    @staticmethod
    def _inverse_cdf(u: float, probabilities: tuple[float, ...], quantiles: tuple[float, ...]) -> float:
        index = min(bisect_right(probabilities, u), len(probabilities) - 1)
        lower = index - 1
        p0, p1 = probabilities[lower], probabilities[index]
        q0, q1 = quantiles[lower], quantiles[index]
        return q0 + (u - p0) / (p1 - p0) * (q1 - q0)

    def _two_state_broadcast(self, tx_id: int, seq: int) -> tuple[float, float]:
        assert self.two_state_parameters is not None
        if tx_id not in self._two_state_current:
            state = int(
                self.seeds.keyed_uniform("net", tx_id, "two_state_initial")
                < float(self.two_state_parameters["initial_contended_probability"])
            )
        else:
            state = self._two_state_current[tx_id]
        name = "contended" if state else "clear"
        parameters = self.two_state_parameters["states"][name]
        probabilities = tuple(float(x) for x in parameters["delay_probabilities"])
        quantiles = tuple(float(x) for x in parameters["delay_quantiles_s"])
        u_delay = self.seeds.keyed_uniform("net", tx_id, seq, "two_state_delay")
        delay = self._inverse_cdf(u_delay, probabilities, quantiles)
        transition = self.two_state_parameters["transition_matrix"]
        u_transition = self.seeds.keyed_uniform("net", tx_id, seq, "two_state_transition")
        self._two_state_current[tx_id] = int(u_transition < float(transition[state][1]))
        return delay, float(parameters["loss_probability"])

    def broadcast(self, t: float, states: dict[int, VehicleState]) -> None:
        active_ids = {i for i, state in states.items() if state.active}
        for tx_id in sorted(active_ids):
            seq = self._seq.get(tx_id, 0) + 1
            self._seq[tx_id] = seq
            shared_delay: float | None = None
            shared_loss: float | None = None
            if self.two_state_parameters is not None:
                shared_delay, shared_loss = self._two_state_broadcast(tx_id, seq)
            for rx_id in self.active_neighbors(tx_id, active_ids):
                key = (tx_id, rx_id)
                old = self._pending.get(key)
                if old is not None:
                    self.records.append(PacketRecord(
                        old.tx_id, old.rx_id, old.seq, old.t_gen, None, None,
                        "dropped", "queue_overwrite", self.packet_size,
                    ))
                delay = 0.0 if self.delay_probabilities else max(
                    0.0, self.access_delay + self._jitter(tx_id, rx_id, seq, "access")
                )
                self._pending[key] = _Pending(
                    tx_id, rx_id, seq, t, round(t + delay, 12), replace(states[tx_id]),
                    shared_delay, shared_loss,
                )
        self.deliver_until(t)

    def deliver_until(self, t: float) -> None:
        due_keys = sorted(
            (key for key, packet in self._pending.items() if packet.t_tx <= t + 1e-12),
            key=lambda key: (self._pending[key].t_tx, key[0], self._pending[key].seq, key[1]),
        )
        for key in due_keys:
            packet = self._pending.pop(key)
            forced = packet.seq in self.forced_drop_sequences
            loss_probability = self.loss_probability if packet.channel_loss_probability is None else packet.channel_loss_probability
            random_drop = self.seeds.keyed_uniform("net", packet.tx_id, packet.rx_id, packet.seq, "drop") < loss_probability
            if forced or random_drop:
                self.records.append(PacketRecord(
                    packet.tx_id, packet.rx_id, packet.seq, packet.t_gen, packet.t_tx, None,
                    "dropped", "forced_drop" if forced else "channel_loss", self.packet_size,
                ))
                continue
            if packet.channel_delay is not None:
                propagation = packet.channel_delay
            elif self.delay_probabilities:
                propagation = self._empirical_delay(packet.tx_id, packet.rx_id, packet.seq)
            else:
                propagation = max(
                    0.0,
                    self.propagation_delay + self._jitter(packet.tx_id, packet.rx_id, packet.seq, "propagation"),
                )
            self._in_flight.append(_InFlight(packet, round(packet.t_tx + propagation, 12)))

        ready = sorted(
            (item for item in self._in_flight if item.t_rx <= t + 1e-12),
            key=lambda item: (item.t_rx, item.pending.tx_id, item.pending.seq, item.pending.rx_id),
        )
        self._in_flight = [item for item in self._in_flight if item.t_rx > t + 1e-12]
        for item in ready:
            packet = item.pending
            self.cache.update(packet.rx_id, NeighborSample(
                source_id=packet.tx_id,
                state_timestamp=packet.t_gen,
                receive_timestamp=item.t_rx,
                s=packet.state.s, d=packet.state.d,
                v_s=packet.state.v_s, v_d=packet.state.v_d,
            ))
            self.records.append(PacketRecord(
                packet.tx_id, packet.rx_id, packet.seq, packet.t_gen, packet.t_tx, item.t_rx,
                "received", "", self.packet_size,
            ))

    def finalize(self, t_end: float) -> None:
        self.deliver_until(t_end)
        for packet in sorted(self._pending.values(), key=lambda p: (p.tx_id, p.seq, p.rx_id)):
            self.records.append(PacketRecord(
                packet.tx_id, packet.rx_id, packet.seq, packet.t_gen, None, None,
                "dropped", "simulation_end", self.packet_size,
            ))
        for item in sorted(self._in_flight, key=lambda x: (x.pending.tx_id, x.pending.seq, x.pending.rx_id)):
            packet = item.pending
            self.records.append(PacketRecord(
                packet.tx_id, packet.rx_id, packet.seq, packet.t_gen, packet.t_tx, None,
                "dropped", "simulation_end", self.packet_size,
            ))
        self._pending.clear()
        self._in_flight.clear()
