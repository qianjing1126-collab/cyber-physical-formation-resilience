from __future__ import annotations

from dataclasses import replace

from tnsm_sim.types import NeighborSample, PacketRecord, VehicleState


class NeighborCache:
    def __init__(self) -> None:
        self._samples: dict[tuple[int, int], NeighborSample] = {}

    def update(self, receiver: int, sample: NeighborSample) -> None:
        key = (receiver, sample.source_id)
        previous = self._samples.get(key)
        if previous is None or sample.state_timestamp >= previous.state_timestamp:
            self._samples[key] = sample

    def get(self, receiver: int, sender: int) -> NeighborSample | None:
        return self._samples.get((receiver, sender))

    def samples_for(self, receiver: int, neighbor_ids: list[int]) -> list[NeighborSample]:
        return [sample for sender in neighbor_ids if (sample := self.get(receiver, sender)) is not None]


class IdealNetwork:
    """One-hop broadcast with deterministic delivery and explicit packet records."""

    def __init__(self, fixed_delay: float, packet_size: int, logical_edges: tuple[tuple[int, int], ...]):
        if fixed_delay < 0:
            raise ValueError("fixed_delay must be nonnegative")
        self.fixed_delay = fixed_delay
        self.packet_size = packet_size
        self.logical_edges = logical_edges
        self.cache = NeighborCache()
        self.records: list[PacketRecord] = []
        self._seq: dict[int, int] = {}
        self._pending: list[tuple[float, int, int, int, VehicleState]] = []

    def active_neighbors(self, vehicle_id: int, active_ids: set[int]) -> list[int]:
        neighbors: list[int] = []
        for i, j in self.logical_edges:
            if i == vehicle_id and j in active_ids:
                neighbors.append(j)
            elif j == vehicle_id and i in active_ids:
                neighbors.append(i)
        return sorted(neighbors)

    def broadcast(self, t: float, states: dict[int, VehicleState]) -> None:
        active_ids = {i for i, state in states.items() if state.active}
        for tx_id in sorted(active_ids):
            seq = self._seq.get(tx_id, 0) + 1
            self._seq[tx_id] = seq
            for rx_id in self.active_neighbors(tx_id, active_ids):
                arrival = round(t + self.fixed_delay, 12)
                self._pending.append((arrival, tx_id, rx_id, seq, replace(states[tx_id])))
        self._pending.sort(key=lambda item: (item[0], item[1], item[3], item[2]))
        self.deliver_until(t)

    def deliver_until(self, t: float) -> None:
        ready: list[tuple[float, int, int, int, VehicleState]] = []
        waiting: list[tuple[float, int, int, int, VehicleState]] = []
        for item in self._pending:
            (ready if item[0] <= t + 1e-12 else waiting).append(item)
        self._pending = waiting
        for arrival, tx_id, rx_id, seq, state in ready:
            sample = NeighborSample(
                source_id=tx_id,
                state_timestamp=state.t,
                receive_timestamp=arrival,
                s=state.s,
                d=state.d,
                v_s=state.v_s,
                v_d=state.v_d,
            )
            self.cache.update(rx_id, sample)
            self.records.append(PacketRecord(
                tx_id=tx_id,
                rx_id=rx_id,
                seq=seq,
                t_gen=state.t,
                t_tx=state.t,
                t_rx=arrival,
                status="received",
                drop_reason="",
                size=self.packet_size,
            ))

    def finalize(self, t_end: float) -> None:
        self.deliver_until(t_end)
