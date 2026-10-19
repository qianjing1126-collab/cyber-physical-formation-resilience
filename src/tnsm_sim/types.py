from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any


@dataclass
class VehicleState:
    t: float
    vehicle_id: int
    x: float
    y: float
    s: float
    d: float
    heading: float
    v: float
    v_s: float
    v_d: float
    a_x: float = 0.0
    a_y: float = 0.0
    active: bool = True
    slot_rank: int = 0
    coverage_width: float = 0.0
    recovery_speed_factor: float = 1.0
    recovery_acceleration_factor: float = 1.0

    def row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReferenceState:
    s: float
    d: float
    v_s: float
    v_d: float
    a_s: float
    a_d: float


@dataclass
class ControlCommand:
    t: float
    vehicle_id: int
    ref_s: float
    ref_d: float
    u_s_cmd: float
    u_d_cmd: float
    u_s_applied: float
    u_d_applied: float
    saturated: bool
    neighbor_count: int

    def row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PacketRecord:
    tx_id: int
    rx_id: int
    seq: int
    t_gen: float
    t_tx: float | None
    t_rx: float | None
    status: str
    drop_reason: str
    size: int

    def row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NeighborSample:
    source_id: int
    state_timestamp: float
    receive_timestamp: float
    s: float
    d: float
    v_s: float
    v_d: float


@dataclass(frozen=True)
class SimEvent:
    event_id: str
    kind: str
    target: str
    t_occurrence: float
    t_confirm: float | None = None
    t_effective: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        row = asdict(self)
        row["payload"] = json.dumps(self.payload, ensure_ascii=False, sort_keys=True)
        return row
