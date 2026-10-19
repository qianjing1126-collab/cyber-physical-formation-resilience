from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from tnsm_sim.network.ideal import NeighborCache
from tnsm_sim.types import ControlCommand, ReferenceState, VehicleState


def _clip(value: float, limit: float) -> tuple[float, bool]:
    clipped = max(-limit, min(limit, value))
    return clipped, abs(clipped - value) > 1e-12


@dataclass
class DistributedController:
    kp_s: float
    kv_s: float
    kp_d: float
    kv_d: float
    coupling_gain: float
    edge_weight: float
    a_x_max: float
    a_y_max: float

    def compute(
        self,
        t: float,
        state: VehicleState,
        reference: ReferenceState,
        neighbor_ids: list[int],
        cache: NeighborCache,
        state_at: Callable[[int, float], VehicleState],
        reference_at: Callable[[int, float], ReferenceState],
    ) -> ControlCommand:
        coupling_s = 0.0
        coupling_d = 0.0
        samples = cache.samples_for(state.vehicle_id, neighbor_ids)
        for sample in samples:
            own_past = state_at(state.vehicle_id, sample.state_timestamp)
            own_ref = reference_at(state.vehicle_id, sample.state_timestamp)
            neighbor_ref = reference_at(sample.source_id, sample.state_timestamp)
            err_s = (own_past.s - sample.s) - (own_ref.s - neighbor_ref.s)
            err_vs = (own_past.v_s - sample.v_s) - (own_ref.v_s - neighbor_ref.v_s)
            err_d = (own_past.d - sample.d) - (own_ref.d - neighbor_ref.d)
            err_vd = (own_past.v_d - sample.v_d) - (own_ref.v_d - neighbor_ref.v_d)
            coupling_s -= self.edge_weight * (self.kp_s * err_s + self.kv_s * err_vs)
            coupling_d -= self.edge_weight * (self.kp_d * err_d + self.kv_d * err_vd)
        gain = self.coupling_gain
        u_s_cmd = (
            reference.a_s
            + self.kp_s * (reference.s - state.s)
            + self.kv_s * (reference.v_s - state.v_s)
            + gain * coupling_s
        )
        u_d_cmd = (
            reference.a_d
            + self.kp_d * (reference.d - state.d)
            + self.kv_d * (reference.v_d - state.v_d)
            + gain * coupling_d
        )
        u_s, sat_s = _clip(u_s_cmd, self.a_x_max)
        u_d, sat_d = _clip(u_d_cmd, self.a_y_max)
        return ControlCommand(
            t=t,
            vehicle_id=state.vehicle_id,
            ref_s=reference.s,
            ref_d=reference.d,
            u_s_cmd=u_s_cmd,
            u_d_cmd=u_d_cmd,
            u_s_applied=u_s,
            u_d_applied=u_d,
            saturated=sat_s or sat_d,
            neighbor_count=len(samples),
        )
