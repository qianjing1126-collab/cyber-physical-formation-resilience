from __future__ import annotations

from tnsm_sim.types import NeighborSample


def age_of_information(t: float, sample: NeighborSample | None) -> float | None:
    if sample is None:
        return None
    return max(0.0, t - sample.state_timestamp)

