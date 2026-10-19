from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationClock:
    t_start: float
    t_end: float
    dt: float

    def __post_init__(self) -> None:
        span = (self.t_end - self.t_start) / self.dt
        if abs(span - round(span)) > 1e-9:
            raise ValueError("simulation span must be an integer number of dynamics ticks")

    @property
    def final_tick(self) -> int:
        return int(round((self.t_end - self.t_start) / self.dt))

    def time(self, tick: int) -> float:
        return round(self.t_start + tick * self.dt, 12)

    def tick_of(self, time_value: float) -> int:
        raw = (time_value - self.t_start) / self.dt
        tick = int(round(raw))
        if abs(raw - tick) > 1e-9:
            raise ValueError(f"time {time_value} does not lie on the dynamics tick grid")
        return tick

    def period_ticks(self, period: float) -> int:
        raw = period / self.dt
        ticks = int(round(raw))
        if ticks <= 0 or abs(raw - ticks) > 1e-9:
            raise ValueError(f"period {period} is not a positive integer multiple of dt")
        return ticks

    def ticks(self):
        return range(self.final_tick + 1)

