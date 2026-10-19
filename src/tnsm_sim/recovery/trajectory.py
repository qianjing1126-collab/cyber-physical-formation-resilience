from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


C_V = 15.0 / 8.0
C_A = 10.0 * sqrt(3.0) / 3.0


def quintic_phi(xi: float) -> tuple[float, float, float]:
    x = min(1.0, max(0.0, xi))
    phi = 10 * x**3 - 15 * x**4 + 6 * x**5
    dphi = 30 * x**2 - 60 * x**3 + 30 * x**4
    ddphi = 60 * x - 180 * x**2 + 120 * x**3
    return phi, dphi, ddphi


def sufficient_duration(delta: float, v_max: float, a_max: float, minimum: float = 0.0) -> float:
    if delta < 0 or v_max <= 0 or a_max <= 0:
        raise ValueError("invalid recovery limits")
    return max(minimum, C_V * delta / v_max, sqrt(C_A * delta / a_max) if delta else 0.0)


@dataclass(frozen=True)
class QuinticOffset:
    t_start: float
    duration: float
    start: float
    target: float

    def sample(self, t: float) -> tuple[float, float, float]:
        if t <= self.t_start:
            return self.start, 0.0, 0.0
        if t >= self.t_start + self.duration:
            return self.target, 0.0, 0.0
        xi = (t - self.t_start) / self.duration
        phi, dphi, ddphi = quintic_phi(xi)
        delta = self.target - self.start
        return (
            self.start + phi * delta,
            dphi * delta / self.duration,
            ddphi * delta / self.duration**2,
        )


@dataclass(frozen=True)
class PiecewiseQuinticOffset:
    """A continuous sequence of rest-to-rest quintic offset segments."""

    segments: tuple[QuinticOffset, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("a piecewise trajectory needs at least one segment")
        previous = self.segments[0]
        for segment in self.segments[1:]:
            if segment.t_start + 1e-12 < previous.t_start + previous.duration:
                raise ValueError("piecewise trajectory segments overlap")
            if abs(segment.start - previous.target) > 1e-9:
                raise ValueError("piecewise trajectory is position-discontinuous")
            previous = segment

    @property
    def t_start(self) -> float:
        return self.segments[0].t_start

    @property
    def duration(self) -> float:
        return self.segments[-1].t_start + self.segments[-1].duration - self.t_start

    @property
    def start(self) -> float:
        return self.segments[0].start

    @property
    def target(self) -> float:
        return self.segments[-1].target

    def sample(self, t: float) -> tuple[float, float, float]:
        if t <= self.t_start:
            return self.start, 0.0, 0.0
        previous = self.segments[0]
        for segment in self.segments:
            if t <= segment.t_start + segment.duration:
                if t < segment.t_start:
                    return previous.target, 0.0, 0.0
                return segment.sample(t)
            previous = segment
        return self.target, 0.0, 0.0
