from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, hypot, pi, sin

import numpy as np


@dataclass(frozen=True)
class CircularPath:
    length: float
    radius: float

    def __post_init__(self) -> None:
        if self.length <= 0 or self.radius <= 0:
            raise ValueError("circular path length and radius must be positive")
        if self.length / self.radius >= pi:
            raise ValueError("G4 circular path must turn by less than pi radians")

    def world(self, s: float, d: float) -> tuple[float, float]:
        theta = s / self.radius
        return (
            (self.radius - d) * sin(theta),
            self.radius - (self.radius - d) * cos(theta),
        )

    def project(self, x: float, y: float) -> tuple[float, float]:
        theta = atan2(x, self.radius - y)
        theta = min(max(theta, 0.0), self.length / self.radius)
        rho = hypot(x, y - self.radius)
        return self.radius * theta, self.radius - rho

    def curvature(self, s: float) -> float:
        return 1.0 / self.radius

    def heading(self, s: float) -> float:
        return s / self.radius

    def inside(self, s: float) -> bool:
        return 0.0 <= s <= self.length


def _phi(u: np.ndarray | float) -> np.ndarray | float:
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def _dphi(u: np.ndarray | float) -> np.ndarray | float:
    return 30 * u**2 - 60 * u**3 + 30 * u**4


def _ddphi(u: np.ndarray | float) -> np.ndarray | float:
    return 60 * u - 180 * u**2 + 120 * u**3


@dataclass(frozen=True)
class SmoothTransitionPath:
    length: float
    transition_start: float
    transition_length: float
    lateral_shift: float
    samples: int = 4001
    _horizontal_span: float = field(init=False, repr=False)
    _x_grid: np.ndarray = field(init=False, repr=False, compare=False)
    _arc_grid: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.length <= 0 or self.transition_start < 0 or self.transition_length <= 0:
            raise ValueError("invalid smooth-transition path dimensions")
        if self.transition_start + self.transition_length >= self.length:
            raise ValueError("smooth transition must leave a final straight segment")
        if self.samples < 501:
            raise ValueError("smooth-transition arc table is undersampled")

        # Solve for the horizontal span whose quintic centerline has exactly
        # the configured arc length. This keeps s as an arc-length coordinate.
        low, high = 1e-6, self.transition_length
        for _ in range(80):
            span = 0.5 * (low + high)
            u = np.linspace(0.0, 1.0, self.samples)
            slope = self.lateral_shift / span * _dphi(u)
            x = span * u
            arc = float(np.trapezoid(np.sqrt(1.0 + slope**2), x))
            if arc < self.transition_length:
                low = span
            else:
                high = span
        span = 0.5 * (low + high)
        u = np.linspace(0.0, 1.0, self.samples)
        x = span * u
        slope = self.lateral_shift / span * _dphi(u)
        increments = 0.5 * (
            np.sqrt(1.0 + slope[:-1] ** 2) + np.sqrt(1.0 + slope[1:] ** 2)
        ) * np.diff(x)
        arc = np.concatenate(([0.0], np.cumsum(increments)))
        arc *= self.transition_length / arc[-1]
        object.__setattr__(self, "_horizontal_span", span)
        object.__setattr__(self, "_x_grid", x)
        object.__setattr__(self, "_arc_grid", arc)

    def _local_geometry(self, s: float) -> tuple[float, float, float, float]:
        local_arc = min(max(s - self.transition_start, 0.0), self.transition_length)
        x = float(np.interp(local_arc, self._arc_grid, self._x_grid))
        u = x / self._horizontal_span
        y = self.lateral_shift * float(_phi(u))
        slope = self.lateral_shift / self._horizontal_span * float(_dphi(u))
        second = self.lateral_shift / self._horizontal_span**2 * float(_ddphi(u))
        return x, y, slope, second

    def _centerline(self, s: float) -> tuple[float, float, float, float]:
        if s <= self.transition_start:
            return s, 0.0, 0.0, 0.0
        end = self.transition_start + self.transition_length
        if s >= end:
            x_end = self.transition_start + self._horizontal_span
            return x_end + s - end, self.lateral_shift, 0.0, 0.0
        x, y, slope, second = self._local_geometry(s)
        return self.transition_start + x, y, slope, second

    def world(self, s: float, d: float) -> tuple[float, float]:
        x, y, slope, _ = self._centerline(s)
        norm = hypot(1.0, slope)
        return x - d * slope / norm, y + d / norm

    def project(self, x: float, y: float) -> tuple[float, float]:
        # Dense one-dimensional nearest-point search is deterministic and
        # sufficiently accurate for initialization/audit; the controller uses
        # native path coordinates and does not project at every dynamics step.
        s_grid = np.linspace(0.0, self.length, self.samples)
        centers = np.asarray([self._centerline(float(s))[:2] for s in s_grid])
        index = int(np.argmin((centers[:, 0] - x) ** 2 + (centers[:, 1] - y) ** 2))
        lo = max(0, index - 2)
        hi = min(len(s_grid) - 1, index + 2)
        fine = np.linspace(float(s_grid[lo]), float(s_grid[hi]), 101)
        fine_centers = np.asarray([self._centerline(float(s))[:2] for s in fine])
        best = int(np.argmin((fine_centers[:, 0] - x) ** 2 + (fine_centers[:, 1] - y) ** 2))
        s = float(fine[best])
        cx, cy, slope, _ = self._centerline(s)
        norm = hypot(1.0, slope)
        d = (x - cx) * (-slope / norm) + (y - cy) / norm
        return s, d

    def curvature(self, s: float) -> float:
        _, _, slope, second = self._centerline(s)
        return second / (1.0 + slope**2) ** 1.5

    def heading(self, s: float) -> float:
        return atan2(self._centerline(s)[2], 1.0)

    def inside(self, s: float) -> bool:
        return 0.0 <= s <= self.length
