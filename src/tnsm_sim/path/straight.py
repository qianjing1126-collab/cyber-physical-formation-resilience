from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StraightPath:
    length: float

    def world(self, s: float, d: float) -> tuple[float, float]:
        return s, d

    def project(self, x: float, y: float) -> tuple[float, float]:
        return x, y

    def curvature(self, s: float) -> float:
        return 0.0

    def heading(self, s: float) -> float:
        return 0.0

    def inside(self, s: float) -> bool:
        return 0.0 <= s <= self.length
