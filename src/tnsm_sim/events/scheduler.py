from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EventSchedule:
    failure_tick: int | None
    confirmation_tick: int | None
    targets: tuple[int, ...]

    def failure_due(self, tick: int) -> bool:
        return self.failure_tick is not None and tick == self.failure_tick

    def confirmation_due(self, tick: int) -> bool:
        return self.confirmation_tick is not None and tick == self.confirmation_tick

