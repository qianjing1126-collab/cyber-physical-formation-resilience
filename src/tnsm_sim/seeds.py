from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class SeedStreams:
    base: int

    @property
    def graph(self) -> int:
        return self.base + 1000

    @property
    def net(self) -> int:
        return self.base + 2000

    @property
    def vehicle(self) -> int:
        return self.base + 3000

    @property
    def disturb(self) -> int:
        return self.base + 4000

    @property
    def fault(self) -> int:
        return self.base + 5000

    @property
    def stats(self) -> int:
        return self.base + 6000

    def keyed_uniform(self, stream: str, *keys: object) -> float:
        seed = getattr(self, stream)
        token = "|".join([str(seed), *(str(k) for k in keys)])
        integer = int.from_bytes(sha256(token.encode("utf-8")).digest()[:8], "big")
        return integer / float(2**64)

