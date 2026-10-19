from __future__ import annotations

import hashlib
import csv
import json
from pathlib import Path
from typing import Any, Iterable


class ArtifactStore:
    def __init__(self, root: Path, base_name: str):
        root.mkdir(parents=True, exist_ok=True)
        candidate = root / base_name
        retry = 0
        while candidate.exists():
            retry += 1
            candidate = root / f"{base_name}_retry{retry:02d}"
        candidate.mkdir()
        self.path = candidate

    def write_json(self, name: str, value: object) -> None:
        (self.path / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")

    def write_csv(self, name: str, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
        with (self.path / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def finalize(self) -> Path:
        lines = []
        for path in sorted(item for item in self.path.iterdir() if item.is_file()):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.name}")
        (self.path / "manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return self.path
