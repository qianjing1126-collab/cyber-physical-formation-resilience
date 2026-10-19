from __future__ import annotations

import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any


class RunStore:
    def __init__(self, project_root: Path, base_run_id: str):
        self.project_root = project_root.resolve()
        self.raw_root = self.project_root / "runs" / "raw"
        self.staging_root = self.project_root / "runs" / "staging"
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.retry = 0
        run_id = base_run_id
        while (self.raw_root / run_id).exists() or (self.staging_root / f"{run_id}.tmp").exists():
            self.retry += 1
            run_id = f"{base_run_id}_retry{self.retry:02d}"
        self.run_id = run_id
        self.staging = self.staging_root / f"{run_id}.tmp"
        self.final = self.raw_root / run_id
        self.staging.mkdir()
        self._finalized = False

    def write_json(self, name: str, data: Any) -> None:
        (self.staging / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    def write_text(self, name: str, text: str) -> None:
        (self.staging / name).write_text(text, encoding="utf-8")

    def write_csv(self, name: str, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
        if fieldnames is None:
            fieldnames = list(rows[0]) if rows else []
        with (self.staging / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            if fieldnames:
                writer.writeheader()
                writer.writerows(rows)

    def write_environment(self) -> None:
        source_files = sorted((self.project_root / "src").rglob("*.py"))
        source_digest = sha256()
        for path in source_files:
            source_digest.update(path.relative_to(self.project_root).as_posix().encode("utf-8"))
            source_digest.update(path.read_bytes())
        self.write_json("environment.json", {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "pid": os.getpid(),
            "threads": 1,
            "source_tree_sha256": source_digest.hexdigest(),
            "source_file_count": len(source_files),
            "git_commit": None,
        })

    def finalize(self, status: dict[str, Any]) -> Path:
        if self._finalized:
            raise RuntimeError("run already finalized")
        self.write_json("status.json", {**status, "run_id": self.run_id, "retry": self.retry})
        entries: list[str] = []
        for path in sorted(self.staging.iterdir(), key=lambda p: p.name):
            if path.name == "manifest.sha256" or not path.is_file():
                continue
            digest = sha256(path.read_bytes()).hexdigest()
            entries.append(f"{digest}  {path.name}")
        self.write_text("manifest.sha256", "\n".join(entries) + "\n")
        self.staging.replace(self.final)
        self._finalized = True
        return self.final
