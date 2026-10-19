from __future__ import annotations

from hashlib import sha256
from pathlib import Path


REQUIRED_FILES = {
    "canonical_config.yaml", "environment.json", "status.json", "solver.json",
    "adjacency.csv", "weights.csv", "packets.csv", "states.csv", "controls.csv",
    "events.csv", "metrics_timeseries.csv", "metrics.json", "manifest.sha256",
}


def verify_manifest(run_dir: str | Path) -> dict[str, object]:
    run_dir = Path(run_dir).resolve()
    missing = sorted(name for name in REQUIRED_FILES if not (run_dir / name).exists())
    mismatches: list[str] = []
    listed: list[str] = []
    manifest = run_dir / "manifest.sha256"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, name = line.split("  ", 1)
            listed.append(name)
            path = run_dir / name
            if not path.exists() or sha256(path.read_bytes()).hexdigest() != expected:
                mismatches.append(name)
    return {
        "run_dir": str(run_dir),
        "valid": not missing and not mismatches,
        "missing": missing,
        "hash_mismatches": sorted(mismatches),
        "hashed_file_count": len(listed),
    }

