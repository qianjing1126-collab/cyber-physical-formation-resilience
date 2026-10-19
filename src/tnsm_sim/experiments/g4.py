from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from tnsm_sim.config import load_config
from tnsm_sim.io import verify_manifest
from tnsm_sim.runner import build_run_id, run_simulation


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_VERSION = "g4_recovery_safety_v1.2"
SEEDS = tuple(range(101, 131))

PATHS = {
    "P1": {"path_id": "P1", "length": 220.0},
    "P2": {"path_id": "P2", "length": 220.0, "radius": 80.0},
    "P3": {
        "path_id": "P3", "length": 220.0, "transition_start": 40.0,
        "transition_length": 80.0, "lateral_shift": 12.0,
    },
}

LAYOUTS = {
    "F1": {"layout_id": "F1"},
    "F2": {
        "layout_id": "F2",
        "longitudinal_offsets": [-0.25, 0.25, -0.25, 0.25, -0.25, 0.25, -0.25, 0.25],
    },
    "F3": {"layout_id": "F3", "heterogeneity_min": 0.9, "heterogeneity_max": 1.1},
}

FAULTS = {
    "P1": {"type": "P-1", "targets": [4], "t_d": 5.0, "t_c": 5.5},
    "P2A": {"type": "P-2A", "targets": [4, 5], "t_d": 5.0, "t_c": 5.5},
    "P2N": {"type": "P-2N", "targets": [3, 6], "t_d": 5.0, "t_c": 5.5},
}


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_frozen(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"frozen G4 configuration changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _configuration(
    template: dict[str, Any], module: str, path_id: str, layout_id: str,
    fault_id: str, method: str, seed: int,
) -> dict[str, Any]:
    cfg = deepcopy(template)
    scenario = f"{module.lower()}_{path_id}_{layout_id}_{fault_id}"
    cfg["meta"].update({
        "protocol_version": PROTOCOL_VERSION,
        "module": "G4R2",
        "scenario": scenario,
        "method": method,
        "base_seed": seed,
    })
    cfg["path"] = deepcopy(PATHS[path_id])
    formation = cfg["formation"]
    for key in ("longitudinal_offsets", "heterogeneity_min", "heterogeneity_max"):
        formation.pop(key, None)
    formation.update(deepcopy(LAYOUTS[layout_id]))
    cfg["failure"] = deepcopy(FAULTS[fault_id])
    cfg["recovery"]["method"] = method
    cfg["recovery"]["enabled"] = method != "B0"
    # The admitted G3-B3 parameters are embedded so generated reportable
    # configurations do not depend on a relocated profile-file reference.
    cfg["network"].pop("two_state_profile_file", None)
    cfg["network"].pop("two_state_regime", None)
    return cfg


def build_g4_matrix(project_root: Path = PROJECT_ROOT) -> Path:
    project_root = project_root.resolve()
    base = load_config(project_root / "configs" / "scenarios" / "g1_mce1_g3b3_moderate.yaml")
    template = json.loads(base.canonical_json)
    config_root = project_root / "runs" / "g4" / "configs" / "reportable_v2"
    rows: list[dict[str, Any]] = []

    designs = []
    for path_id in ("P1", "P2", "P3"):
        for method in ("B0", "B9", "B11", "S3"):
            designs.append(("P04", path_id, "F1", "P1", method))
    for layout_id in ("F1", "F2", "F3"):
        for fault_id in ("P2A", "P2N"):
            for method in ("B9", "B11", "S3"):
                designs.append(("P05", "P3", layout_id, fault_id, method))

    for module, path_id, layout_id, fault_id, method in designs:
        for seed in SEEDS:
            cfg = _configuration(template, module, path_id, layout_id, fault_id, method, seed)
            relative = Path(module) / f"{path_id}_{layout_id}_{fault_id}_{method}_seed{seed:04d}.json"
            config_path = config_root / relative
            text = json.dumps(cfg, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
            _write_frozen(config_path, text)
            frozen = load_config(config_path)
            rows.append({
                "module": module,
                "path": path_id,
                "layout": layout_id,
                "fault": fault_id,
                "targets": list(FAULTS[fault_id]["targets"]),
                "method": method,
                "seed": seed,
                "config_path": str(config_path),
                "config_hash": frozen.config_hash,
                "run_id": build_run_id(frozen),
            })

    if len(rows) != 900 or len({row["run_id"] for row in rows}) != 900:
        raise RuntimeError("G4 matrix must contain exactly 900 unique runs")
    counts = {
        "P04": sum(row["module"] == "P04" for row in rows),
        "P05": sum(row["module"] == "P05" for row in rows),
    }
    if counts != {"P04": 360, "P05": 540}:
        raise RuntimeError(f"unexpected G4 module counts: {counts}")
    matrix_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "paired_seeds": list(SEEDS),
        "counts": {**counts, "total": len(rows)},
        "rows_sha256": sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "rows": rows,
    }
    index_path = project_root / "runs" / "g4" / "G4_reportable_matrix_v2.json"
    if index_path.exists():
        prior = json.loads(index_path.read_text(encoding="utf-8"))
        if prior != matrix_payload:
            raise RuntimeError("frozen G4 matrix index changed")
    else:
        _atomic_json(index_path, matrix_payload)
    return index_path


def _run_one(row: dict[str, Any], project_root: str) -> dict[str, Any]:
    root = Path(project_root)
    expected = root / "runs" / "raw" / row["run_id"]
    if expected.exists():
        audit = verify_manifest(expected)
        if not audit["valid"]:
            raise RuntimeError(f"existing reportable run failed its manifest audit: {expected}")
        metrics = json.loads((expected / "metrics.json").read_text(encoding="utf-8"))
        if metrics["config_hash"] != row["config_hash"]:
            raise RuntimeError(f"existing reportable run has a different config hash: {expected}")
        return {"run_id": row["run_id"], "state": "SKIPPED_VALID", "scientific_status": metrics["scientific_status"]}
    config = load_config(row["config_path"])
    run_dir, metrics = run_simulation(config, root)
    if run_dir != expected:
        raise RuntimeError(f"unexpected retry directory for reportable run: {run_dir}")
    audit = verify_manifest(run_dir)
    if not audit["valid"]:
        raise RuntimeError(f"new reportable run failed its manifest audit: {run_dir}")
    return {"run_id": row["run_id"], "state": "COMPLETE", "scientific_status": metrics["scientific_status"]}


def run_g4_batch(index_path: Path, workers: int = 4, project_root: Path = PROJECT_ROOT) -> Path:
    project_root = project_root.resolve()
    matrix = json.loads(index_path.resolve().read_text(encoding="utf-8"))
    rows = matrix["rows"]
    progress_path = project_root / "runs" / "g4" / "G4_batch_progress_v2.json"
    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, row, str(project_root)): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                results[row["run_id"]] = future.result()
            except Exception as exc:  # preserve every process/config failure
                failures[row["run_id"]] = f"{type(exc).__name__}: {exc}"
            _atomic_json(progress_path, {
                "protocol_version": PROTOCOL_VERSION,
                "expected": len(rows),
                "finished": len(results) + len(failures),
                "valid": len(results),
                "failed_processes": len(failures),
                "failures": failures,
            })
    if failures:
        raise RuntimeError(f"{len(failures)} G4 batch runs failed; inspect {progress_path}")
    _atomic_json(progress_path, {
        "protocol_version": PROTOCOL_VERSION,
        "expected": len(rows),
        "finished": len(results),
        "valid": len(results),
        "failed_processes": 0,
        "complete": True,
        "scientific_status_counts": {
            status: sum(item["scientific_status"] == status for item in results.values())
            for status in sorted({item["scientific_status"] for item in results.values()})
        },
    })
    return progress_path
