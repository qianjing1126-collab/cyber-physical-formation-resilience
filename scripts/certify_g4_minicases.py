from __future__ import annotations

import csv
import json
from pathlib import Path

from tnsm_sim.io import verify_manifest


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "runs" / "raw"
OUTPUT = ROOT / "runs" / "g4" / "G4_minicase_certification_v2.json"

CASES = {
    "G4DEV2_minicase_p3_f1_single_B0_N8_seed0101": "B0",
    "G4DEV2_minicase_p3_f1_single_B9_N8_seed0101": "B9",
    "G4DEV2_minicase_p3_f1_single_B11_N8_seed0101": "B11",
    "G4DEV2_minicase_p3_f1_single_S3_N8_seed0101": "S3",
    "G4DEV2_minicase_p3_f2_nonadjacent_S3_N8_seed0101": "S3",
    "G4DEV2_minicase_p3_f3_adjacent_S3_N8_seed0101": "S3",
}

REQUIRED_METRICS = {
    "config_hash", "scientific_status", "recovery_method", "recovery_success",
    "Q_min", "R_Te", "R_rec", "T_r", "T_r_censored", "minimum_clearance",
    "peak_abs_ax", "peak_abs_ay", "path_violation", "safety_ok",
    "nominal_path_peak_abs_ay", "lateral_control_margin", "planner_time_s",
}


def main() -> None:
    records = []
    all_pass = True
    f1_network_signatures = set()
    for run_id, expected_method in CASES.items():
        run_dir = RAW / run_id
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        audit = verify_manifest(run_dir)
        with (run_dir / "events.csv").open(newline="", encoding="utf-8") as handle:
            events = list(csv.DictReader(handle))
        plan_events = [row for row in events if row["kind"] == "formation_recovery_planned"]
        payload_method = None
        if plan_events:
            payload_method = json.loads(plan_events[0]["payload"])["method"]
        checks = {
            "manifest_valid": bool(audit["valid"]),
            "required_metrics_present": REQUIRED_METRICS <= set(metrics),
            "method_identity": metrics["recovery_method"] == expected_method,
            "plan_event_semantics": (
                len(plan_events) == 0 if expected_method == "B0"
                else len(plan_events) == 1 and payload_method == expected_method
            ),
            "continuous_safety": bool(metrics["safety_ok"]),
            "clearance_limit": float(metrics["minimum_clearance"]) >= 0.5,
            "longitudinal_acceleration_limit": float(metrics["peak_abs_ax"]) <= 1.5,
            "lateral_acceleration_limit": float(metrics["peak_abs_ay"]) <= 1.5,
            "path_boundary": not bool(metrics["path_violation"]),
            "scientific_status_recorded": metrics["scientific_status"] in {
                "COMPLETE", "RECOVERY_FAILED", "SAFETY_VIOLATION"
            },
        }
        case_pass = all(checks.values())
        all_pass = all_pass and case_pass
        if "p3_f1_single" in run_id:
            f1_network_signatures.add((metrics["packet_count"], round(metrics["PDR"], 15), metrics["AoI_max"]))
        records.append({
            "run_id": run_id,
            "expected_method": expected_method,
            "checks": checks,
            "pass": case_pass,
            "scientific_status": metrics["scientific_status"],
            "recovery_success": metrics["recovery_success"],
            "minimum_clearance_m": metrics["minimum_clearance"],
            "peak_abs_ay_mps2": metrics["peak_abs_ay"],
            "R_Te": metrics["R_Te"],
            "T_r_s": metrics["T_r"],
        })
    paired_network_realization = len(f1_network_signatures) == 1
    all_pass = all_pass and paired_network_realization
    result = {
        "certification": "G4 deterministic mini-case gate",
        "protocol_version": "g4_recovery_safety_v1.2",
        "reportable_evidence": False,
        "case_count": len(records),
        "paired_f1_network_realization": paired_network_realization,
        "all_pass": all_pass,
        "cases": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
