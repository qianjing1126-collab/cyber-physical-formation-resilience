from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


MORANDI = ["#5F766D", "#A77C6B", "#7D8EA3", "#B39B6B", "#7E6D82", "#6F8F8B", "#B07C61", "#8A947B"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return default if value in (None, "") else float(value)


def generate_g1_overview(run_dir: str | Path, *, show: bool = False) -> list[Path]:
    """Create read-only diagnostic figures from an already completed raw run."""
    run_dir = Path(run_dir).resolve()
    required = ["states.csv", "metrics_timeseries.csv", "events.csv", "metrics.json", "canonical_config.yaml"]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise ValueError(f"run directory is missing required files: {missing}")

    states = _read_csv(run_dir / "states.csv")
    metrics = _read_csv(run_dir / "metrics_timeseries.csv")
    events = _read_csv(run_dir / "events.csv")
    summary: dict[str, Any] = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config: dict[str, Any] = json.loads((run_dir / "canonical_config.yaml").read_text(encoding="utf-8"))

    project_root = run_dir.parents[2]
    out_dir = project_root / "figures" / "g1" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "g1_overview.png"
    pdf = out_dir / "g1_overview.pdf"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "figure.dpi": 120,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    ax_traj, ax_q, ax_clear, ax_acc = axes.flat

    # Panel 1: raw vehicle center trajectories, downsampled only for rendering.
    by_vehicle: dict[int, list[dict[str, str]]] = {}
    for row in states:
        by_vehicle.setdefault(int(row["vehicle_id"]), []).append(row)
    for index, vehicle_id in enumerate(sorted(by_vehicle)):
        rows = by_vehicle[vehicle_id]
        stride = max(1, len(rows) // 500)
        sampled = rows[::stride]
        active = [row for row in sampled if row["active"].lower() == "true"]
        ax_traj.plot([_float(row, "x") for row in active], [_float(row, "y") for row in active],
                     lw=1.7, color=MORANDI[index % len(MORANDI)], label=f"Vehicle {vehicle_id}")
    ax_traj.set_title("Vehicle center trajectories (raw states)")
    ax_traj.set_xlabel("Longitudinal position x / m")
    ax_traj.set_ylabel("Lateral position y / m")
    ax_traj.legend(ncol=2, fontsize=8, frameon=False)

    time = [_float(row, "t") for row in metrics]
    q = [_float(row, "Q") for row in metrics]
    qrec = [_float(row, "Q_rec") for row in metrics]
    qcap = [_float(row, "Q_cap") for row in metrics]
    ax_q.plot(time, q, color="#5F766D", lw=2.1, label="Q(t)")
    ax_q.plot(time, qrec, color="#A77C6B", lw=1.8, label="Q_rec(t)")
    ax_q.plot(time, qcap, color="#7D8EA3", lw=1.4, ls="--", label="Q_cap(t)")
    if "Q_target" in summary:
        ax_q.axhline(float(summary["Q_target"]), color="#B39B6B", ls=":", lw=1.5, label="Recovery target")
    ax_q.set_ylim(0, 1.05)
    ax_q.set_title("Mission service and residual recovery")
    ax_q.set_xlabel("Time / s")
    ax_q.set_ylabel("Normalized performance")
    ax_q.legend(frameon=False, fontsize=8)

    safety = float(config["recovery"]["safety_clearance"])
    clearance = [_float(row, "min_clearance", float("nan")) for row in metrics]
    ax_clear.plot(time, clearance, color="#6F8F8B", lw=2.0, label="Minimum clearance")
    ax_clear.axhline(safety, color="#9B4A43", ls="--", lw=1.4, label=f"Safety threshold ({safety:.2f} m)")
    ax_clear.set_title("Continuous safety clearance")
    ax_clear.set_xlabel("Time / s")
    ax_clear.set_ylabel("Envelope clearance / m")
    ax_clear.legend(frameon=False, fontsize=8)

    ax_values = [_float(row, "max_abs_ax") for row in metrics]
    ay_values = [_float(row, "max_abs_ay") for row in metrics]
    ax_acc.plot(time, ax_values, color="#7D8EA3", lw=1.8, label="max |a_x|")
    ax_acc.plot(time, ay_values, color="#B07C61", lw=1.8, label="max |a_y|")
    ax_acc.axhline(float(config["vehicle"]["a_x_max"]), color="#7D8EA3", ls=":", lw=1.0)
    ax_acc.axhline(float(config["vehicle"]["a_y_max"]), color="#B07C61", ls=":", lw=1.0)
    ax_acc.set_title("Applied acceleration and limits")
    ax_acc.set_xlabel("Time / s")
    ax_acc.set_ylabel("Acceleration / m s$^{-2}$")
    ax_acc.legend(frameon=False, fontsize=8)

    for event in events:
        event_time = _float(event, "t_effective", _float(event, "t_occurrence"))
        kind = event.get("kind", "event")
        for axis in (ax_q, ax_clear, ax_acc):
            axis.axvline(event_time, color="#8B8B83", lw=0.9, ls="--", alpha=0.7)
        ax_q.text(event_time, 0.04, kind.replace("_", " "), rotation=90,
                  va="bottom", ha="right", fontsize=6.8, color="#66665F")

    fig.suptitle(f"G1 diagnostic overview — {run_dir.name}\n"
                 f"R_mission={summary['R_mission']:.4f}, safety={summary['safety_ok']}",
                 fontsize=13, fontweight="bold", color="#33443D")
    fig.savefig(png, dpi=180, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return [png, pdf]

