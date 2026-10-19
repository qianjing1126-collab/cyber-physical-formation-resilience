from __future__ import annotations

import csv
from pathlib import Path
import sys
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Polygon, Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import plot_v21_g6_results as g6  # noqa: E402


OUT = ROOT / "figures" / "paper_final"
DARK = "#3F4D47"
RED = "#B65F59"
SERVICE = "#5F786C"
CAPACITY = "#9D9992"
COUNTERFACTUAL = "#7B716B"

LAYOUTS = (
    ("F1_parallel", "f1"),
    ("F2_oblique", "f2"),
    ("FV_vshape", "fv"),
)
VEHICLE_MARKERS = ("o", "s", "^", "x", "D", "P", "v", "*")


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.2,
        "axes.labelsize": 7.8,
        "axes.titlesize": 8.1,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "legend.fontsize": 6.2,
        "axes.edgecolor": DARK,
        "axes.linewidth": 0.72,
        "grid.color": "#D9DEDA",
        "grid.linewidth": 0.52,
        "grid.alpha": 0.68,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def nearest(rows: list[dict[str, str]], time_s: float) -> dict[str, str]:
    return min(rows, key=lambda row: abs(float(row["t"]) - time_s))


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png")
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)


def draw_coverage_strip(axis: plt.Axes, x: np.ndarray, y: np.ndarray,
                        half_width: float = 2.0) -> None:
    """Draw a constant physical-width coverage strip around one trajectory."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    tangent_norm = np.hypot(dx, dy)
    tangent_norm[tangent_norm < 1.0e-9] = 1.0
    normal_x = -dy / tangent_norm
    normal_y = dx / tangent_norm
    left_x, left_y = x + half_width * normal_x, y + half_width * normal_y
    right_x, right_y = x - half_width * normal_x, y - half_width * normal_y
    polygon_x = np.concatenate((left_x, right_x[::-1]))
    polygon_y = np.concatenate((left_y, right_y[::-1]))
    axis.add_patch(Polygon(
        np.column_stack((polygon_x, polygon_y)), closed=True,
        facecolor="#AEB8B3", edgecolor="none", alpha=0.035, zorder=0,
    ))
    axis.plot(left_x, left_y, color="#737D78", linewidth=0.48,
              alpha=0.68, zorder=1)
    axis.plot(right_x, right_y, color="#737D78", linewidth=0.48,
              alpha=0.68, zorder=1)


def plot_trajectory_panel(layout: str, suffix: str,
                          states: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 1.94))
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.22, top=0.96)
    layout_states = [row for row in states if row["layout"] == layout]

    for vehicle_id in range(1, 9):
        rows = [row for row in layout_states if int(row["vehicle_id"]) == vehicle_id]
        active = [row for row in rows if row["active"].lower() == "true"]
        active_x = np.asarray([float(row["x"]) for row in active])
        active_y = np.asarray([float(row["y"]) for row in active])
        draw_coverage_strip(ax, active_x, active_y, half_width=2.0)
        marker_step = max(1, len(active_x) // 11)
        ax.plot(
            active_x, active_y,
            color=g6.VEHICLE_COLORS[vehicle_id - 1], linewidth=0.90, alpha=0.92,
            marker=VEHICLE_MARKERS[vehicle_id - 1], markevery=marker_step,
            markersize=2.1, markeredgewidth=0.45,
            label=f"Vehicle {vehicle_id}", zorder=3,
        )

        if vehicle_id in (2, 5):
            failure_time = 15.0 if vehicle_id == 2 else 30.0
            failed = nearest(rows, failure_time)
            failure_point = (float(failed["x"]), float(failed["y"]))
            ax.scatter(*failure_point, marker="X", s=25, color=RED,
                       edgecolor="white", linewidth=0.45, zorder=10)
            # Keep both labels in open regions. Straight leader arrows make
            # the event-to-marker relation unambiguous in the compact panel.
            if vehicle_id == 2:
                text_position = (0.55, 0.95) if layout == "F1_parallel" else (0.56, 0.94)
            else:
                text_position = (0.79, 0.64) if layout == "F1_parallel" else (0.79, 0.61)
            ax.annotate(
                f"{failure_time:.0f} s: Vehicle {vehicle_id} dropout",
                failure_point, xycoords="data", xytext=text_position,
                textcoords="axes fraction", fontsize=5.4, color=RED,
                ha="center", va="center",
                arrowprops={"arrowstyle": "->", "color": RED, "lw": 0.55,
                            "shrinkA": 1.5, "shrinkB": 1.5},
                bbox={"boxstyle": "round,pad=0.16", "facecolor": "white",
                      "edgecolor": "none", "alpha": 0.88}
                if vehicle_id == 2 else None,
                zorder=11,
            )

    for snapshot_time, alpha in ((10.0, 0.48), (24.0, 0.70), (44.0, 1.0)):
        for vehicle_id in range(1, 9):
            rows = [row for row in layout_states if int(row["vehicle_id"]) == vehicle_id]
            row = nearest(rows, snapshot_time)
            if row["active"].lower() == "true":
                g6.draw_snowplow(
                    ax, float(row["x"]), float(row["y"]), float(row["heading"]),
                    g6.VEHICLE_COLORS[vehicle_id - 1], "", alpha,
                )

    ax.grid(alpha=0.45)
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.set_xlim(15, 225)
    ax.set_ylim(-85, 20)
    ax.set_aspect("equal", adjustable="box")

    if layout == "F1_parallel":
        ax.legend(loc="upper right", ncols=2, fontsize=5.0,
                  columnspacing=0.55, handlelength=1.25, labelspacing=0.20)
        ax.annotate(
            "", xy=(31.0, 14.6), xytext=(31.0, 10.6),
            arrowprops={"arrowstyle": "<->", "color": DARK, "lw": 0.55},
            zorder=12,
        )
        ax.text(33.2, 12.6, "4 m", fontsize=5.3,
                va="center", ha="left", color=DARK)

    # Local 10-s formation inset.
    inset = ax.inset_axes([0.055, 0.065, 0.25, 0.43])
    local_rows: list[tuple[int, float, float, float]] = []
    for vehicle_id in range(1, 9):
        rows = [row for row in layout_states if int(row["vehicle_id"]) == vehicle_id]
        row = nearest(rows, 10.0)
        if row["active"].lower() == "true":
            local_rows.append((vehicle_id, float(row["x"]), float(row["y"]),
                               float(row["heading"])))
    center_x = float(np.mean([row[1] for row in local_rows]))
    center_y = float(np.mean([row[2] for row in local_rows]))
    for vehicle_id, x_value, y_value, heading in local_rows:
        g6.draw_snowplow(
            inset, x_value - center_x, y_value - center_y, heading,
            g6.VEHICLE_COLORS[vehicle_id - 1], "", 1.0,
        )
    local_x = [row[1] - center_x for row in local_rows]
    local_y = [row[2] - center_y for row in local_rows]
    inset.set_xlim(min(local_x) - 5.5, max(local_x) + 5.5)
    inset.set_ylim(min(local_y) - 3.5, max(local_y) + 3.5)
    inset.set_aspect("equal", adjustable="box")
    inset.set_title("10 s formation", fontsize=5.4, pad=1.5)
    inset.tick_params(labelsize=4.4, length=1.5, pad=1.0)
    inset.grid(alpha=0.28)
    source_x = [row[1] for row in local_rows]
    source_y = [row[2] for row in local_rows]
    ax.add_patch(Rectangle(
        (min(source_x) - 2.0, min(source_y) - 2.0),
        max(source_x) - min(source_x) + 4.0,
        max(source_y) - min(source_y) + 4.0,
        fill=False, linestyle="--", linewidth=0.55,
        edgecolor="#9D9992", zorder=8,
    ))
    save(fig, f"fig7{('a' if suffix == 'f1' else 'b' if suffix == 'f2' else 'c')}_{suffix}_trajectory")


def plot_service_panel(layout: str, suffix: str,
                       metrics: list[dict[str, str]], relative_gain: float) -> None:
    fig, ax = plt.subplots(figsize=(3.72, 1.79))
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.25, top=0.96)
    recovery = [row for row in metrics
                if row["layout"] == layout and row["scenario"] == "S3_recovery"]
    no_recovery = [row for row in metrics
                   if row["layout"] == layout and row["scenario"] == "no_recovery"]
    time = np.asarray([float(row["t"]) for row in recovery])
    q = np.asarray([float(row["Q"]) for row in recovery])
    qcap = np.asarray([float(row["Q_cap"]) for row in recovery])
    q_no = np.asarray([float(row["Q"]) for row in no_recovery])

    ax.plot(time, q, color=SERVICE, linewidth=1.25, label="$Q(t)$ recovery")
    ax.plot(time, qcap, color=CAPACITY, linewidth=0.95, linestyle="--",
            label="$Q_{cap}(t)$")
    ax.plot(time, q_no, color=COUNTERFACTUAL, linewidth=0.90,
            linestyle=(0, (5, 3)), label="$Q_{no-rec}(t)$")
    ax.fill_between(time, q_no, q, where=q >= q_no - 1.0e-12,
                    color=SERVICE, alpha=0.18, interpolate=True)
    for failure_time in (15.0, 30.0):
        ax.axvline(failure_time, color=RED, linestyle="--", linewidth=0.72)
    ax.set_xlim(0, 50)
    ax.set_ylim(-0.02, 1.06)
    ax.grid()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized service")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=SERVICE, edgecolor="none", alpha=0.18))
    labels.append(f"Mission gain vs no recovery: +{relative_gain:.2f}%")
    ax.legend(handles, labels, loc="lower right", fontsize=5.0,
              handlelength=1.7, labelspacing=0.20, borderaxespad=0.42)
    save(fig, f"fig8{('a' if suffix == 'f1' else 'b' if suffix == 'f2' else 'c')}_{suffix}_service")


def main() -> None:
    configure_style()
    run = ROOT / "runs" / "v2_2_clearance" / "g6" / "V21_G6_sequential_recovery_N8"
    states = read_csv(run / "representative_states.csv")
    metrics = read_csv(run / "representative_metrics.csv")
    records = read_csv(run / "run_records.csv")
    relative_gains = {
        layout: mean(
            100.0 * float(row["R_mission_gain"]) / float(row["R_mission_no_recovery"])
            for row in records if row["layout"] == layout
        ) for layout, _ in LAYOUTS
    }
    for layout, suffix in LAYOUTS:
        plot_trajectory_panel(layout, suffix, states)
        plot_service_panel(layout, suffix, metrics, relative_gains[layout])
    print(f"Generated split recovery panels in: {OUT}")


if __name__ == "__main__":
    main()
