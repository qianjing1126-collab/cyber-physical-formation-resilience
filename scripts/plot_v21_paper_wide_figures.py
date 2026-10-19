from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Polygon, Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_v21_g4_dynamic as g4  # noqa: E402
import plot_v21_g6_results as g6  # noqa: E402


OUT = ROOT / "figures" / "paper_v21"
DARK = "#3F4D47"
GREEN = "#607B70"
GRAY = "#9D9992"
RED = "#B65F59"
SERVICE = "#5F786C"
CAPACITY = "#9D9992"
COUNTERFACTUAL = "#7B716B"
METHOD_LABELS = {
    "B7_uniform": "B3",
    "B8_convergence_priority": "B4",
    "S2_delay_constrained": "S2",
}


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
        "legend.fontsize": 6.5,
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


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png")
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_e4() -> None:
    config = json.loads((ROOT / "configs" / "v2_1_g4.json").read_text(encoding="utf-8"))
    g3_summary = json.loads((ROOT / config["g3_source"] / "summary.json").read_text(encoding="utf-8"))
    edges = [tuple(edge) for edge in g3_summary["topology"]["edges"]]
    matrices = {
        item["method"]: g4.laplacian(8, edges, item["weights"])
        for item in g3_summary["methods"]
    }
    delay = float(config["stress_delay_s"])
    traces: dict[str, dict[str, np.ndarray]] = {}
    for method, matrix in matrices.items():
        trace = g4.simulate_network(delay, matrix, config)
        traces[method] = trace
    methods = ("B7_uniform", "B8_convergence_priority", "S2_delay_constrained")
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.45), sharex=True)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.105, top=0.84,
                        hspace=0.16, wspace=0.20)
    end_time = float(config["stress_plot_end_s"])
    for column, method in enumerate(methods):
        trace = traces[method]
        shown = trace["time"] <= end_time + 1.0e-12
        for row, key in enumerate(("position", "velocity")):
            ax = axes[row, column]
            for vehicle in range(8):
                ax.plot(trace["time"][shown], trace[key][vehicle, shown],
                        color=g4.VEHICLE_COLORS[vehicle], linewidth=0.72, alpha=0.88,
                        label=f"Agent {vehicle + 1}" if column == 2 else None)
            ax.axvspan(0, float(config["disturbance_duration_s"]), color="#D8C7A7", alpha=0.22)
            ax.axhline(0, color=DARK, linewidth=0.6)
            if row == 0:
                ax.set_ylim(-1.5, 1.5)
                ax.set_yticks((-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5))
            else:
                ax.set_ylim(-0.8, 0.8)
                ax.set_yticks((-0.8, -0.4, 0.0, 0.4, 0.8))
            ax.set_xlim(0, end_time)
            ax.grid()
            if column == 2:
                ax.legend(loc="upper right", ncols=2, fontsize=5.2,
                          columnspacing=0.55, handlelength=1.2, labelspacing=0.22)
        axes[0, column].set_title(METHOD_LABELS[method])
    axes[0, 0].set_ylabel("Position error (m)")
    axes[1, 0].set_ylabel(r"Velocity error (m s$^{-1}$)")
    fig.supxlabel("Time (s)", y=0.025, fontsize=7.2)
    fig.suptitle(r"Common delay: $\tau=160$ ms",
                 y=0.965, fontsize=8.4, color=DARK)
    save(fig, "E4_delay_stress_tracking_errors")


def nearest(rows: list[dict[str, str]], time_s: float) -> dict[str, str]:
    return min(rows, key=lambda row: abs(float(row["t"]) - time_s))


def draw_coverage_strip(axis: plt.Axes, x: np.ndarray, y: np.ndarray,
                        half_width: float = 2.0) -> None:
    """Draw constant-width physical strip boundaries around a center path."""
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
    axis.plot(left_x, left_y, color="#737D78", linewidth=0.48, alpha=0.68, zorder=1)
    axis.plot(right_x, right_y, color="#737D78", linewidth=0.48, alpha=0.68, zorder=1)


def plot_e5() -> None:
    run = ROOT / "runs" / "v2_1" / "g6" / "V21_G6_sequential_recovery_N8_retry01"
    states = read_csv(run / "representative_states.csv")
    metrics = read_csv(run / "representative_metrics.csv")
    layouts = (("F1_parallel", "Formation 1: aligned"),
               ("F2_oblique", "Formation 2: uniformly staggered"),
               ("FV_vshape", "Formation 3: asymmetric"))
    fig, axes = plt.subplots(
        3, 2, figsize=(7.16, 7.55),
        gridspec_kw={"width_ratios": (1.72, 1.0)},
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.075, top=0.965,
                        hspace=0.30, wspace=0.20)
    relative_gains = {
        "F1_parallel": 24.76,
        "F2_oblique": 23.85,
        "FV_vshape": 23.95,
    }
    vehicle_markers = ("o", "s", "^", "x", "D", "P", "v", "*")
    for row_index, (layout, label) in enumerate(layouts):
        trajectory_ax = axes[row_index, 0]
        service_ax = axes[row_index, 1]
        layout_states = [row for row in states if row["layout"] == layout]
        recovery = [row for row in metrics if row["layout"] == layout and row["scenario"] == "S3_recovery"]
        no_recovery = [row for row in metrics if row["layout"] == layout and row["scenario"] == "no_recovery"]
        for vehicle_id in range(1, 9):
            rows = [row for row in layout_states if int(row["vehicle_id"]) == vehicle_id]
            active = [row for row in rows if row["active"].lower() == "true"]
            active_x = np.asarray([float(row["x"]) for row in active])
            active_y = np.asarray([float(row["y"]) for row in active])
            draw_coverage_strip(trajectory_ax, active_x, active_y, half_width=2.0)
            marker_step = max(1, len(active_x) // 11)
            trajectory_ax.plot(
                active_x, active_y,
                color=g6.VEHICLE_COLORS[vehicle_id - 1], linewidth=0.90, alpha=0.92,
                marker=vehicle_markers[vehicle_id - 1], markevery=marker_step,
                markersize=2.1, markeredgewidth=0.45,
                label=f"Agent {vehicle_id}" if row_index == 0 else None,
                zorder=3,
            )
            if vehicle_id in (2, 5):
                failure_time = 15.0 if vehicle_id == 2 else 30.0
                row = nearest(rows, failure_time)
                failure_point = (float(row["x"]), float(row["y"]))
                trajectory_ax.scatter(*failure_point, marker="X", s=25,
                                      color=RED, edgecolor="white", linewidth=0.45, zorder=10)
                text_position = (0.43, 0.79) if vehicle_id == 2 else (0.70, 0.48)
                trajectory_ax.annotate(
                    f"{failure_time:.0f} s: Agent {vehicle_id} dropout",
                    failure_point, xycoords="data", xytext=text_position,
                    textcoords="axes fraction", fontsize=5.5, color=RED,
                    ha="center", va="center",
                    arrowprops={"arrowstyle": "->", "color": RED, "lw": 0.55},
                    zorder=11,
                )
        for snapshot_time, alpha in ((10.0, 0.48), (24.0, 0.70), (44.0, 1.0)):
            centers = []
            for vehicle_id in range(1, 9):
                rows = [row for row in layout_states if int(row["vehicle_id"]) == vehicle_id]
                row = nearest(rows, snapshot_time)
                if row["active"].lower() != "true":
                    continue
                x, y, heading = float(row["x"]), float(row["y"]), float(row["heading"])
                centers.append((x, y))
                g6.draw_snowplow(trajectory_ax, x, y, heading,
                                 g6.VEHICLE_COLORS[vehicle_id - 1], "", alpha)
        trajectory_ax.set_title(label)
        trajectory_ax.grid(alpha=0.45)
        trajectory_ax.set_ylabel("$y$ (m)")
        trajectory_ax.set_xlim(15, 225)
        trajectory_ax.set_ylim(-85, 20)
        trajectory_ax.set_aspect("equal", adjustable="box")
        if row_index == 0:
            trajectory_ax.legend(loc="upper right", ncols=2, fontsize=5.1,
                                 columnspacing=0.55, handlelength=1.25, labelspacing=0.22)
            trajectory_ax.annotate(
                "", xy=(31.0, 14.6), xytext=(31.0, 10.6),
                arrowprops={"arrowstyle": "<->", "color": DARK, "lw": 0.55},
                zorder=12,
            )
            trajectory_ax.text(33.2, 12.6, "4 m", fontsize=5.3,
                               va="center", ha="left", color=DARK)

        # Enlarge the intact 10-s formation inside each trajectory panel.
        inset = trajectory_ax.inset_axes([0.045, 0.055, 0.27, 0.42])
        local_rows: list[tuple[int, float, float, float]] = []
        for vehicle_id in range(1, 9):
            rows = [row for row in layout_states if int(row["vehicle_id"]) == vehicle_id]
            row = nearest(rows, 10.0)
            if row["active"].lower() == "true":
                local_rows.append((vehicle_id, float(row["x"]), float(row["y"]), float(row["heading"])))
        center_x = float(np.mean([row[1] for row in local_rows]))
        center_y = float(np.mean([row[2] for row in local_rows]))
        for vehicle_id, x_value, y_value, heading in local_rows:
            g6.draw_snowplow(inset, x_value - center_x, y_value - center_y, heading,
                             g6.VEHICLE_COLORS[vehicle_id - 1], "", 1.0)
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
        trajectory_ax.add_patch(Rectangle(
            (min(source_x) - 2.0, min(source_y) - 2.0),
            max(source_x) - min(source_x) + 4.0,
            max(source_y) - min(source_y) + 4.0,
            fill=False, linestyle="--", linewidth=0.55, edgecolor=GRAY, zorder=8,
        ))

        time = np.asarray([float(row["t"]) for row in recovery])
        q = np.asarray([float(row["Q"]) for row in recovery])
        qcap = np.asarray([float(row["Q_cap"]) for row in recovery])
        q_no = np.asarray([float(row["Q"]) for row in no_recovery])
        service_ax.plot(time, q, color=SERVICE, linewidth=1.25, label="$Q(t)$ recovery")
        service_ax.plot(time, qcap, color=CAPACITY, linewidth=0.95, linestyle="--",
                        label="$Q_{cap}(t)$")
        service_ax.plot(time, q_no, color=COUNTERFACTUAL, linewidth=0.90,
                        linestyle=(0, (5, 3)), label="$Q_{no-rec}(t)$")
        service_ax.fill_between(time, q_no, q, where=q >= q_no - 1.0e-12,
                                color=SERVICE, alpha=0.18, interpolate=True)
        for failure_time in (15.0, 30.0):
            service_ax.axvline(failure_time, color=RED, linestyle="--", linewidth=0.72)
        service_ax.set_xlim(0, 50)
        service_ax.set_ylim(-0.02, 1.06)
        service_ax.grid()
        service_ax.set_ylabel("Normalized service")
        handles, legend_labels = service_ax.get_legend_handles_labels()
        handles.append(Patch(facecolor=SERVICE, edgecolor="none", alpha=0.18))
        legend_labels.append(f"Mission gain vs no recovery: +{relative_gains[layout]:.2f}%")
        service_ax.legend(handles, legend_labels, loc="lower right", fontsize=4.9,
                          handlelength=1.7, labelspacing=0.22, borderaxespad=0.45)
        if row_index == 2:
            trajectory_ax.set_xlabel("$x$ (m)")
            service_ax.set_xlabel("Time (s)")
    save(fig, "E5_layout_generalization_sequential_failures")


def main() -> None:
    configure_style()
    plot_e4()
    plot_e5()
    print(json.dumps({"output_dir": str(OUT), "figures": sorted(path.name for path in OUT.glob("E[45]*"))}, indent=2))


if __name__ == "__main__":
    main()
