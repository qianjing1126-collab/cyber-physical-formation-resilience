from __future__ import annotations

import argparse
import csv
import json
from math import cos, sin
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D
import numpy as np


LAYOUTS = ("F1_parallel", "F2_oblique", "FV_vshape")
LAYOUT_LABELS = {
    "F1_parallel": "Formation 1 对齐编队",
    "F2_oblique": "Formation 2 等距错开编队",
    "FV_vshape": "Formation 3 非均匀错开编队",
}
VEHICLE_COLORS = (
    "#50766B", "#C7896F", "#7A8FA3", "#B7A07A",
    "#8D7B8E", "#789A90", "#B27B68", "#738468",
)
SERVICE_COLOR = "#5F786C"
CAPACITY_COLOR = "#9D9992"
COUNTERFACTUAL_COLOR = "#7B716B"
FAILURE_COLOR = "#B55A4F"


def style() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11,
        "axes.edgecolor": "#3F4D47",
        "axes.linewidth": 0.9,
        "grid.color": "#D9DEDA",
        "grid.alpha": 0.72,
        "legend.frameon": False,
        "savefig.dpi": 240,
        "savefig.bbox": "tight",
    })


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.png")
    fig.savefig(output / f"{stem}.pdf")
    plt.close(fig)


def draw_snowplow(
    axis: plt.Axes,
    x: float,
    y: float,
    heading: float,
    color: str,
    label: str,
    alpha: float,
) -> None:
    """Draw a compact top-view airport snowplow with visible running gear."""
    transform = Affine2D().rotate(heading).translate(x, y) + axis.transData
    chassis = FancyBboxPatch(
        (-2.25, -0.96), 4.35, 1.92,
        boxstyle="round,pad=0.03,rounding_size=0.18",
        facecolor="#39443F", edgecolor="#28312D", linewidth=0.55,
        alpha=alpha, transform=transform, zorder=6,
    )
    hopper = Polygon(
        [(-2.05, -0.73), (-0.30, -0.84), (0.12, -0.67), (0.12, 0.67),
         (-0.30, 0.84), (-2.05, 0.73)],
        closed=True, facecolor=color, edgecolor="#2F3935", linewidth=0.55,
        alpha=alpha, transform=transform, zorder=7,
    )
    hopper_panel = Rectangle(
        (-1.72, -0.55), 1.25, 1.10,
        facecolor="none", edgecolor="#5C6964", linewidth=0.40,
        alpha=0.75 * alpha, transform=transform, zorder=8,
    )
    cab = Polygon(
        [(0.22, -0.78), (1.43, -0.68), (1.88, -0.38), (1.98, 0.0),
         (1.88, 0.38), (1.43, 0.68), (0.22, 0.78)],
        closed=True, facecolor="#DCE2DF", edgecolor="#2F3935", linewidth=0.55,
        alpha=alpha, transform=transform, zorder=7,
    )
    windshield = Polygon(
        [(0.96, -0.54), (1.48, -0.45), (1.78, -0.25), (1.84, 0.0),
         (1.78, 0.25), (1.48, 0.45), (0.96, 0.54)],
        closed=True, facecolor="#8FAAA7", edgecolor="#5C6D68", linewidth=0.32,
        alpha=0.78 * alpha, transform=transform, zorder=8,
    )
    hood = Rectangle(
        (1.82, -0.34), 0.34, 0.68,
        facecolor=color, edgecolor="#2F3935", linewidth=0.35,
        alpha=alpha, transform=transform, zorder=8,
    )
    support_top = Polygon(
        [(2.05, 0.29), (2.63, 0.70), (2.77, 0.52), (2.13, 0.13)],
        closed=True, facecolor="#59635E", edgecolor="#313A36", linewidth=0.25,
        alpha=alpha, transform=transform, zorder=6,
    )
    support_bottom = Polygon(
        [(2.05, -0.29), (2.63, -0.70), (2.77, -0.52), (2.13, -0.13)],
        closed=True, facecolor="#59635E", edgecolor="#313A36", linewidth=0.25,
        alpha=alpha, transform=transform, zorder=6,
    )
    plow = Polygon(
        [(2.55, -1.12), (2.88, -1.45), (3.40, -1.25),
         (3.40, 1.25), (2.88, 1.45), (2.55, 1.12)],
        closed=True, facecolor="#D9AA3D", edgecolor="#66532A", linewidth=0.65,
        alpha=alpha, transform=transform, zorder=8,
    )
    for patch in (chassis, hopper, hopper_panel, cab, windshield, hood,
                  support_top, support_bottom, plow):
        axis.add_patch(patch)
    # Four protruding tires remain visible after reduction to a journal figure.
    for wheel_x in (-1.38, 1.12):
        for wheel_y in (-1.10, 1.10):
            tire = FancyBboxPatch(
                (wheel_x - 0.43, wheel_y - 0.18), 0.86, 0.36,
                boxstyle="round,pad=0.01,rounding_size=0.09",
                facecolor="#202725", edgecolor="#111614", linewidth=0.35,
                alpha=alpha, transform=transform, zorder=9,
            )
            hub = Rectangle(
                (wheel_x - 0.16, wheel_y - 0.055), 0.32, 0.11,
                facecolor="#AEB8B3", edgecolor="none",
                alpha=0.85 * alpha, transform=transform, zorder=10,
            )
            axis.add_patch(tire)
            axis.add_patch(hub)
    for mirror_y in (-0.91, 0.91):
        axis.add_patch(Circle(
            (1.15, mirror_y), 0.10, facecolor="#2E3733", edgecolor="none",
            alpha=alpha, transform=transform, zorder=10,
        ))
    axis.add_patch(Circle(
        (0.55, 0.0), 0.12, facecolor="#E08A35", edgecolor="#8A5126",
        linewidth=0.28, alpha=alpha, transform=transform, zorder=10,
    ))
    for stripe_y in (-0.72, 0.0, 0.72):
        axis.plot([3.01, 3.30], [stripe_y - 0.17, stripe_y + 0.17],
                  color="#F2D685", linewidth=0.45, alpha=alpha,
                  transform=transform, zorder=9)
    if label:
        axis.text(
            x - 0.72 * cos(heading), y - 0.72 * sin(heading), label,
            ha="center", va="center", fontsize=6.4, color="#26312D",
            zorder=11, alpha=alpha,
        )


def nearest(rows: list[dict[str, str]], time_s: float) -> dict[str, str]:
    return min(rows, key=lambda row: abs(float(row["t"]) - time_s))


def _snapshot(
    axis: plt.Axes,
    layout_states: list[dict[str, str]],
    snapshot_time: float,
    alpha: float,
) -> tuple[float, float]:
    centers: list[tuple[float, float]] = []
    for vehicle_id in range(1, 9):
        rows = [row for row in layout_states if int(row["vehicle_id"]) == vehicle_id]
        row = nearest(rows, snapshot_time)
        if row["active"].lower() != "true":
            continue
        x, y, heading = float(row["x"]), float(row["y"]), float(row["heading"])
        centers.append((x, y))
        draw_snowplow(
            axis, x, y, heading, VEHICLE_COLORS[vehicle_id - 1], str(vehicle_id), alpha,
        )
    return float(np.mean([p[0] for p in centers])), float(np.mean([p[1] for p in centers]))


def layout_figure(
    layout: str,
    states: list[dict[str, str]],
    metrics: list[dict[str, str]],
    summary: dict,
    output: Path,
    figure_index: int,
) -> None:
    layout_states = [row for row in states if row["layout"] == layout]
    recovery_metrics = [
        row for row in metrics
        if row["layout"] == layout and row.get("scenario", "S3_recovery") == "S3_recovery"
    ]
    no_recovery_metrics = [
        row for row in metrics
        if row["layout"] == layout and row.get("scenario") == "no_recovery"
    ]
    if not no_recovery_metrics:
        raise ValueError("G6 no-recovery representative trace is missing")

    fig, (trajectory_axis, service_axis) = plt.subplots(
        2, 1, figsize=(12.8, 8.7), constrained_layout=True,
        gridspec_kw={"height_ratios": (1.55, 1.0)},
    )

    for vehicle_id in range(1, 9):
        rows = [row for row in layout_states if int(row["vehicle_id"]) == vehicle_id]
        active = [row for row in rows if row["active"].lower() == "true"]
        trajectory_axis.plot(
            [float(row["x"]) for row in active],
            [float(row["y"]) for row in active],
            color=VEHICLE_COLORS[vehicle_id - 1], linewidth=1.55, alpha=0.90,
        )
        if vehicle_id in (2, 5):
            failed_at = 15.0 if vehicle_id == 2 else 30.0
            failed_row = nearest(rows, failed_at)
            point = (float(failed_row["x"]), float(failed_row["y"]))
            trajectory_axis.scatter(
                *point, marker="X", s=72, color=FAILURE_COLOR,
                edgecolor="white", linewidth=0.7, zorder=10,
            )
            trajectory_axis.annotate(
                f"车辆{vehicle_id}失效\n$t={failed_at:.0f}$ s", point,
                xytext=(8, 8), textcoords="offset points", fontsize=8.1,
                color=FAILURE_COLOR,
            )

    for snapshot_time, alpha in ((0.5, 0.34), (14.0, 0.52), (24.0, 0.72), (44.0, 1.0)):
        center = _snapshot(trajectory_axis, layout_states, snapshot_time, alpha)
        trajectory_axis.annotate(
            f"{snapshot_time:.1f} s" if snapshot_time < 1.0 else f"{snapshot_time:.0f} s",
            center, xytext=(0, 17), textcoords="offset points",
            ha="center", va="bottom", fontsize=8.1, color="#5C6964",
        )

    # A small in-plot enlargement keeps the snowplow shape legible without
    # consuming a separate right-hand panel.
    snapshot_axis = trajectory_axis.inset_axes([0.055, 0.055, 0.30, 0.36])
    local_rows: list[tuple[int, float, float, float]] = []
    for vehicle_id in range(1, 9):
        rows = [row for row in layout_states if int(row["vehicle_id"]) == vehicle_id]
        row = nearest(rows, 14.0)
        if row["active"].lower() == "true":
            local_rows.append((vehicle_id, float(row["x"]), float(row["y"]), float(row["heading"])))
    center_x = float(np.mean([row[1] for row in local_rows]))
    center_y = float(np.mean([row[2] for row in local_rows]))
    for vehicle_id, x_value, y_value, heading in local_rows:
        draw_snowplow(
            snapshot_axis, x_value - center_x, y_value - center_y, heading,
            VEHICLE_COLORS[vehicle_id - 1], str(vehicle_id), 1.0,
        )
    local_x = [row[1] - center_x for row in local_rows]
    local_y = [row[2] - center_y for row in local_rows]
    snapshot_axis.set_xlim(min(local_x) - 5.5, max(local_x) + 5.5)
    snapshot_axis.set_ylim(min(local_y) - 3.5, max(local_y) + 3.5)
    snapshot_axis.set_aspect("equal", adjustable="box")
    snapshot_axis.set_title("14 s 编队局部放大", fontsize=8.2)
    snapshot_axis.tick_params(labelsize=6.5, length=2)
    snapshot_axis.grid(alpha=0.35)

    trajectory_axis.set(
        title="(a) 平滑双弯道上的车辆轨迹、顺序失效与编队快照",
        xlabel="全局位置 $x$ / m",
        ylabel="全局位置 $y$ / m",
    )
    trajectory_axis.grid(alpha=0.48)

    time = np.asarray([float(row["t"]) for row in recovery_metrics])
    q = np.asarray([float(row["Q"]) for row in recovery_metrics])
    qcap = np.asarray([float(row["Q_cap"]) for row in recovery_metrics])
    no_recovery_time = np.asarray([float(row["t"]) for row in no_recovery_metrics])
    q_no_recovery = np.asarray([float(row["Q"]) for row in no_recovery_metrics])
    if len(time) != len(no_recovery_time) or not np.allclose(time, no_recovery_time):
        raise ValueError("G6 recovery and no-recovery traces are not time aligned")

    service_axis.plot(
        time, q, color=SERVICE_COLOR, linewidth=2.3,
        label="$Q(t)$ 实际作业性能",
    )
    service_axis.plot(
        time, qcap, color=CAPACITY_COLOR, linewidth=1.8, linestyle="--",
        label="$Q_{cap}(t)$ 残余容量上限",
    )
    service_axis.plot(
        time, q_no_recovery, color=COUNTERFACTUAL_COLOR, linewidth=1.65,
        linestyle=(0, (6, 4)), label="$Q_{no-rec}(t)$ 无恢复反事实",
    )
    service_axis.fill_between(
        time, q_no_recovery, q, where=q >= q_no_recovery - 1e-12,
        color=SERVICE_COLOR, alpha=0.20, interpolate=True,
        label="恢复带来的累计作业增益",
    )
    for failure_time in (15.0, 30.0):
        service_axis.axvline(failure_time, color=FAILURE_COLOR, linestyle="--", linewidth=1.1)
    group = summary["group_summary"][layout]
    for event_index, failure_time in enumerate((15.0, 30.0), start=1):
        recovery = group[f"event{event_index}_recovery_time_s"]["mean"]
        service_axis.annotate(
            f"恢复时间 {recovery:.2f} s",
            (failure_time + recovery, 0.74 if event_index == 1 else 0.61),
            xytext=(8, 14), textcoords="offset points", fontsize=8.6,
            color=SERVICE_COLOR,
            arrowprops={"arrowstyle": "->", "color": SERVICE_COLOR, "lw": 0.8},
        )
    service_axis.set(
        title="(b) 实际恢复与无恢复反事实的覆盖任务性能演化",
        xlabel="时间 $t$ / s",
        ylabel="归一化任务性能",
        xlim=(0, 50),
        ylim=(-0.02, 1.06),
    )
    service_axis.grid()
    service_axis.legend(loc="lower left", ncol=2, fontsize=8.4)
    fig.suptitle(
        f"V2.1-G6：{LAYOUT_LABELS[layout]}的顺序失效与编队恢复",
        fontsize=14.2, color="#3F4D47",
    )
    save(fig, output, f"G6_Fig{figure_index}_{layout}_evolution")


def comparison_figure(summary: dict, output: Path) -> None:
    labels = ("并排", "斜线", "V型")
    x = np.arange(3)
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    for axis, metric_root, title, ylabel in (
        (axes[0], "recovery_time_s", "(a) 两次失效后的恢复时间", "恢复时间 / s"),
        (axes[1], "R_Te", "(b) 固定窗口任务韧性", "$R_{T_e}$"),
    ):
        for event_index, shift, color, event_label in (
            (1, -width / 2, "#7F988D", "第一次失效：车辆2"),
            (2, width / 2, "#C58F79", "第二次失效：车辆5"),
        ):
            groups = [
                summary["group_summary"][layout][f"event{event_index}_{metric_root}"]
                for layout in LAYOUTS
            ]
            means = np.asarray([group["mean"] for group in groups])
            lower = np.asarray([group["ci95_low"] for group in groups])
            upper = np.asarray([group["ci95_high"] for group in groups])
            bars = axis.bar(
                x + shift, means, width=width, color=color, edgecolor="white",
                yerr=np.vstack([means - lower, upper - means]), capsize=3,
                label=event_label,
            )
            for bar, value in zip(bars, means, strict=True):
                axis.annotate(
                    f"{value:.3f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8.3,
                )
        axis.set_xticks(x, labels)
        axis.set(title=title, ylabel=ylabel)
        axis.grid(axis="y")
    axes[0].legend(loc="upper left", fontsize=8.4)
    fig.suptitle(
        "V2.1-G6：三种典型编队的顺序失效恢复比较（30个配对种子）",
        fontsize=13.8, color="#3F4D47",
    )
    save(fig, output, "G6_Fig4_layout_recovery_comparison")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("figures") / "v2_1" / "g6")
    args = parser.parse_args()
    style()
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    states = read_csv(args.run_dir / "representative_states.csv")
    metrics = read_csv(args.run_dir / "representative_metrics.csv")
    for index, layout in enumerate(LAYOUTS, start=1):
        layout_figure(layout, states, metrics, summary, args.output_dir, index)
    comparison_figure(summary, args.output_dir)
    outputs = sorted(str(path) for path in args.output_dir.glob("G6_Fig*"))
    print(json.dumps({"figures": outputs}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
