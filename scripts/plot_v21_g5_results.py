from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "B7_uniform": "#9D9992",
    "B8_convergence_priority": "#C58F79",
    "S2_delay_constrained": "#5F786C",
}
LABELS = {
    "B7_uniform": "B3 均匀权重",
    "B8_convergence_priority": "B4 收敛优先",
    "S2_delay_constrained": "S2 时延约束",
}
REGIMES = ("light", "moderate", "heavy")
REGIME_LABELS = ("C1", "C2", "C3")


def style() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 10.5, "axes.titlesize": 12.5,
        "axes.labelsize": 11, "axes.edgecolor": "#3F4D47", "axes.linewidth": 0.9,
        "grid.color": "#D9DEDA", "grid.alpha": 0.72, "legend.frameon": False,
        "savefig.dpi": 240, "savefig.bbox": "tight",
    })


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.png")
    fig.savefig(output / f"{stem}.pdf")
    plt.close(fig)


def network_quality(summary: dict, output: Path) -> None:
    panels = (
        ("pdr", "(a) 单跳分组投递率", "PDR", 1.0, (0.70, 1.01)),
        ("delay_p95_s", "(b) 95%分组时延", "时延 / ms", 1000.0, None),
        ("aoi_p95_s", "(c) 95%信息年龄", "AoI / ms", 1000.0, None),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5), constrained_layout=True)
    for axis, (metric, title, ylabel, scale, ylim) in zip(axes, panels, strict=True):
        groups = [summary["group_summary"][f"{regime}/B7_uniform"][metric] for regime in REGIMES]
        means = np.asarray([group["mean"] * scale for group in groups])
        errors = np.asarray([
            [means[i] - group["ci95_low"] * scale for i, group in enumerate(groups)],
            [group["ci95_high"] * scale - means[i] for i, group in enumerate(groups)],
        ])
        bars = axis.bar(REGIME_LABELS, means, color=["#AAB9B0", "#8299AC", "#C58F79"],
                        edgecolor="white", width=0.62, yerr=errors, capsize=4)
        axis.set(title=title, ylabel=ylabel)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.grid(axis="y")
        for bar, value in zip(bars, means, strict=True):
            fmt = f"{value:.3f}" if metric == "pdr" else f"{value:.1f}"
            axis.annotate(fmt, (bar.get_x() + bar.get_width()/2, value), xytext=(0, 5),
                          textcoords="offset points", ha="center", fontsize=9.3)
    fig.suptitle("V2.1-G5：ns-3标定的三个明确通信条件", fontsize=14.5, color="#3F4D47")
    save(fig, output, "G5_Fig1_packet_network_conditions")


def closed_loop(summary: dict, output: Path) -> None:
    panels = (
        ("position_iae_m_s", "(a) 积分RMS位置误差", r"$J_p$ / (m·s)"),
        ("settling_time_penalized_s", "(b) 稳定时间", "时间 / s"),
        ("control_energy_m2ps3", "(c) 二次控制努力", r"$J_u$ / (m$^2$·s$^{-3}$)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7), constrained_layout=True)
    x = np.arange(len(REGIMES))
    for axis, (metric, title, ylabel) in zip(axes, panels, strict=True):
        all_low, all_high = [], []
        for method in LABELS:
            groups = [summary["group_summary"][f"{regime}/{method}"][metric] for regime in REGIMES]
            means = np.asarray([group["mean"] for group in groups])
            lower = np.asarray([group["ci95_low"] for group in groups])
            upper = np.asarray([group["ci95_high"] for group in groups])
            all_low.extend(lower); all_high.extend(upper)
            axis.errorbar(x, means, yerr=np.vstack([means-lower, upper-means]), color=COLORS[method],
                          marker="o", markersize=6, linewidth=2.0, capsize=3, label=LABELS[method])
        span = max(all_high) - min(all_low)
        pad = max(0.08 * span, 1e-4)
        axis.set_ylim(min(all_low) - pad, max(all_high) + pad)
        axis.set_xticks(x, REGIME_LABELS)
        axis.set(title=title, ylabel=ylabel)
        axis.grid()
    axes[0].legend(loc="best", fontsize=8.7)
    fig.suptitle("V2.1-G5：相同分组过程下的加权协调闭环比较（均值及95%置信区间）",
                 fontsize=14.2, color="#3F4D47")
    save(fig, output, "G5_Fig2_closed_loop_method_comparison")


def heavy_trace(rows: list[dict[str, str]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    for method in LABELS:
        subset = [row for row in rows if row["regime"] == "heavy" and row["method"] == method]
        time = [float(row["time_s"]) for row in subset]
        axes[0].plot(time, [float(row["position_rms_m"]) for row in subset],
                     color=COLORS[method], linewidth=2.1, label=LABELS[method])
        axes[1].plot(time, [float(row["velocity_rms_mps"]) for row in subset],
                     color=COLORS[method], linewidth=2.1, label=LABELS[method])
    for axis, title, ylabel in zip(
        axes, ("(a) 位置一致性误差", "(b) 速度一致性误差"),
        ("RMS / m", "RMS / (m/s)"), strict=True,
    ):
        axis.axvspan(0, 1.5, color="#D8C7A7", alpha=0.22)
        axis.set(xlabel="时间 / s", ylabel=ylabel, title=title, xlim=(0, 8))
        axis.grid()
    axes[0].legend(fontsize=8.7)
    fig.suptitle("V2.1-G5：C3随机分组网络下的代表性闭环轨迹", fontsize=14, color="#3F4D47")
    save(fig, output, "G5_Fig3_heavy_network_representative_trace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/v2_1/g5"))
    args = parser.parse_args()
    style()
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = read_csv(args.run_dir / "representative_traces.csv")
    network_quality(summary, args.output_dir)
    closed_loop(summary, args.output_dir)
    heavy_trace(rows, args.output_dir)
    print(json.dumps({"figures": sorted(str(path) for path in args.output_dir.glob("G5_Fig*"))},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
