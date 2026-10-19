from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"B9": "#C58F79", "B11": "#7D94A7", "B10": "#B6A477", "S3": "#607B70"}
MARKERS = {"B11": "D", "B10": "s", "S3": "o"}
DISPLAY_LABELS = {"B9": "B5", "B10": "B6", "B11": "B7", "S3": "S3", "B10/S3": "B6/S3"}


def style() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10.5,
        "axes.titlesize": 12.0,
        "axes.labelsize": 11,
        "axes.edgecolor": "#3F4D47",
        "axes.linewidth": 0.9,
        "grid.color": "#D9DEDA",
        "grid.alpha": 0.72,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def plot(run_dir: Path, output_dir: Path) -> list[Path]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if not summary["passed"]:
        raise RuntimeError("G8 formal acceptance failed; refuse to plot final result")
    resilience = sorted(read_csv(run_dir / "resilience_summary.csv"), key=lambda row: int(row["n"]))
    planner = read_csv(run_dir / "plan_construction_summary.csv")
    communication = read_csv(run_dir / "communication_summary.csv")
    n_values = np.asarray([int(row["n"]) for row in resilience], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.65), constrained_layout=True)

    gain = np.asarray([float(row["relative_gain_mean_percent"]) for row in resilience])
    gain_low = np.asarray([float(row["relative_gain_ci95_low_percent"]) for row in resilience])
    gain_high = np.asarray([float(row["relative_gain_ci95_high_percent"]) for row in resilience])
    axes[0].errorbar(
        n_values, gain, yerr=np.vstack((gain - gain_low, gain_high - gain)),
        color=COLORS["S3"], marker="o", markersize=6.5, linewidth=2.0,
        markeredgecolor="white", markeredgewidth=0.7, capsize=3.0,
        label="S3相对LC",
    )
    axes[0].set_title("(a) 固定窗口韧性提升")
    axes[0].set_ylabel("相对提升 $G_R$ / %")
    axes[0].set_ylim(60.0, 83.0)
    axes[0].legend(loc="lower right")

    for method in ("B11", "B10/S3"):
        rows = sorted((row for row in planner if row["method"] == method), key=lambda row: int(row["n"]))
        mean = np.asarray([float(row["mean_construction_time_us"]) for row in rows])
        low = np.asarray([float(row["mean_ci95_low_us"]) for row in rows])
        high = np.asarray([float(row["mean_ci95_high_us"]) for row in rows])
        axes[1].errorbar(
            n_values, mean, yerr=np.vstack((mean - low, high - mean)),
            color=COLORS["S3" if method == "B10/S3" else method],
            marker=MARKERS["S3" if method == "B10/S3" else method], markersize=6.2,
            linewidth=2.0, markeredgecolor="white", markeredgewidth=0.7,
            capsize=3.0, label=DISPLAY_LABELS[method],
        )
    axes[1].set_title("(b) 在线恢复规划时间")
    axes[1].set_ylabel("平均规划时间 / μs")
    axes[1].set_ylim(0.0, 175.0)
    axes[1].legend(loc="upper left")

    for method in ("B11", "B10", "S3"):
        rows = sorted((row for row in communication if row["method"] == method), key=lambda row: int(row["n"]))
        megabytes = np.asarray([float(row["wireless_megabytes"]) for row in rows])
        axes[2].plot(
            n_values, megabytes, color=COLORS[method], marker=MARKERS[method],
            markersize=6.2, linewidth=2.0, markeredgecolor="white",
            markeredgewidth=0.7,
            linestyle="--" if method == "B10" else "-",
            zorder=4 if method == "B10" else 3,
            label=DISPLAY_LABELS[method],
        )
    axes[2].set_title("(c) 30 s累计无线发送开销")
    axes[2].set_ylabel("累计空口流量 / MB")
    axes[2].set_ylim(0.0, 2.0)
    axes[2].legend(loc="upper left")

    for axis in axes:
        axis.set_xlabel("车辆数 $N$")
        axis.set_xticks(n_values)
        axis.grid(axis="y")
    fig.suptitle(
        "V2.1-G8：车辆规模对韧性收益、在线计算与通信开销的影响",
        fontsize=14.0, color="#3F4D47",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "G8_Fig1_scale_resilience_computation_communication.png",
        output_dir / "G8_Fig1_scale_resilience_computation_communication.pdf",
    ]
    fig.savefig(outputs[0])
    fig.savefig(outputs[1])
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/v2_1/g8"))
    args = parser.parse_args()
    style()
    outputs = plot(args.run_dir, args.output_dir)
    print(json.dumps({"figures": [str(path) for path in outputs]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
