from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("B9", "B11", "S3")
COLORS = {"B9": "#C58F79", "B11": "#7D94A7", "S3": "#607B70"}
MARKERS = {"B9": "s", "B11": "D", "S3": "o"}
DISPLAY_LABELS = {"B9": "B5", "B11": "B7", "S3": "S3"}
SCENARIOS = (
    ("F1_parallel", "A_adjacent", "Formation 1 / Adj."),
    ("F1_parallel", "N_nonadjacent", "Formation 1 / Nonadj."),
    ("F2_oblique", "A_adjacent", "Formation 2 / Adj."),
    ("F2_oblique", "N_nonadjacent", "Formation 2 / Nonadj."),
    ("FV_vshape", "A_adjacent", "Formation 3 / Adj."),
    ("FV_vshape", "N_nonadjacent", "Formation 3 / Nonadj."),
)


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
        "savefig.dpi": 260,
        "savefig.bbox": "tight",
    })


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def lookup(rows: list[dict[str, str]], layout: str, fault: str, method: str) -> dict[str, str]:
    return next(
        row for row in rows
        if row["layout"] == layout and row["fault_pattern"] == fault and row["method"] == method
    )


def plot(run_dir: Path, output_dir: Path) -> list[Path]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = read_csv(run_dir / "pilot_group_summary.csv")
    x = np.arange(len(SCENARIOS), dtype=float)
    offsets = {"B9": -0.18, "B11": 0.0, "S3": 0.18}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)

    for method in METHODS:
        selected = [lookup(rows, layout, fault, method) for layout, fault, _ in SCENARIOS]
        success = np.asarray([float(row["recovery_success_rate"]) for row in selected])
        success_low = np.asarray([float(row["success_ci_low"]) for row in selected])
        success_high = np.asarray([float(row["success_ci_high"]) for row in selected])
        rte = np.asarray([float(row["R_Te_mean"]) for row in selected])
        rte_low = np.asarray([float(row["R_Te_ci_low"]) for row in selected])
        rte_high = np.asarray([float(row["R_Te_ci_high"]) for row in selected])
        axes[0].errorbar(
            x + offsets[method], success,
            yerr=np.maximum(0.0, np.vstack((success - success_low, success_high - success))),
            fmt=MARKERS[method], markersize=7.0, color=COLORS[method],
            markeredgecolor="white", markeredgewidth=0.7,
            capsize=3.2, elinewidth=1.2, linestyle="none", label=DISPLAY_LABELS[method],
        )
        axes[1].errorbar(
            x + offsets[method], rte,
            yerr=np.maximum(0.0, np.vstack((rte - rte_low, rte_high - rte))),
            fmt=MARKERS[method], markersize=7.0, color=COLORS[method],
            markeredgecolor="white", markeredgewidth=0.7,
            capsize=3.2, elinewidth=1.2, linestyle="none", label=DISPLAY_LABELS[method],
        )

    labels = [label for _, _, label in SCENARIOS]
    axes[0].set(
        title="(a) 双车同时失效后的恢复成功率",
        ylabel="恢复成功率",
        ylim=(-0.06, 1.10),
        xticks=x,
        xticklabels=labels,
    )
    axes[1].set(
        title="(b) 固定窗口任务韧性",
        ylabel="$R_{T_e}$",
        ylim=(0.20, 0.66),
        xticks=x,
        xticklabels=labels,
    )
    for axis in axes:
        axis.set_xlabel("编队构型—故障类型（A：相邻，N：不相邻）")
        axis.grid(axis="y")
        axis.axvline(1.5, color="#C9CFCC", linewidth=0.8)
        axis.axvline(3.5, color="#C9CFCC", linewidth=0.8)
    axes[0].legend(
        ncol=3, loc="upper center", bbox_to_anchor=(1.04, 1.25),
        columnspacing=1.8, handletextpad=0.5,
    )
    fig.suptitle(
        "V2.1-G7预实验：相邻与不相邻双车同时失效（5个配对网络种子）",
        fontsize=14.0, color="#3F4D47",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "G7_Pilot_Fig1_simultaneous_failure_statistics.png",
        output_dir / "G7_Pilot_Fig1_simultaneous_failure_statistics.pdf",
    ]
    fig.savefig(outputs[0])
    fig.savefig(outputs[1])
    plt.close(fig)
    if not summary["success_rate_is_informative"]:
        raise RuntimeError("pilot selected recovery time; this success-rate candidate must be revised")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/v2_1/g7_pilot"))
    args = parser.parse_args()
    style()
    outputs = plot(args.run_dir, args.output_dir)
    print(json.dumps({"figures": [str(path) for path in outputs]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
