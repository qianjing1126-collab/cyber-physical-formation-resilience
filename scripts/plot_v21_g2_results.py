from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "B1": "#9D9992",
    "B3": "#C58F79",
    "S1": "#6F857A",
    "blue": "#8299AC",
    "green_light": "#AAB9B0",
    "dark": "#3F4D47",
    "red": "#B65F59",
    "gold": "#B79B6C",
}
METHOD_LABELS = {"B1": "RG 随机拓扑", "B3": r"NC 名义 $\lambda_2$ 拓扑", "S1": "S1 残余优化拓扑"}


def configure_style() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10.5,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": COLORS["dark"],
        "axes.linewidth": 0.9,
        "grid.color": "#D9DEDA",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.75,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 240,
    })


def read_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric = {
        "case_id", "residual_lambda2", "residual_S_c", "e2e_reachability",
        "effective_e2e_PDR", "AoI_p95_e2e_censored", "position_rmse",
        "velocity_rmse", "Q_min", "R_Te", "R_mission", "T_r_penalized",
        "coordination_plan_issued", "recovery_success", "safety_ok",
    }
    for row in rows:
        for key in numeric:
            if row.get(key, "") != "":
                row[key] = float(row[key])
    return rows


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png")
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def method_overview(records: list[dict], summary: dict, output_dir: Path) -> None:
    methods = ["B1", "B3", "S1"]
    grouped = {method: [row for row in records if row["method"] == method] for method in methods}
    colors = [COLORS[method] for method in methods]
    labels = [METHOD_LABELS[method] for method in methods]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 4, figsize=(16.2, 4.7), constrained_layout=True)

    lambda_data = [[row["residual_lambda2"] for row in grouped[method]] for method in methods]
    boxes = axes[0].boxplot(lambda_data, positions=x, widths=0.58, patch_artist=True, showfliers=False)
    for box, color in zip(boxes["boxes"], colors, strict=True):
        box.set(facecolor=color, edgecolor="white", alpha=0.88)
    for element in ("whiskers", "caps", "medians"):
        for artist in boxes[element]:
            artist.set(color=COLORS["dark"], linewidth=1.1)
    axes[0].set(title=r"(a) 故障后残余连通度", ylabel=r"残余代数连通度 $\lambda_2$")

    panels = [
        ("e2e_reachability", "(b) 端到端可达率", "平均可达率"),
        ("coordination_plan_issued", "(c) 恢复方案下发率", "下发成功率"),
        ("R_Te", r"(d) 固定窗韧性 $R_{T_e}$", r"$R_{T_e}$"),
    ]
    for axis, (metric, title, ylabel) in zip(axes[1:], panels, strict=True):
        values = [summary["method_summary"][method][metric]["mean"] for method in methods]
        bars = axis.bar(x, values, color=colors, width=0.64, edgecolor="white")
        axis.set(title=title, ylabel=ylabel)
        floor = 0.55 if metric == "R_Te" else 0.68
        axis.set_ylim(floor, 1.035 if metric != "R_Te" else 0.69)
        for bar, value in zip(bars, values, strict=True):
            text = f"{100*value:.1f}%" if metric != "R_Te" else f"{value:.3f}"
            axis.annotate(text, (bar.get_x() + bar.get_width()/2, value), xytext=(0, 5),
                          textcoords="offset points", ha="center", fontsize=9.5)
        axis.grid(axis="y")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=0)
        axis.grid(axis="y")
    fig.suptitle("V2.1-G2：相同双车故障下的跨层结果（100组严格配对，作业速度6 m/s）",
                 fontsize=15, color=COLORS["dark"])
    save(fig, output_dir, "G2_Fig1_cross_layer_method_overview")


def paired_effects(summary: dict, output_dir: Path) -> None:
    effects = summary["paired_effects"]
    comparisons = [("S1_minus_B1", "S1 − RG", COLORS["B1"]), ("S1_minus_B3", "S1 − NC", COLORS["B3"])]
    definitions = [
        ("e2e_reachability", "(a) 端到端可达率提升", 100.0, "百分点", False),
        ("effective_e2e_PDR", "(b) 有效端到端PDR提升", 100.0, "百分点", False),
        ("AoI_p95_e2e_censored", "(c) 端到端AoI降低", -1.0, "s", True),
        ("coordination_plan_issued", "(d) 恢复方案下发率提升", 100.0, "百分点", False),
        ("R_Te", r"(e) 固定窗韧性 $R_{T_e}$ 提升", 100.0, "百分点", False),
        ("T_r_penalized", "(f) 惩罚恢复时间缩短", -1.0, "s", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.0), constrained_layout=True)
    for axis, (metric, title, scale, unit, reduction) in zip(axes.flat, definitions, strict=True):
        y = np.array([1.0, 0.0])
        means, lows, highs = [], [], []
        for comparison, _, _ in comparisons:
            item = effects[comparison][metric]
            values = np.array([item["mean_delta"], item["ci95_low"], item["ci95_high"]]) * scale
            means.append(values[0])
            lows.append(min(values[1], values[2]))
            highs.append(max(values[1], values[2]))
        means = np.asarray(means)
        errors = np.vstack([means - np.asarray(lows), np.asarray(highs) - means])
        for index, (_, label, color) in enumerate(comparisons):
            axis.errorbar(means[index], y[index], xerr=errors[:, index:index+1], fmt="o", markersize=8,
                          color=color, ecolor=color, elinewidth=2, capsize=4, label=label)
            axis.annotate(f"{means[index]:.2f} {unit}", (means[index], y[index]), xytext=(7, 7),
                          textcoords="offset points", fontsize=9)
        axis.axvline(0, color=COLORS["dark"], linewidth=0.9, linestyle="--")
        axis.set_yticks(y, [item[1] for item in comparisons])
        axis.set_xlabel(("降低量 / " if reduction else "提升量 / ") + unit)
        axis.set_title(title)
        axis.grid(axis="x")
    fig.suptitle("V2.1-G2：S1相对基线的配对效应及95%置信区间",
                 fontsize=15, color=COLORS["dark"])
    save(fig, output_dir, "G2_Fig2_paired_effects_CI")


def cross_layer_chain(records: list[dict], summary: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    for method in ("B1", "B3", "S1"):
        rows = [row for row in records if row["method"] == method]
        axes[0].scatter([row["residual_lambda2"] for row in rows],
                        [row["effective_e2e_PDR"] for row in rows],
                        s=25, alpha=0.62, color=COLORS[method], label=METHOD_LABELS[method])
    correlation = summary["cross_layer_correlations"]["structure_to_network"]
    axes[0].set(title=r"(a) 结构层 $\rightarrow$ 网络层", xlabel=r"残余 $\lambda_2$", ylabel="有效端到端PDR")
    axes[0].text(0.04, 0.07, fr"Spearman $\rho$={correlation['spearman_rho']:.3f}", transform=axes[0].transAxes,
                 bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": COLORS["green_light"]})
    axes[0].legend(fontsize=9)

    issued_groups = [[row["effective_e2e_PDR"] for row in records if row["coordination_plan_issued"] == value] for value in (0.0, 1.0)]
    boxes = axes[1].boxplot(issued_groups, positions=[0, 1], widths=0.55, patch_artist=True, showfliers=False)
    for box, color in zip(boxes["boxes"], [COLORS["red"], COLORS["green_light"]], strict=True):
        box.set(facecolor=color, edgecolor="white", alpha=0.82)
    axes[1].set(title=r"(b) 网络层 $\rightarrow$ 协调层", ylabel="有效端到端PDR")
    axes[1].set_xticks([0, 1], ["未能下发\n恢复方案", "成功下发\n恢复方案"])
    correlation = summary["cross_layer_correlations"]["network_to_coordination"]
    axes[1].text(0.04, 0.07, fr"Spearman $\rho$={correlation['spearman_rho']:.3f}", transform=axes[1].transAxes,
                 bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": COLORS["green_light"]})

    task_groups = [[row["R_Te"] for row in records if row["coordination_plan_issued"] == value] for value in (0.0, 1.0)]
    boxes = axes[2].boxplot(task_groups, positions=[0, 1], widths=0.55, patch_artist=True, showfliers=False)
    for box, color in zip(boxes["boxes"], [COLORS["red"], COLORS["S1"]], strict=True):
        box.set(facecolor=color, edgecolor="white", alpha=0.82)
    axes[2].set(title=r"(c) 协调层 $\rightarrow$ 任务层", ylabel=r"固定窗韧性 $R_{T_e}$")
    axes[2].set_xticks([0, 1], ["未能下发\n恢复方案", "成功下发\n恢复方案"])
    correlation = summary["cross_layer_correlations"]["coordination_to_task"]
    axes[2].text(0.04, 0.07, fr"Spearman $\rho$={correlation['spearman_rho']:.3f}", transform=axes[2].transAxes,
                 bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": COLORS["green_light"]})
    for axis in axes:
        axis.grid(axis="y")
    fig.suptitle("V2.1-G2：拓扑结构如何传导到任务韧性",
                 fontsize=15, color=COLORS["dark"])
    save(fig, output_dir, "G2_Fig3_cross_layer_evidence_chain")


def read_timeseries(project_root: Path, run_id: str) -> dict[str, np.ndarray]:
    path = project_root / "runs" / "raw" / run_id / "metrics_timeseries.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.asarray([float(row[key]) for row in rows]) for key in ("t", "Q", "Q_cap", "Q_rec")}


def representative_curve(records: list[dict], project_root: Path, output_dir: Path) -> None:
    by_case: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in records:
        by_case[int(row["case_id"])][row["method"]] = row
    selected = None
    for case_id, methods in sorted(by_case.items()):
        if (methods["B1"]["scientific_status"] == "COMMUNICATION_INFEASIBLE"
                and methods["B3"]["scientific_status"] == "COMMUNICATION_INFEASIBLE"
                and methods["S1"]["scientific_status"] == "COMPLETE"):
            selected = (case_id, methods)
            break
    if selected is None:
        raise RuntimeError("no representative paired case found")
    case_id, methods = selected
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.9), constrained_layout=True)
    for method in ("B1", "B3", "S1"):
        series = read_timeseries(project_root, methods[method]["run_id"])
        axes[0].plot(series["t"], series["Q"], color=COLORS[method], linewidth=2.1, label=METHOD_LABELS[method])
        axes[1].plot(series["t"], series["Q_rec"], color=COLORS[method], linewidth=2.1, label=METHOD_LABELS[method])
    s1_series = read_timeseries(project_root, methods["S1"]["run_id"])
    axes[0].plot(s1_series["t"], s1_series["Q_cap"], color=COLORS["gold"], linestyle="--", linewidth=1.6,
                 label="故障后容量上限")
    for axis in axes:
        axis.axvline(15.0, color=COLORS["red"], linestyle="--", linewidth=1.2)
        axis.axvline(15.5, color=COLORS["blue"], linestyle=":", linewidth=1.2)
        axis.axvspan(15.0, 15.5, color=COLORS["red"], alpha=0.08)
        axis.set_xlim(0, 35)
        axis.set_xlabel("时间 / s")
        axis.grid(axis="both")
    axes[0].set(title="(a) 容量归一化任务服务", ylabel=r"$Q(t)$")
    axes[1].set(title="(b) 残余容量归一化恢复程度", ylabel=r"$Q_{rec}(t)$")
    axes[0].legend(loc="lower left", fontsize=9)
    axes[1].annotate("15 s：车辆3、7同时失效", xy=(15.0, 0.51), xytext=(2.0, 0.64),
                     arrowprops={"arrowstyle": "->", "color": COLORS["red"]}, color=COLORS["red"])
    axes[1].annotate("S1恢复至残余容量上限", xy=(25.0, 1.0), xytext=(19.0, 0.83),
                     arrowprops={"arrowstyle": "->", "color": COLORS["S1"]}, color=COLORS["dark"])
    failures = methods["S1"]["failure_targets"]
    fig.suptitle(f"V2.1-G2：代表性严格配对案例（案例{case_id}，故障节点{failures}）",
                 fontsize=15, color=COLORS["dark"])
    save(fig, output_dir, "G2_Fig4_representative_mission_curve")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("figures") / "v2_1" / "g2")
    args = parser.parse_args()
    configure_style()
    records = read_records(args.analysis_dir / "paired_records.csv")
    summary = json.loads((args.analysis_dir / "summary.json").read_text(encoding="utf-8"))
    method_overview(records, summary, args.output_dir)
    paired_effects(summary, args.output_dir)
    cross_layer_chain(records, summary, args.output_dir)
    representative_curve(records, args.project_root.resolve(), args.output_dir)
    print(json.dumps({"figures": sorted(str(path) for path in args.output_dir.glob("G2_Fig*"))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
