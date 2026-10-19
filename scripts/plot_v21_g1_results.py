from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


COLORS = {
    "green": "#6F857A",
    "green_light": "#AAB9B0",
    "blue": "#8299AC",
    "orange": "#C58F79",
    "gray": "#9D9992",
    "dark": "#3F4D47",
    "red": "#B65F59",
    "paper": "#F7F7F4",
}


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


def load_records(run_dir: Path) -> tuple[list[dict], dict]:
    with (run_dir / "topology_records.csv").open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return records, summary


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png")
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def edge_budget_scan(records: list[dict], summary: dict, output_dir: Path) -> None:
    random_by_edge: dict[int, list[dict]] = defaultdict(list)
    s1_by_edge: dict[int, dict] = {}
    for record in records:
        if record["scope"] != "anchor":
            continue
        edge_count = int(record["edge_count"])
        if record["method"] == "B1_random_connected":
            random_by_edge[edge_count].append(record)
        elif record["method"] == "S1_worst_residual_greedy":
            s1_by_edge[edge_count] = record
    edge_counts = sorted(random_by_edge)
    connected = [float(summary["anchor_random"][str(edge)]["all_scenario_connected_rate"]) for edge in edge_counts]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), constrained_layout=True)

    axes[0].plot(edge_counts, connected, color=COLORS["green"], marker="o", linewidth=2.2)
    axes[0].fill_between(edge_counts, connected, color=COLORS["green_light"], alpha=0.28)
    for x, value in zip(edge_counts, connected, strict=True):
        axes[0].annotate(f"{100*value:.1f}%", (x, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
    axes[0].set(title="(a) 随机拓扑完全存活概率", xlabel="通信边数 |E|", ylabel="37种故障场景全部连通的比例")
    axes[0].set_xticks(edge_counts)
    axes[0].set_ylim(-0.025, 0.405)
    axes[0].grid(axis="y")

    lambda_data = [[float(row["worst_lambda2"]) for row in random_by_edge[edge]] for edge in edge_counts]
    boxes = axes[1].boxplot(lambda_data, positions=edge_counts, widths=0.55, patch_artist=True, showfliers=False)
    for box in boxes["boxes"]:
        box.set(facecolor=COLORS["green_light"], edgecolor=COLORS["green"], alpha=0.75)
    for element in ("whiskers", "caps", "medians"):
        for artist in boxes[element]:
            artist.set(color=COLORS["dark"], linewidth=1.1)
    s1_lambda = [float(s1_by_edge[edge]["worst_lambda2"]) for edge in edge_counts]
    axes[1].plot(edge_counts, s1_lambda, color=COLORS["orange"], marker="D", linewidth=2.0, label="S1认证候选")
    exact = float(summary["exact_enumeration"]["optimal_worst_lambda2"])
    axes[1].scatter([12], [exact], color=COLORS["red"], marker="*", s=150, zorder=5, label="E=12精确最优")
    axes[1].annotate(f"精确最优 {exact:.3f}", (12, exact), xytext=(12, exact + 0.16), ha="center", color=COLORS["red"], arrowprops={"arrowstyle": "-", "color": COLORS["red"]})
    axes[1].set(title="(b) 最坏残余代数连通度", xlabel="通信边数 |E|", ylabel=r"$\lambda_2^{\mathrm{worst}}$")
    axes[1].set_xticks(edge_counts)
    axes[1].grid(axis="y")
    axes[1].legend(loc="upper left")

    sc_data = [[float(row["S_c_min"]) for row in random_by_edge[edge]] for edge in edge_counts]
    boxes = axes[2].boxplot(sc_data, positions=edge_counts, widths=0.55, patch_artist=True, showfliers=False)
    for box in boxes["boxes"]:
        box.set(facecolor=COLORS["blue"], edgecolor=COLORS["dark"], alpha=0.58)
    for element in ("whiskers", "caps", "medians"):
        for artist in boxes[element]:
            artist.set(color=COLORS["dark"], linewidth=1.1)
    s1_sc = [float(s1_by_edge[edge]["S_c_min"]) for edge in edge_counts]
    axes[2].plot(edge_counts, s1_sc, color=COLORS["orange"], marker="D", linewidth=2.0, label="S1认证候选")
    axes[2].set(title="(c) 最小最大连通分量比例", xlabel="通信边数 |E|", ylabel=r"$S_c^{\min}$")
    axes[2].set_xticks(edge_counts)
    axes[2].set_ylim(0.1, 1.06)
    axes[2].grid(axis="y")
    axes[2].legend(loc="lower right")
    fig.suptitle("V2.1-G1：边预算、随机拓扑分布与Stage 1结构保证", fontsize=15, color=COLORS["dark"])
    save(fig, output_dir, "G1_Fig1_edge_budget_scan")


def method_comparison(records: list[dict], summary: dict, output_dir: Path) -> None:
    selected = {
        row["method"]: row for row in records
        if row["scope"] == "anchor" and int(row["edge_count"]) == 12 and row["graph_seed"] == ""
    }
    labels = ["B2\n循环三正则", "B3\n名义 $\\lambda_2$", "S1\n残余优化", "精确最优"]
    rows = [
        selected["B2_circulant_regular"],
        selected["B3_nominal_lambda2_greedy"],
        selected["S1_worst_residual_greedy"],
    ]
    exact_value = float(summary["exact_enumeration"]["optimal_worst_lambda2"])
    metrics = [
        ("37种场景连通率", [float(row["connectivity_rate"]) for row in rows] + [1.0], (0.88, 1.01), lambda value: f"{100*value:.1f}%"),
        (r"最坏残余 $\lambda_2$", [float(row["worst_lambda2"]) for row in rows] + [exact_value], (0.0, 0.72), lambda value: f"{value:.3f}"),
        (r"最小 $S_c$", [float(row["S_c_min"]) for row in rows] + [1.0], (0.78, 1.02), lambda value: f"{value:.3f}"),
    ]
    bar_colors = [COLORS["gray"], COLORS["orange"], COLORS["green"], COLORS["blue"]]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)
    for axis, (title, values, limits, formatter) in zip(axes, metrics, strict=True):
        bars = axis.bar(np.arange(4), values, color=bar_colors, width=0.67, edgecolor="white", linewidth=0.8)
        axis.set_xticks(np.arange(4), labels)
        axis.set_ylim(*limits)
        axis.set_title(title)
        axis.grid(axis="y")
        for bar, value in zip(bars, values, strict=True):
            axis.annotate(formatter(value), (bar.get_x() + bar.get_width()/2, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9.5)
    axes[1].annotate("B2、S1与精确最优并列", xy=(2.5, exact_value), xytext=(1.6, 0.69), ha="center", color=COLORS["red"], arrowprops={"arrowstyle": "->", "color": COLORS["red"]})
    fig.suptitle("V2.1-G1：N=8、|E|=12、f=2的同成本方法比较", fontsize=15, color=COLORS["dark"])
    save(fig, output_dir, "G1_Fig2_E12_method_comparison")


def topology_figure(summary: dict, output_dir: Path) -> None:
    edges = [tuple(edge) for edge in summary["exact_enumeration"]["edges"]]
    failed = set(summary["exact_enumeration"]["certificate"]["worst_failure"])
    n = int(summary["exact_enumeration"]["n"])
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n, endpoint=False)
    positions = {node: (np.cos(angles[node - 1]), np.sin(angles[node - 1])) for node in range(1, n + 1)}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.3), constrained_layout=True)

    def draw(axis, residual: bool) -> None:
        active = set(range(1, n + 1)) - failed if residual else set(range(1, n + 1))
        for i, j in edges:
            xi, yi = positions[i]
            xj, yj = positions[j]
            if residual and (i not in active or j not in active):
                continue
            axis.plot([xi, xj], [yi, yj], color=COLORS["blue"] if residual else COLORS["gray"], linewidth=2.0, zorder=1)
        for node, (x, y) in positions.items():
            if residual and node in failed:
                axis.scatter(x, y, s=520, facecolor="none", edgecolor=COLORS["red"], linewidth=2.0, linestyle="--", zorder=3)
                axis.text(x, y, "×", color=COLORS["red"], ha="center", va="center", fontsize=22, zorder=4)
                continue
            color = COLORS["orange"] if (not residual and node in failed) else COLORS["green"]
            axis.scatter(x, y, s=560, color=color, edgecolor="white", linewidth=1.4, zorder=3)
            axis.text(x, y, str(node), color="white", ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)
        axis.set_aspect("equal")
        axis.set_xlim(-1.35, 1.35)
        axis.set_ylim(-1.35, 1.35)
        axis.axis("off")

    draw(axes[0], residual=False)
    axes[0].set_title("(a) 精确最优12边三正则拓扑\n橙色：最坏故障节点5和8")
    draw(axes[1], residual=True)
    axes[1].set_title(r"(b) 删除节点5和8后的残余网络" "\n" r"仍连通，$\lambda_2=0.6571$，$S_c=1$")
    fig.suptitle("V2.1-G1：精确最优拓扑及其最坏双节点故障", fontsize=15, color=COLORS["dark"])
    save(fig, output_dir, "G1_Fig3_exact_topology_and_worst_failure")


def scale_extension(summary: dict, output_dir: Path) -> None:
    nodes, degrees, failures = [8, 12, 16], [3, 4, 5], [1, 2]
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.2), constrained_layout=True)
    for column, failure in enumerate(failures):
        lambda_matrix = np.zeros((len(nodes), len(degrees)))
        survival_matrix = np.zeros_like(lambda_matrix)
        for row, n in enumerate(nodes):
            for col, degree in enumerate(degrees):
                cell = summary["scale_random"][f"N{n}_d{degree}_f{failure}"]
                lambda_matrix[row, col] = float(cell["worst_lambda2"]["median"])
                survival_matrix[row, col] = float(cell["all_scenario_connected_rate"])
        for axis, matrix, title, fmt, vmax in [
            (axes[0, column], lambda_matrix, fr"f={failure}：随机拓扑最坏 $\lambda_2$ 中位数", ".3f", max(0.7, float(lambda_matrix.max()))),
            (axes[1, column], survival_matrix, f"f={failure}：全部故障场景保持连通的比例", ".0%", 1.0),
        ]:
            image = axis.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=vmax, aspect="auto")
            axis.set_xticks(range(len(degrees)), degrees)
            axis.set_yticks(range(len(nodes)), nodes)
            axis.set_xlabel(r"平均度 $\bar d$")
            axis.set_ylabel("节点数 N")
            axis.set_title(title)
            for row in range(len(nodes)):
                for col in range(len(degrees)):
                    value = matrix[row, col]
                    label = format(value, fmt)
                    axis.text(col, row, label, ha="center", va="center", color="white" if value > 0.55 * vmax else COLORS["dark"], fontsize=10)
            fig.colorbar(image, ax=axis, shrink=0.80)
    fig.suptitle("V2.1-G1：随机拓扑的规模、平均度与故障预算效应", fontsize=15, color=COLORS["dark"])
    save(fig, output_dir, "G1_Fig4_scale_extension")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("figures") / "v2_1" / "g1")
    args = parser.parse_args()
    configure_style()
    records, summary = load_records(args.run_dir)
    edge_budget_scan(records, summary, args.output_dir)
    method_comparison(records, summary, args.output_dir)
    topology_figure(summary, args.output_dir)
    scale_extension(summary, args.output_dir)
    print(json.dumps({"figures": sorted(str(path) for path in args.output_dir.glob("G1_Fig*"))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
