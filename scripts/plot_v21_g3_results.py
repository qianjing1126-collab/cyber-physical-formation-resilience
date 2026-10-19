from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "green": "#6F857A",
    "green_light": "#AAB9B0",
    "blue": "#8299AC",
    "orange": "#C58F79",
    "gray": "#9D9992",
    "dark": "#3F4D47",
    "red": "#B65F59",
    "gold": "#B79B6C",
}
METHOD_COLORS = {"B7_uniform": COLORS["gray"], "B8_convergence_priority": COLORS["orange"], "S2_delay_constrained": COLORS["green"]}
METHOD_LABELS = {"B7_uniform": "B3 均匀权重", "B8_convergence_priority": "B4 收敛优先", "S2_delay_constrained": "S2 时延约束"}


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


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png")
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def pareto_main(run_dir: Path, summary: dict, output_dir: Path) -> None:
    cloud = read_csv(run_dir / "candidate_cloud.csv")
    frontier = read_csv(run_dir / "pareto_frontier.csv")
    x = np.asarray([float(row["worst_lambda_max"]) for row in cloud])
    y = np.asarray([float(row["worst_lambda2"]) for row in cloud])
    fx = np.asarray([float(row["worst_lambda_max"]) for row in frontier])
    fy = np.asarray([float(row["worst_lambda2"]) for row in frontier])
    fig, axis = plt.subplots(figsize=(8.6, 6.2), constrained_layout=True)
    axis.scatter(x, y, s=8, alpha=0.10, color=COLORS["blue"], rasterized=True,
                 label="20,000组随机可行权重")
    axis.plot(fx, fy, color=COLORS["green"], linewidth=2.5, marker="o", markersize=4.5,
              label="稳健SDP前沿")
    for method in summary["methods"]:
        axis.scatter(method["worst_lambda_max"], method["worst_lambda2"], s=130,
                     color=METHOD_COLORS[method["method"]], edgecolor="white", linewidth=1.2,
                     marker="*" if method["method"].startswith("S2") else "D", zorder=5,
                     label=METHOD_LABELS[method["method"]])
    selected = summary["selection"]["primary"]
    target_ms = 1000 * summary["selection"]["delay_requirement"]["target_delay_margin_s"]
    axis.annotate(fr"S2：$\tau_{{req}}={target_ms:.0f}$ ms" +
                  fr"，$\bar{{\lambda}}={summary['selection']['lambda_cap']:.3f}$" +
                  "\n由信息新鲜度要求反算，并非事后膝点",
                  xy=(selected["worst_lambda_max"], selected["worst_lambda2"]),
                  xytext=(selected["worst_lambda_max"] + 0.48, selected["worst_lambda2"] - 0.015),
                  arrowprops={"arrowstyle": "->", "color": COLORS["green"]}, color=COLORS["dark"])
    axis.set(xlabel=r"最坏残余 $\lambda_{max}$（时延敏感性）",
             ylabel=r"最坏残余 $\lambda_2$（协调能力下界）",
             title="V2.1-G3：随机权重候选云与稳健Pareto前沿")
    axis.grid()
    axis.legend(loc="upper left", fontsize=9)
    save(fig, output_dir, "G3_Fig1_weight_cloud_and_pareto_frontier")


def candidate_cloud_supplement(run_dir: Path, summary: dict, output_dir: Path) -> None:
    cloud = read_csv(run_dir / "candidate_cloud.csv")
    frontier = read_csv(run_dir / "pareto_frontier.csv")
    x = np.asarray([float(row["worst_lambda_max"]) for row in cloud])
    y = np.asarray([float(row["worst_lambda2"]) for row in cloud])
    fx = np.asarray([float(row["worst_lambda_max"]) for row in frontier])
    fy = np.asarray([float(row["worst_lambda2"]) for row in frontier])
    fig, axis = plt.subplots(figsize=(8.6, 6.2), constrained_layout=True)
    axis.scatter(x, y, s=8, alpha=0.10, color=COLORS["blue"], rasterized=True, label="20,000组随机可行权重")
    axis.plot(fx, fy, color=COLORS["green"], linewidth=2.5, marker="o", markersize=4.5, label="稳健SDP前沿")
    for method in summary["methods"]:
        axis.scatter(method["worst_lambda_max"], method["worst_lambda2"], s=130,
                     color=METHOD_COLORS[method["method"]], edgecolor="white", linewidth=1.2,
                     marker="*" if method["method"].startswith("S2") else "D", zorder=5,
                     label=METHOD_LABELS[method["method"]])
    selected = summary["selection"]["primary"]
    target_ms = 1000 * summary["selection"]["delay_requirement"]["target_delay_margin_s"]
    axis.annotate(fr"S2：$\tau_{{req}}={target_ms:.0f}$ ms" +
                  fr"，$\bar{{\lambda}}={summary['selection']['lambda_cap']:.3f}$" +
                  "\n由信息新鲜度要求反算，并非事后膝点",
                  xy=(selected["worst_lambda_max"], selected["worst_lambda2"]),
                  xytext=(selected["worst_lambda_max"] + 0.48, selected["worst_lambda2"] - 0.015),
                  arrowprops={"arrowstyle": "->", "color": COLORS["green"]}, color=COLORS["dark"])
    axis.set(xlabel=r"最坏残余 $\lambda_{max}$（时延敏感性）",
             ylabel=r"最坏残余 $\lambda_2$（协调能力下界）",
             title="V2.1-G3补充证据：20,000组随机权重候选云")
    axis.grid()
    axis.legend(loc="upper left", fontsize=9)
    save(fig, output_dir, "G3_FigS1_candidate_cloud_supplement")


def constraint_sweep(run_dir: Path, summary: dict, output_dir: Path) -> None:
    frontier = read_csv(run_dir / "pareto_frontier.csv")
    delay_ms = np.asarray([1000 * float(row["delay_margin_s"]) for row in frontier])
    gamma = np.asarray([float(row["worst_lambda2"]) for row in frontier])
    caps = np.asarray([float(row["lambda_cap"]) for row in frontier])
    order = np.argsort(delay_ms)
    fig, axis = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
    axis.plot(delay_ms[order], gamma[order], color=COLORS["green"], linewidth=2.5, marker="o", markersize=5)
    for index in order[::3]:
        axis.annotate(fr"上界 {caps[index]:.2g}", (delay_ms[index], gamma[index]), xytext=(5, 6),
                      textcoords="offset points", fontsize=8.5, color=COLORS["dark"])
    selected = summary["selection"]["primary"]
    requirement = summary["selection"]["delay_requirement"]
    axis.scatter(1000 * selected["delay_margin_s"], selected["worst_lambda2"], marker="*", s=210,
                 color=COLORS["red"], edgecolor="white", linewidth=1.0, zorder=5, label="正式S2权重")
    axis.axvline(1000 * selected["delay_margin_s"], color=COLORS["red"], linestyle="--", linewidth=1.1)
    axis.annotate(
                  "信息新鲜度要求\n"
                  f"{1000*requirement['state_message_period_s']:.0f}+"
                  f"{1000*requirement['control_period_s']:.0f}+"
                  f"{1000*requirement['admitted_network_tail_s']:.0f}+"
                  f"{1000*requirement['engineering_allowance_s']:.0f}="
                  f"{1000*requirement['target_delay_margin_s']:.0f} ms",
                  xy=(1000 * selected["delay_margin_s"], selected["worst_lambda2"]),
                  xytext=(225, 0.39), arrowprops={"arrowstyle": "->", "color": COLORS["red"]},
                  color=COLORS["dark"], fontsize=10.5)
    axis.set(xlabel="理论统一时延裕度 / ms", ylabel=r"最坏残余 $\lambda_2$",
             title="V2.1-G3：时延要求与协调能力的约束扫描")
    axis.grid()
    axis.legend(loc="lower left")
    save(fig, output_dir, "G3_Fig2_delay_margin_constraint_sweep")


def _spring_positions(n: int, edges: list[tuple[int, int]]) -> dict[int, np.ndarray]:
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    positions = np.column_stack([np.cos(angles), np.sin(angles)])
    k = np.sqrt(4.0 / n)
    temperature = 0.20
    for _ in range(450):
        displacement = np.zeros_like(positions)
        for i in range(n):
            for j in range(i + 1, n):
                delta = positions[i] - positions[j]
                distance = max(np.linalg.norm(delta), 1e-5)
                force = k*k / distance
                displacement[i] += delta / distance * force
                displacement[j] -= delta / distance * force
        for first, second in edges:
            i, j = first - 1, second - 1
            delta = positions[i] - positions[j]
            distance = max(np.linalg.norm(delta), 1e-5)
            force = distance*distance / k
            displacement[i] -= delta / distance * force
            displacement[j] += delta / distance * force
        norms = np.linalg.norm(displacement, axis=1)
        positions += displacement / np.maximum(norms[:, None], 1e-9) * np.minimum(norms[:, None], temperature)
        positions -= positions.mean(axis=0)
        temperature *= 0.992
    scale = np.max(np.linalg.norm(positions, axis=1))
    positions /= scale
    return {node: positions[node - 1] for node in range(1, n + 1)}


def weight_topology(summary: dict, output_dir: Path) -> None:
    source_edges = [tuple(edge) for edge in summary["topology"]["edges"]]
    # The frozen topology is the original manuscript's pre-specified
    # three-regular representative, so its agent identities and circular
    # arrangement are preserved directly.
    display_label = {node: node for node in range(1, 9)}
    circular_order = (2, 1, 8, 7, 6, 5, 4, 3)
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2*np.pi, len(circular_order), endpoint=False)
    positions = {
        node: np.asarray([0.88*np.cos(angle), 0.88*np.sin(angle)])
        for node, angle in zip(circular_order, angles, strict=True)
    }
    methods = {item["method"]: item for item in summary["methods"]}
    method = methods["S2_delay_constrained"]
    values = method["weights"]
    displayed_edges: list[tuple[int, int, float]] = []
    for first, second in source_edges:
        displayed_edges.append((display_label[first], display_label[second],
                                float(values[f"{first}-{second}"])))
    weights = np.asarray([weight for _, _, weight in displayed_edges])
    low, high = float(weights.min()), float(weights.max())
    light_rgb = np.asarray([0xAF, 0xBD, 0xB6]) / 255.0
    dark_rgb = np.asarray([0x2F, 0x79, 0x6B]) / 255.0
    fig, axis = plt.subplots(figsize=(7.2, 6.1), constrained_layout=True)
    for edge_index, (first, second, weight) in enumerate(displayed_edges):
        fraction = (weight - low) / max(high - low, 1e-12)
        color = light_rgb * (1.0 - fraction) + dark_rgb * fraction
        width = 2.2 + 3.3 * fraction
        start, end = positions[first], positions[second]
        axis.plot([start[0], end[0]], [start[1], end[1]], color=color,
                  linewidth=width, alpha=0.94, solid_capstyle="round", zorder=1)
        midpoint = 0.5 * (start + end)
        direction = end - start
        normal = np.asarray([-direction[1], direction[0]])
        normal /= max(np.linalg.norm(normal), 1e-12)
        label_point = midpoint + normal * (0.035 if edge_index % 2 == 0 else -0.035)
        axis.text(label_point[0], label_point[1], f"{weight:.3f}", ha="center", va="center", fontsize=8.5,
                  color=COLORS["dark"], bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 0.6}, zorder=2)
    for node, point in positions.items():
        axis.scatter(point[0], point[1], s=1120, color="#3E8E7E", edgecolor="white", linewidth=2.0, zorder=3)
        axis.text(point[0], point[1], f"Agent {node}", ha="center", va="center", color="white", fontsize=8.8,
                  fontweight="bold", zorder=4)
    axis.plot([], [], color=light_rgb, linewidth=2.2, label=f"较低权重 {low:.3f}")
    axis.plot([], [], color=dark_rgb, linewidth=5.5, label=f"较高权重 {high:.3f}")
    axis.legend(loc="lower center", ncols=2, fontsize=9.5)
    axis.text(-1.25, 1.16, "S2 weighted topology:", fontsize=11, color=COLORS["dark"])
    axis.text(-1.25, 1.05, r"$N=8,\ |E|=12,\ d_i=3$", fontsize=10.5, color=COLORS["dark"])
    axis.text(-1.25, 0.94, f"delay margin = {1000*method['delay_margin_s']:.1f} ms",
              fontsize=10.0, color=COLORS["dark"])
    axis.set_aspect("equal")
    axis.set_xlim(-1.30, 1.30)
    axis.set_ylim(-1.15, 1.22)
    axis.axis("off")
    save(fig, output_dir, "G3_Fig3_selected_weight_topology")


def method_comparison(summary: dict, output_dir: Path) -> None:
    methods = summary["methods"]
    labels = [METHOD_LABELS[item["method"]] for item in methods]
    colors = [METHOD_COLORS[item["method"]] for item in methods]
    panels = [
        ("worst_lambda2", r"(a) 最坏残余 $\lambda_2$", r"$\lambda_2^{worst}$", 1.0, "{:.3f}"),
        ("worst_lambda_max", r"(b) 最坏残余 $\lambda_{max}$", r"$\lambda_{max}^{worst}$", 1.0, "{:.3f}"),
        ("delay_margin_s", "(c) 理论统一时延裕度", "时延裕度 / ms", 1000.0, "{:.1f}"),
        ("weight_sum", "(d) 总协调耦合权重", r"$\sum w_{ij}$", 1.0, "{:.3f}"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.8), constrained_layout=True)
    for axis, (metric, title, ylabel, scale, formatter) in zip(axes, panels, strict=True):
        values = [float(item[metric]) * scale for item in methods]
        bars = axis.bar(np.arange(3), values, color=colors, width=0.64, edgecolor="white")
        axis.set_xticks(np.arange(3), labels)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, max(values) * 1.20)
        axis.grid(axis="y")
        for bar, value in zip(bars, values, strict=True):
            axis.annotate(formatter.format(value), (bar.get_x() + bar.get_width()/2, value), xytext=(0, 5),
                          textcoords="offset points", ha="center", fontsize=9.5)
    fig.suptitle("V2.1-G3：均匀、收敛优先与时延约束权重的同拓扑比较", fontsize=15, color=COLORS["dark"])
    save(fig, output_dir, "G3_Fig4_weight_method_comparison")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("figures") / "v2_1" / "g3")
    args = parser.parse_args()
    configure_style()
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    pareto_main(args.run_dir, summary, args.output_dir)
    constraint_sweep(args.run_dir, summary, args.output_dir)
    weight_topology(summary, args.output_dir)
    method_comparison(summary, args.output_dir)
    print(json.dumps({"figures": sorted(str(path) for path in args.output_dir.glob("G3_Fig*"))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
