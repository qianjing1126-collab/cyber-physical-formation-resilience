from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "paper_v21"
GREEN = "#607B70"
GREEN_LIGHT = "#AAB9B0"
BLUE = "#8299AC"
ORANGE = "#C58F79"
GOLD = "#B6A477"
GRAY = "#9D9992"
DARK = "#3F4D47"
RED = "#B65F59"


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.4,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.7,
        "axes.edgecolor": DARK,
        "axes.linewidth": 0.75,
        "grid.color": "#D9DEDA",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.70,
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


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png")
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)


def plot_e1() -> None:
    run = ROOT / "runs" / "v2_1" / "g1" / "V21_G1_stage1_structural_N8_f2_retry01"
    rows = read_csv(run / "topology_records.csv")
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    random_by_edge: dict[int, list[dict[str, str]]] = defaultdict(list)
    s1_by_edge: dict[int, dict[str, str]] = {}
    for row in rows:
        if row["scope"] != "anchor":
            continue
        edge_count = int(row["edge_count"])
        if row["method"] == "B1_random_connected":
            random_by_edge[edge_count].append(row)
        elif row["method"] == "S1_worst_residual_greedy":
            s1_by_edge[edge_count] = row
    edges = sorted(random_by_edge)
    fig = plt.figure(figsize=(3.48, 3.82), constrained_layout=True)
    grid = fig.add_gridspec(4, 1, height_ratios=[1.35, 0.13, 1.0, 0.13], hspace=0.02)
    ax_top = fig.add_subplot(grid[0, 0])
    caption_top = fig.add_subplot(grid[1, 0])
    ax_bottom = fig.add_subplot(grid[2, 0], sharex=ax_top)
    caption_bottom = fig.add_subplot(grid[3, 0])
    caption_top.axis("off")
    caption_bottom.axis("off")
    s1 = [float(s1_by_edge[e]["worst_lambda2"]) for e in edges]
    ax_top.plot(edges, s1, color=ORANGE, marker="D", markersize=3.7, linewidth=1.35,
                label=r"S1 worst-residual $\lambda_2$")
    exact = float(summary["exact_enumeration"]["optimal_worst_lambda2"])
    ax_top.scatter([12], [exact], marker="*", s=64, color=RED, edgecolor="white", linewidth=0.5,
                   zorder=5, label="Best set")
    ax_top.annotate(f"{exact:.3f}", xy=(12, exact), xytext=(7, 7), textcoords="offset points",
                    color=RED, fontsize=6.8)
    ax_top.axvline(12, color=RED, linestyle=":", linewidth=0.8, alpha=0.75)
    ax_top.set_ylabel(r"Worst residual $\lambda_2$")
    ax_top.set_ylim(-0.005, max(s1) * 1.17)
    ax_top.grid(axis="y")
    ax_top.legend(loc="upper left", handlelength=2.0)
    caption_top.text(0.5, 0.5, "(a) Worst-residual connectivity of the S1 topology",
                     ha="center", va="center", color=DARK, fontsize=6.8)

    survival = np.asarray([
        100.0 * float(summary["anchor_random"][str(e)]["all_scenario_connected_rate"])
        for e in edges
    ])
    ax_bottom.bar(edges, survival, width=0.58, color=GREEN_LIGHT,
                  edgecolor=GREEN, linewidth=0.75)
    for edge, rate in zip(edges, survival, strict=True):
        ax_bottom.text(edge, rate + 1.1, f"{rate:.0f}%", ha="center", va="bottom",
                       fontsize=6.2, color=DARK)
    ax_bottom.axvline(12, color=RED, linestyle=":", linewidth=0.8, alpha=0.75)
    ax_bottom.set_ylabel("Pass rate (%)")
    ax_bottom.set_xlabel("Number of communication links $|E|$")
    ax_bottom.set_xticks(edges)
    ax_bottom.set_ylim(0, max(42.0, float(survival.max()) * 1.20))
    ax_bottom.grid(axis="y")
    caption_bottom.text(0.5, 0.5, "(b) Pass rate of 1,000 random graphs per edge budget",
                        ha="center", va="center", color=DARK, fontsize=6.8)
    save(fig, "E1_edge_budget_residual_connectivity")


def plot_e2() -> None:
    run = ROOT / "runs" / "v2_1" / "g2" / "V21_G2_analysis_v5"
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    metrics = [
        ("End-to-end reachability", "e2e_reachability"),
        ("Effective end-to-end PDR", "effective_e2e_PDR"),
        ("Recovery-plan issuance", "coordination_plan_issued"),
        (r"Fixed-window resilience $R_{T_e}$", "R_Te"),
    ]
    comparisons = [("S1 $-$ B1", "S1_minus_B1", BLUE, "o"),
                   ("S1 $-$ B2", "S1_minus_B3", ORANGE, "D")]
    y = np.arange(len(metrics))[::-1]
    offsets = {"S1_minus_B1": 0.10, "S1_minus_B3": -0.10}
    fig, ax = plt.subplots(figsize=(3.48, 2.55), constrained_layout=True)
    for label, key, color, marker in comparisons:
        effects = summary["paired_effects"][key]
        means = np.asarray([100 * float(effects[m]["mean_delta"]) for _, m in metrics])
        low = np.asarray([100 * float(effects[m]["ci95_low"]) for _, m in metrics])
        high = np.asarray([100 * float(effects[m]["ci95_high"]) for _, m in metrics])
        yy = y + offsets[key]
        ax.errorbar(means, yy, xerr=np.vstack((means - low, high - means)), fmt=marker,
                    color=color, markeredgecolor="white", markeredgewidth=0.45,
                    markersize=4.5, linewidth=1.1, capsize=2.0, label=label)
    ax.axvline(0, color=DARK, linestyle="--", linewidth=0.8)
    ax.set_yticks(y, [label for label, _ in metrics])
    ax.set_xlabel("Mean paired improvement (percentage points)")
    ax.grid(axis="x")
    ax.legend(loc="lower right")
    save(fig, "E2_paired_cross_layer_effects")


def plot_e3() -> None:
    run = ROOT / "runs" / "v2_1" / "g3" / "V21_G3_stage2_weights_N8_f2_retry03"
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    edges = [tuple(edge) for edge in summary["topology"]["edges"]]
    selected = summary["selection"]["primary"]
    weights = {key: float(value) for key, value in selected["weights"].items()}
    circular_order = (2, 1, 8, 7, 6, 5, 4, 3)
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(circular_order), endpoint=False)
    pos = {node: np.asarray([np.cos(angle), np.sin(angle)])
           for node, angle in zip(circular_order, angles, strict=True)}
    values = np.asarray(list(weights.values()))
    low, high = float(values.min()), float(values.max())
    light = np.asarray([0xAF, 0xBD, 0xB6]) / 255.0
    dark = np.asarray([0x2F, 0x79, 0x6B]) / 255.0
    fig, ax = plt.subplots(figsize=(3.48, 3.12), constrained_layout=True)
    for index, (first, second) in enumerate(edges):
        weight = weights[f"{first}-{second}"]
        fraction = (weight - low) / max(high - low, 1e-12)
        color = light * (1 - fraction) + dark * fraction
        width = 1.1 + 2.5 * fraction
        start, end = pos[first], pos[second]
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=width,
                solid_capstyle="round", zorder=1)
        midpoint = 0.5 * (start + end)
        direction = end - start
        normal = np.asarray([-direction[1], direction[0]])
        normal /= max(np.linalg.norm(normal), 1e-12)
        point = midpoint + normal * (0.045 if index % 2 == 0 else -0.045)
        ax.text(point[0], point[1], f"{weight:.2f}", ha="center", va="center", fontsize=5.7,
                color=DARK, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.25})
    for node, point in pos.items():
        ax.scatter(point[0], point[1], s=300, color="#3E8E7E", edgecolor="white", linewidth=1.0, zorder=3)
        ax.text(point[0], point[1], f"{node}", ha="center", va="center", color="white",
                fontsize=8.4, fontweight="bold", zorder=4)
    ax.text(-1.27, 1.24, rf"$\lambda_2^{{\rm worst}}={selected['worst_lambda2']:.3f}$",
            fontsize=6.8, color=DARK)
    ax.text(-1.27, 1.13, rf"$\lambda_{{\max}}^{{\rm worst}}={selected['worst_lambda_max']:.3f}$",
            fontsize=6.8, color=DARK)
    ax.text(-1.27, 1.02, rf"$\tau_{{\rm m}}={1000*selected['delay_margin_s']:.1f}$ ms",
            fontsize=6.8, color=DARK)
    ax.plot([], [], color=light, linewidth=1.2, label=f"Low weight ({low:.2f})")
    ax.plot([], [], color=dark, linewidth=3.5, label=f"High weight ({high:.2f})")
    ax.legend(loc="lower center", ncols=2, bbox_to_anchor=(0.5, -0.02), handlelength=2.1)
    ax.set_aspect("equal")
    ax.set_xlim(-1.30, 1.30)
    ax.set_ylim(-1.20, 1.30)
    ax.axis("off")
    save(fig, "E3_selected_coordination_weights")


def plot_e6() -> None:
    run = ROOT / "runs" / "v2_2_clearance" / "g7" / "V21_G7_formal_simultaneous_N8"
    rows = read_csv(run / "group_summary.csv")
    layouts = [("F1_parallel", "Formation 1"),
               ("F2_oblique", "Formation 2"),
               ("FV_vshape", "Formation 3")]
    patterns = [("A_adjacent", "Adj."), ("N_nonadjacent", "Nonadj.")]
    methods = [("B9", ORANGE, "s", "B5"),
               ("B11", BLUE, "D", "B7"),
               ("S3", GREEN, "o", "S3")]
    categories = [(layout, pattern, f"{short}/{p_short}")
                  for layout, short in layouts for pattern, p_short in patterns]
    y = np.arange(len(categories))[::-1]
    offsets = {"B9": -0.16, "B11": 0.0, "S3": 0.16}
    fig, ax = plt.subplots(figsize=(3.48, 3.02), constrained_layout=True)
    for method, color, marker, display_label in methods:
        means, lows, highs, success = [], [], [], []
        for layout, pattern, _ in categories:
            row = next(r for r in rows if r["layout"] == layout and r["fault_pattern"] == pattern and r["method"] == method)
            means.append(float(row["R_Te_mean"]))
            lows.append(float(row["R_Te_ci_low"]))
            highs.append(float(row["R_Te_ci_high"]))
            success.append(100 * float(row["recovery_success_rate"]))
        mean = np.asarray(means); low = np.asarray(lows); high = np.asarray(highs)
        yy = y + offsets[method]
        ax.errorbar(mean, yy, xerr=np.vstack((mean - low, high - mean)), fmt=marker,
                    color=color, markeredgecolor="white", markeredgewidth=0.45,
                    markersize=4.2, linewidth=1.0, capsize=1.8,
                    label=display_label)
        if method == "B9":
            for x_hi, yp, rate in zip(high, yy, success, strict=True):
                ax.text(x_hi + 0.006, yp, f"Success={rate:.0f}%", ha="left", va="center",
                        fontsize=5.2, color=color)
    ax.set_yticks(y, [label for _, _, label in categories])
    ax.set_xlabel(r"Mean fixed-window resilience $R_{T_e}$")
    ax.set_xlim(0.28, 0.76)
    ax.grid(axis="x")
    ax.legend(loc="upper left", ncols=3, columnspacing=0.8, handletextpad=0.35,
              bbox_to_anchor=(0.01, 0.995))
    save(fig, "E6_simultaneous_failure_resilience")


def plot_e7() -> None:
    run = ROOT / "runs" / "v2_4" / "g8" / "V24_G8_certified_stage12_inputs"
    rows = sorted(read_csv(run / "resilience_summary.csv"), key=lambda row: int(row["n"]))
    n = np.asarray([int(row["n"]) for row in rows], dtype=float)
    mean = np.asarray([float(row["relative_gain_mean_percent"]) for row in rows])
    low = np.asarray([float(row["relative_gain_ci95_low_percent"]) for row in rows])
    high = np.asarray([float(row["relative_gain_ci95_high_percent"]) for row in rows])
    fig, ax = plt.subplots(figsize=(3.48, 2.35), constrained_layout=True)
    ax.errorbar(n, mean, yerr=np.vstack((mean - low, high - mean)), color=GREEN, marker="o",
                markersize=3.8, markerfacecolor="white", markeredgecolor=GREEN,
                markeredgewidth=0.9, linewidth=1.4, elinewidth=0.9,
                capsize=3.0, capthick=0.9, barsabove=True)
    ax.axhspan(18, 35, color=GREEN_LIGHT, alpha=0.20, zorder=0)
    ax.set_xlabel("Fleet size $N$")
    ax.set_ylabel(r"Relative $R_{T_e}$ gain of S3 over B5 (%)")
    ax.set_xticks(n)
    ax.set_ylim(17, 36)
    ax.grid(axis="y")
    save(fig, "E7_scale_resilience_gain")


def main() -> None:
    configure_style()
    plot_e1()
    plot_e2()
    plot_e3()
    plot_e6()
    plot_e7()
    print(json.dumps({"output_dir": str(OUT), "figures": sorted(path.name for path in OUT.glob("E[12367]*"))}, indent=2))


if __name__ == "__main__":
    main()
