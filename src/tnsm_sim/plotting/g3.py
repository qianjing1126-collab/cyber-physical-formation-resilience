from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_g3_solver_gate(run_dir: Path) -> tuple[Path, Path]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    stage1, stage2 = summary["stage1"], summary["stage2"]
    colors = ["#B69A6A", "#647C73", "#A9B9AE"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
    labels = ["SDP relaxation\nupper bound", "Exact\nenumeration", "Rounded +\nrecertified"]
    values = [stage1["relaxation_upper_bound"], stage1["exact_optimum"], stage1["certified_incumbent"]]
    axes[0].bar(np.arange(3), values, color=colors)
    axes[0].set_xticks(np.arange(3), labels)
    axes[0].set_ylabel("Worst residual $\\lambda_2$")
    axes[0].set_title("Stage 1: bound and exact cross-check")
    for x, value in enumerate(values):
        axes[0].text(x, value + 0.04, f"{value:.4f}", ha="center", fontsize=9)

    solver_labels = [stage2["primary_solver"], stage2["independent_solver"]]
    gamma = [stage2["primary_gamma"], stage2["independent_gamma"]]
    recertified = [
        stage2["primary_recertification"]["worst_lambda2"],
        stage2["independent_recertification"]["worst_lambda2"],
    ]
    x = np.arange(2)
    width = 0.34
    axes[1].bar(x - width / 2, gamma, width, color="#71869A", label="Solver $\\gamma$")
    axes[1].bar(x + width / 2, recertified, width, color="#B78270", label="Recomputed worst $\\lambda_2$")
    axes[1].set_xticks(x, solver_labels)
    axes[1].set_ylabel("Stage 2 objective")
    axes[1].set_title("Stage 2: independent-solver agreement")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("G3-A formal optimization and numerical certification")
    png, pdf = run_dir / "g3a_solver_gate.png", run_dir / "g3a_solver_gate.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf
