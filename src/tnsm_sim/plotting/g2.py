from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "green": "#647C73",
    "light_green": "#A9B9AE",
    "rose": "#B78270",
    "sand": "#B69A6A",
    "blue": "#71869A",
}


def plot_p01(run_dir: Path) -> tuple[Path, Path]:
    records = json.loads((run_dir / "method_records.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "canonical_config.json").read_text(encoding="utf-8"))
    labels = ["Random\nconnected", "Regular", "Nominal\n$\\lambda_2$", "Stage 1\nrobust"]
    method_order = [
        "B1_random_connected", "B2_circulant_regular",
        "B3_nominal_lambda2_greedy", "Stage1_robust_greedy_pilot",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    rng = np.random.default_rng(7)
    for x, method in enumerate(method_order):
        subset = [record for record in records if record["method"] == method]
        values = [record["worst_lambda2"] for record in subset]
        rates = [record["connectivity_rate"] for record in subset]
        jitter = rng.uniform(-0.10, 0.10, len(subset)) if len(subset) > 1 else np.zeros(1)
        axes[0].scatter(x + jitter, values, color=COLORS["green"], s=28, alpha=0.75)
        axes[1].scatter(x + jitter, rates, color=COLORS["rose"], s=28, alpha=0.75)
    axes[0].axhline(config["z_req"], color=COLORS["sand"], linestyle="--", label="$z_{req}$")
    axes[0].set_ylabel("Worst residual $\\lambda_2$")
    axes[0].legend(frameon=False)
    axes[1].axhline(1.0, color=COLORS["sand"], linestyle="--")
    axes[1].set_ylabel("All-scenario connectivity rate")
    axes[1].set_ylim(-0.03, 1.06)
    for axis in axes:
        axis.set_xticks(range(4), labels)
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("G2 P0-1 pilot: residual-connectivity recertification ($N=8$, $f=2$, 37 scenarios)")
    png, pdf = run_dir / "g2_p01_diagnostic.png", run_dir / "g2_p01_diagnostic.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_p02(run_dir: Path) -> tuple[Path, Path]:
    records = json.loads((run_dir / "delay_records.json").read_text(encoding="utf-8"))
    labels = ["Uniform", "Convergence\npriority", "Stage 2\ndelay-aware"]
    x = np.arange(len(records))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    width = 0.24
    axes[0].bar(x - width, [r["tau_theory_s"] for r in records], width, label="Theory", color=COLORS["green"])
    axes[0].bar(x, [r["tau_pade_s"] for r in records], width, label="Padé roots", color=COLORS["light_green"])
    axes[0].bar(x + width, [r["tau_empirical_s"] for r in records], width, label="Time domain", color=COLORS["rose"])
    axes[0].set_ylabel("Critical delay / s")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].bar(x - width / 2, [r["worst_lambda2"] for r in records], width, label="Worst $\\lambda_2$", color=COLORS["blue"])
    axes[1].bar(x + width / 2, [r["worst_lambda_max"] for r in records], width, label="Worst $\\lambda_{max}$", color=COLORS["sand"])
    axes[1].axhline(records[0]["lambda_cap"], linestyle="--", color=COLORS["rose"], label="Spectral cap")
    axes[1].set_ylabel("Residual spectral value")
    axes[1].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("G2 P0-2 pilot: independent verification of the Stage 2 delay margin")
    png, pdf = run_dir / "g2_p02_diagnostic.png", run_dir / "g2_p02_diagnostic.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf
