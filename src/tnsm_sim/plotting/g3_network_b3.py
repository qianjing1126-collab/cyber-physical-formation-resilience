from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_g3b3_calibration(run_dir: Path) -> tuple[Path, Path]:
    with (run_dir / "holdout_comparison.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = [row["regime"].capitalize() for row in rows]
    x = np.arange(len(rows))
    width = 0.34
    panels = [
        ("pdr", "Packet delivery ratio", 1.0),
        ("delay_p95_s", "95th-percentile delay / ms", 1000.0),
        ("aoi_p95_s", "95th-percentile AoI / ms", 1000.0),
        ("contention_event_rate", "Contention-event rate", 1.0),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.8), constrained_layout=True)
    for axis, (metric, title, scale) in zip(axes, panels):
        reference = [float(row[f"reference_{metric}"]) * scale for row in rows]
        adapter = [float(row[f"adapter_{metric}"]) * scale for row in rows]
        axis.bar(x - width / 2, reference, width, color="#657C72", label="ns-3 holdout")
        axis.bar(x + width / 2, adapter, width, color="#B4977A", label="Two-state adapter")
        axis.set_xticks(x, labels)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("G3-B3 stationary-reference calibration — untouched seeds 36–45")
    png = run_dir / "g3b3_holdout_calibration.png"
    pdf = run_dir / "g3b3_holdout_calibration.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf
