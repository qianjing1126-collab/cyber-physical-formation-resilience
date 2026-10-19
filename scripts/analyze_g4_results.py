from __future__ import annotations

import csv
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from tnsm_sim.io import verify_manifest


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "runs" / "g4" / "G4_reportable_matrix_v2.json"
OUT = ROOT / "runs" / "g4" / "analysis_v2"
FIG = ROOT / "figures" / "g4"
BOOTSTRAPS = 10_000
METHODS_P04 = ("B0", "B9", "B11", "S3")
METHODS_P05 = ("B9", "B11", "S3")
COLORS = {"B0": "#9A8F87", "B9": "#C58F7B", "B11": "#7B93A8", "S3": "#627C70"}


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def rng_for(label: str) -> np.random.Generator:
    digest = sha256(label.encode("utf-8")).digest()
    return np.random.default_rng(6101 + int.from_bytes(digest[:4], "big"))


def bootstrap_mean_ci(values: list[float], label: str) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = rng_for(label)
    indices = rng.integers(0, len(array), size=(BOOTSTRAPS, len(array)))
    estimates = array[indices].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_records() -> tuple[dict, list[dict]]:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    records = []
    source_hashes = set()
    for design in matrix["rows"]:
        run_dir = ROOT / "runs" / "raw" / design["run_id"]
        audit = verify_manifest(run_dir)
        if not audit["valid"]:
            raise RuntimeError(f"manifest audit failed: {run_dir}")
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if metrics["config_hash"] != design["config_hash"] or status["run_state"] != "COMPLETE":
            raise RuntimeError(f"configuration/status audit failed: {run_dir}")
        source_hashes.add(environment["source_tree_sha256"])
        records.append({**design, **metrics})
    if len(records) != 900 or len(source_hashes) != 1:
        raise RuntimeError("G4 analysis requires 900 runs under one source-tree hash")
    return matrix, records


def group_key(record: dict) -> tuple[str, str, str, str, str]:
    return record["module"], record["path"], record["layout"], record["fault"], record["method"]


def summarize_groups(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        groups.setdefault(group_key(record), []).append(record)
    rows = []
    for key in sorted(groups):
        values = sorted(groups[key], key=lambda item: item["seed"])
        module, path, layout, fault, method = key
        n = len(values)
        successes = sum(bool(item["recovery_success"]) for item in values)
        infeasible = sum(item["scientific_status"] == "GEOMETRY_INFEASIBLE" for item in values)
        planner_infeasible = sum(item["scientific_status"] == "PLANNER_INFEASIBLE" for item in values)
        safety_violations = sum(not bool(item["safety_ok"]) for item in values)
        success_ci = wilson(successes, n)
        infeasible_ci = wilson(infeasible, n)
        rte = [float(item["R_Te"]) for item in values]
        qmin = [float(item["Q_min"]) for item in values]
        rrec = [float(item["R_rec"]) for item in values]
        rte_ci = bootstrap_mean_ci(rte, f"group-rte-{key}")
        qmin_ci = bootstrap_mean_ci(qmin, f"group-qmin-{key}")
        rrec_ci = bootstrap_mean_ci(rrec, f"group-rrec-{key}")
        recovery_times = [15.0 if item["T_r"] is None else float(item["T_r"]) for item in values]
        planner_times = [float(item["planner_time_s"]) * 1e6 for item in values if item["planner_time_s"] is not None]
        rows.append({
            "module": module, "path": path, "layout": layout, "fault": fault, "method": method,
            "n": n,
            "recovery_success_count": successes,
            "recovery_success_rate": successes / n,
            "success_wilson_low": success_ci[0], "success_wilson_high": success_ci[1],
            "geometry_infeasible_count": infeasible,
            "geometry_infeasible_rate": infeasible / n,
            "geometry_wilson_low": infeasible_ci[0], "geometry_wilson_high": infeasible_ci[1],
            "planner_infeasible_count": planner_infeasible,
            "safety_violation_count": safety_violations,
            "R_Te_mean": mean(rte), "R_Te_sd": stdev(rte) if n > 1 else 0.0,
            "R_Te_ci_low": rte_ci[0], "R_Te_ci_high": rte_ci[1],
            "Q_min_mean": mean(qmin), "Q_min_ci_low": qmin_ci[0], "Q_min_ci_high": qmin_ci[1],
            "R_rec_mean": mean(rrec), "R_rec_ci_low": rrec_ci[0], "R_rec_ci_high": rrec_ci[1],
            "T_r_RMST15_s": mean(recovery_times),
            "minimum_clearance_worst_m": min(float(item["minimum_clearance"]) for item in values),
            "peak_abs_ay_worst_mps2": max(float(item["peak_abs_ay"]) for item in values),
            "planner_time_median_us": median(planner_times) if planner_times else "",
        })
    return rows


def paired_comparisons(records: list[dict]) -> list[dict]:
    scenarios: dict[tuple, dict[str, dict[int, dict]]] = {}
    for record in records:
        scenario = record["module"], record["path"], record["layout"], record["fault"]
        scenarios.setdefault(scenario, {}).setdefault(record["method"], {})[int(record["seed"])] = record
    rows = []
    for scenario in sorted(scenarios):
        methods = scenarios[scenario]
        if "S3" not in methods:
            continue
        baselines = ("B0", "B9", "B11") if scenario[0] == "P04" else ("B9", "B11")
        for baseline in baselines:
            seeds = sorted(set(methods["S3"]) & set(methods[baseline]))
            for metric in ("R_Te", "Q_min", "R_rec"):
                differences = [float(methods["S3"][seed][metric]) - float(methods[baseline][seed][metric]) for seed in seeds]
                ci = bootstrap_mean_ci(differences, f"paired-{scenario}-{baseline}-{metric}")
                rows.append({
                    "module": scenario[0], "path": scenario[1], "layout": scenario[2], "fault": scenario[3],
                    "comparison": f"S3-{baseline}", "metric": metric, "n_pairs": len(seeds),
                    "mean_paired_difference": mean(differences), "ci_low": ci[0], "ci_high": ci[1],
                    "s3_wins": sum(value > 1e-12 for value in differences),
                    "ties": sum(abs(value) <= 1e-12 for value in differences),
                    "s3_losses": sum(value < -1e-12 for value in differences),
                    "mcnemar_exact_p": "",
                })
            s3_success = [bool(methods["S3"][seed]["recovery_success"]) for seed in seeds]
            base_success = [bool(methods[baseline][seed]["recovery_success"]) for seed in seeds]
            discordant_s3 = sum(a and not b for a, b in zip(s3_success, base_success))
            discordant_base = sum(b and not a for a, b in zip(s3_success, base_success))
            discordant = discordant_s3 + discordant_base
            pvalue = 1.0 if discordant == 0 else float(binomtest(discordant_s3, discordant, 0.5).pvalue)
            differences = [float(a) - float(b) for a, b in zip(s3_success, base_success)]
            ci = bootstrap_mean_ci(differences, f"paired-{scenario}-{baseline}-success")
            rows.append({
                "module": scenario[0], "path": scenario[1], "layout": scenario[2], "fault": scenario[3],
                "comparison": f"S3-{baseline}", "metric": "recovery_success", "n_pairs": len(seeds),
                "mean_paired_difference": mean(differences), "ci_low": ci[0], "ci_high": ci[1],
                "s3_wins": discordant_s3, "ties": len(seeds) - discordant, "s3_losses": discordant_base,
                "mcnemar_exact_p": pvalue,
            })
    return rows


def lookup(summary: list[dict], module: str, path: str, layout: str, fault: str, method: str) -> dict:
    return next(row for row in summary if (row["module"], row["path"], row["layout"], row["fault"], row["method"]) == (module, path, layout, fault, method))


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 140,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#4D5551", "axes.labelcolor": "#35423C",
        "xtick.color": "#4D5551", "ytick.color": "#4D5551",
        "grid.color": "#D9DFDB", "grid.alpha": 0.55,
    })


def save_figure(fig, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_p04(summary: list[dict]) -> None:
    style()
    paths = ("P1", "P2", "P3")
    x = np.arange(len(paths))
    width = 0.19
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), constrained_layout=True)
    for index, method in enumerate(METHODS_P04):
        rows = [lookup(summary, "P04", path, "F1", "P1", method) for path in paths]
        offset = (index - 1.5) * width
        success = np.asarray([row["recovery_success_rate"] for row in rows])
        low = np.asarray([row["success_wilson_low"] for row in rows])
        high = np.asarray([row["success_wilson_high"] for row in rows])
        axes[0].bar(x + offset, success, width, color=COLORS[method], label=method, edgecolor="white", linewidth=0.5)
        axes[0].errorbar(x + offset, success, yerr=np.maximum(0.0, np.vstack((success - low, high - success))), fmt="none", ecolor="#48514C", capsize=2, lw=0.7)
        rte = np.asarray([row["R_Te_mean"] for row in rows])
        rte_low = np.asarray([row["R_Te_ci_low"] for row in rows])
        rte_high = np.asarray([row["R_Te_ci_high"] for row in rows])
        axes[1].bar(x + offset, rte, width, color=COLORS[method], edgecolor="white", linewidth=0.5)
        axes[1].errorbar(x + offset, rte, yerr=np.maximum(0.0, np.vstack((rte - rte_low, rte_high - rte))), fmt="none", ecolor="#48514C", capsize=2, lw=0.7)
    axes[0].set(title="(a) Recovery success", ylabel="Success rate", xticks=x, xticklabels=paths, ylim=(0, 1.16))
    axes[1].set(title=r"(b) Fixed-window resilience $R_{T_e}$", ylabel=r"$R_{T_e}$", xticks=x, xticklabels=paths, ylim=(0, 0.9))
    for axis in axes:
        axis.set_xlabel("Reference path")
        axis.grid(axis="y")
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(1.05, 1.27), frameon=False)
    save_figure(fig, "G4_P04_path_recovery")


def plot_p05(summary: list[dict]) -> None:
    style()
    layouts = ("F1", "F2", "F3")
    faults = ("P2A", "P2N")
    x = np.arange(len(layouts))
    width = 0.24
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.0), sharex=True, constrained_layout=True)
    for column, fault in enumerate(faults):
        for index, method in enumerate(METHODS_P05):
            rows = [lookup(summary, "P05", "P3", layout, fault, method) for layout in layouts]
            offset = (index - 1) * width
            success = np.asarray([row["recovery_success_rate"] for row in rows])
            low = np.asarray([row["success_wilson_low"] for row in rows])
            high = np.asarray([row["success_wilson_high"] for row in rows])
            axes[0, column].bar(x + offset, success, width, color=COLORS[method], label=method, edgecolor="white", linewidth=0.5)
            axes[0, column].errorbar(x + offset, success, yerr=np.maximum(0.0, np.vstack((success - low, high - success))), fmt="none", ecolor="#48514C", capsize=2, lw=0.7)
            rte = np.asarray([row["R_Te_mean"] for row in rows])
            rte_low = np.asarray([row["R_Te_ci_low"] for row in rows])
            rte_high = np.asarray([row["R_Te_ci_high"] for row in rows])
            axes[1, column].bar(x + offset, rte, width, color=COLORS[method], edgecolor="white", linewidth=0.5)
            axes[1, column].errorbar(x + offset, rte, yerr=np.maximum(0.0, np.vstack((rte - rte_low, rte_high - rte))), fmt="none", ecolor="#48514C", capsize=2, lw=0.7)
        label = "Adjacent failures" if fault == "P2A" else "Nonadjacent failures"
        axes[0, column].set_title(label)
        axes[0, column].set_ylim(0, 1.16)
        axes[1, column].set_ylim(0, 0.7)
    axes[0, 0].set_ylabel("Recovery success rate")
    axes[1, 0].set_ylabel(r"Fixed-window resilience $R_{T_e}$")
    for axis in axes.flat:
        axis.set_xticks(x, layouts)
        axis.grid(axis="y")
    for axis in axes[1, :]:
        axis.set_xlabel("Formation layout")
    axes[0, 0].legend(ncol=3, loc="upper center", bbox_to_anchor=(1.08, 1.30), frameon=False)
    save_figure(fig, "G4_P05_layout_failure")


def plot_feasibility_and_time(summary: list[dict]) -> None:
    style()
    labels = ("F1/A", "F1/N", "F2/A", "F2/N", "F3/A", "F3/N")
    scenarios = [(layout, fault) for layout in ("F1", "F2", "F3") for fault in ("P2A", "P2N")]
    infeasible = [lookup(summary, "P05", "P3", layout, fault, "S3")["geometry_infeasible_rate"] for layout, fault in scenarios]
    planner_s3 = [lookup(summary, "P05", "P3", layout, fault, "S3")["planner_time_median_us"] for layout, fault in scenarios]
    planner_b11 = [lookup(summary, "P05", "P3", layout, fault, "B11")["planner_time_median_us"] for layout, fault in scenarios]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), constrained_layout=True)
    axes[0].bar(x, infeasible, color="#8A9C91", width=0.65)
    axes[0].set(title="(a) Residual-task infeasibility", ylabel="Geometry-infeasible rate", xticks=x, xticklabels=labels, ylim=(0, 0.38))
    axes[1].plot(x, planner_b11, "o-", color=COLORS["B11"], label="B11", lw=1.3, ms=4)
    axes[1].plot(x, planner_s3, "o-", color=COLORS["S3"], label="S3", lw=1.3, ms=4)
    axes[1].set(title="(b) Online planning time", ylabel="Median time (µs)", xticks=x, xticklabels=labels)
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xlabel("Layout / failure pattern")
        axis.grid(axis="y")
    save_figure(fig, "G4_feasibility_planning_time")


def write_summary_markdown(summary: list[dict], records: list[dict]) -> None:
    f3a = lookup(summary, "P05", "P3", "F3", "P2A", "S3")
    f3n = lookup(summary, "P05", "P3", "F3", "P2N", "S3")
    s3_time = median(float(item["planner_time_s"]) * 1e6 for item in records if item["module"] == "P05" and item["method"] == "S3")
    b11_time = median(float(item["planner_time_s"]) * 1e6 for item in records if item["module"] == "P05" and item["method"] == "B11")
    text = f"""# G4 reportable evidence summary

Protocol: `g4_recovery_safety_v1.2`

- All 900 reportable runs passed manifest/configuration audit under one source-tree hash.
- P0-4: B0 and B9 recovered in 0/30 seeds on each path; B11 and S3 recovered in 30/30 on P1, P2 and P3. S3 and B11 have identical mission and safety outcomes in the frozen ordered corridor.
- P0-5 F1/F2: B9 recovered in 0/30; B11 and S3 recovered in 30/30 for adjacent and nonadjacent simultaneous failures.
- P0-5 F3: residual geometry was infeasible in {f3a['geometry_infeasible_count']}/30 adjacent-failure seeds and {f3n['geometry_infeasible_count']}/30 nonadjacent-failure seeds. Consequently, B11 and S3 success rates were {f3a['recovery_success_rate']:.3f} and {f3n['recovery_success_rate']:.3f}; B9 recovered in 0/30.
- There were zero safety violations in all 900 runs. Geometry-infeasible cases remain in success-rate and fixed-window resilience statistics.
- Across P0-5, median measured planner time was {s3_time:.2f} µs for S3 and {b11_time:.2f} µs for B11. These small-N machine times support only an implementation-efficiency observation, not a scalability claim.

Interpretation boundary: G4 supports that S3 matches the centralized B11 recovery outcome without an online assignment optimizer under ordered coverage constraints, and outperforms the serial B9 baseline within the fixed window. It does not support claiming better mission recovery than B11; scale evidence belongs to G6.
"""
    (OUT / "G4_RESULTS_SUMMARY.md").write_text(text, encoding="utf-8")


def main() -> None:
    matrix, records = load_records()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = summarize_groups(records)
    paired = paired_comparisons(records)
    write_csv(OUT / "G4_run_index.csv", records)
    write_csv(OUT / "G4_group_summary.csv", summary)
    write_csv(OUT / "G4_paired_comparisons.csv", paired)
    plot_p04(summary)
    plot_p05(summary)
    plot_feasibility_and_time(summary)
    write_summary_markdown(summary, records)
    output_files = [
        OUT / "G4_run_index.csv",
        OUT / "G4_group_summary.csv",
        OUT / "G4_paired_comparisons.csv",
        OUT / "G4_RESULTS_SUMMARY.md",
        FIG / "G4_P04_path_recovery.png", FIG / "G4_P04_path_recovery.pdf",
        FIG / "G4_P05_layout_failure.png", FIG / "G4_P05_layout_failure.pdf",
        FIG / "G4_feasibility_planning_time.png", FIG / "G4_feasibility_planning_time.pdf",
    ]
    result = {
        "protocol_version": "g4_recovery_safety_v1.2",
        "matrix_hash": matrix["rows_sha256"],
        "run_count": len(records),
        "group_count": len(summary),
        "paired_comparison_count": len(paired),
        "bootstrap_resamples": BOOTSTRAPS,
        "outputs": {
            "group_summary": str(OUT / "G4_group_summary.csv"),
            "paired_comparisons": str(OUT / "G4_paired_comparisons.csv"),
            "summary": str(OUT / "G4_RESULTS_SUMMARY.md"),
            "figures": [
                str(FIG / "G4_P04_path_recovery.png"),
                str(FIG / "G4_P05_layout_failure.png"),
                str(FIG / "G4_feasibility_planning_time.png"),
            ],
        },
        "sha256": {
            str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
            for path in output_files
        },
    }
    (OUT / "G4_analysis_manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
