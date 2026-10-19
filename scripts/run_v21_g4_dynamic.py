from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tnsm_sim.weights import modal_delay_margin
from tnsm_sim.weights.root_verification import spectral_abscissa


COLORS = {
    "B7_uniform": "#9D9992",
    "B8_convergence_priority": "#C58F79",
    "S2_delay_constrained": "#5F786C",
}
LABELS = {
    "B7_uniform": "B3 均匀权重",
    "B8_convergence_priority": "B4 收敛优先",
    "S2_delay_constrained": "S2 时延约束",
}
VEHICLE_COLORS = ["#426B61", "#B47D68", "#71899D", "#A39B8B", "#7C6F84", "#A58B5F", "#5F7E82", "#8A6B61"]


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
        "savefig.dpi": 240,
        "savefig.bbox": "tight",
    })


def laplacian(n: int, edges: list[tuple[int, int]], weights: dict[str, float]) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=float)
    for first, second in edges:
        weight = float(weights[f"{first}-{second}"])
        i, j = first - 1, second - 1
        matrix[i, i] += weight
        matrix[j, j] += weight
        matrix[i, j] -= weight
        matrix[j, i] -= weight
    return matrix


def simulate_mode(delay: float, lam: float, q0: float, v0: float, force_scale: float,
                  config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = float(config["dt_s"])
    duration = float(config["duration_s"])
    steps = int(round(duration / dt)) + 1
    time = np.arange(steps, dtype=float) * dt
    position = np.empty(steps, dtype=float)
    velocity = np.empty(steps, dtype=float)
    position[0], velocity[0] = q0, v0
    k_p, k_v = float(config["k_p"]), float(config["k_v"])
    disturb_end = float(config["disturbance_duration_s"])
    disturb_amp = float(config["disturbance_amplitude_mps2"]) * force_scale

    def history(query: float, known_index: int) -> tuple[float, float]:
        if query <= 0:
            return q0, v0
        scaled = query / dt
        left = min(int(scaled), known_index - 1)
        fraction = scaled - left
        return (
            position[left] * (1.0 - fraction) + position[left + 1] * fraction,
            velocity[left] * (1.0 - fraction) + velocity[left + 1] * fraction,
        )

    def force(query_time: float) -> float:
        if 0.0 <= query_time <= disturb_end:
            return disturb_amp * np.sin(np.pi * query_time / disturb_end)
        return 0.0

    def acceleration(query_time: float, known_index: int) -> float:
        delayed_position, delayed_velocity = history(query_time - delay, known_index)
        return -lam * (k_v * delayed_velocity + k_p * delayed_position) + force(query_time)

    for index in range(steps - 1):
        now = time[index]
        k1_p = velocity[index]
        k1_v = acceleration(now, index)
        k2_p = velocity[index] + 0.5 * dt * k1_v
        k2_v = acceleration(now + 0.5 * dt, index)
        k3_p = velocity[index] + 0.5 * dt * k2_v
        k3_v = acceleration(now + 0.5 * dt, index)
        k4_p = velocity[index] + dt * k3_v
        k4_v = acceleration(now + dt, index)
        position[index + 1] = position[index] + dt * (k1_p + 2*k2_p + 2*k3_p + k4_p) / 6.0
        velocity[index + 1] = velocity[index] + dt * (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6.0
    return time, position, velocity


def simulate_network(delay: float, matrix: np.ndarray, config: dict) -> dict[str, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    initial_position = np.asarray([0.80, -0.65, 0.55, -0.45, 0.35, -0.25, 0.15, -0.50], dtype=float)
    initial_position -= initial_position.mean()
    initial_velocity = np.asarray([0.10, -0.08, 0.07, -0.06, 0.05, -0.04, 0.03, -0.07], dtype=float)
    initial_velocity -= initial_velocity.mean()
    disturbance_shape = np.asarray([1.0, -0.8, 0.6, -0.4, 0.3, -0.2, 0.5, -1.0], dtype=float)
    disturbance_shape -= disturbance_shape.mean()
    disturbance_shape /= np.max(np.abs(disturbance_shape))
    q0 = eigenvectors.T @ initial_position
    v0 = eigenvectors.T @ initial_velocity
    fmodal = eigenvectors.T @ disturbance_shape
    modal_position = []
    modal_velocity = []
    time = None
    for index in range(1, len(eigenvalues)):
        time, position, velocity = simulate_mode(
            delay, float(eigenvalues[index]), float(q0[index]), float(v0[index]),
            float(fmodal[index]), config,
        )
        modal_position.append(position)
        modal_velocity.append(velocity)
    position_error = eigenvectors[:, 1:] @ np.asarray(modal_position)
    velocity_error = eigenvectors[:, 1:] @ np.asarray(modal_velocity)
    assert time is not None
    return {
        "time": time,
        "position": position_error,
        "velocity": velocity_error,
        "position_rms": np.sqrt(np.mean(position_error**2, axis=0)),
        "velocity_rms": np.sqrt(np.mean(velocity_error**2, axis=0)),
        "eigenvalues": eigenvalues,
    }


def settling_time(time: np.ndarray, rms: np.ndarray, threshold: float, hold: float) -> float | None:
    samples = max(1, int(round(hold / (time[1] - time[0]))))
    below = rms <= threshold
    for index in range(len(time) - samples + 1):
        if below[index:index + samples].all() and below[index:].all():
            return float(time[index])
    return None


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.png")
    fig.savefig(output / f"{stem}.pdf")
    plt.close(fig)


def plot_stability(sweep: list[dict], margins: dict[str, float], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.4, 5.3), constrained_layout=True)
    for method in LABELS:
        rows = [row for row in sweep if row["method"] == method]
        axis.plot([1000*row["delay_s"] for row in rows], [row["spectral_abscissa"] for row in rows],
                  color=COLORS[method], linewidth=2.3, label=LABELS[method])
        axis.axvline(1000*margins[method], color=COLORS[method], linestyle="--", linewidth=1.0, alpha=0.85)
    axis.axhline(0.0, color="#3F4D47", linewidth=1.0)
    axis.axvspan(0, 21, color="#AAB9B0", alpha=0.22, label="ns-3标定范围（≤21 ms）")
    axis.set(xlabel="公共对齐时延 / ms", ylabel="最右特征根实部", title="V2.1-G4：动态稳定边界与理论时延裕度")
    axis.grid()
    axis.legend(ncols=2, fontsize=9)
    save(fig, output, "G4_Fig1_stability_boundary")


def plot_stress(traces: dict, delay: float, config: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.3), sharex=True, constrained_layout=True)
    methods = list(LABELS)
    end_time = float(config.get("stress_plot_end_s", 8.0))
    for column, method in enumerate(methods):
        trace = traces[(method, delay)]
        stable = trace["spectral_abscissa"] < 0
        shown = trace["time"] <= end_time + 1e-12
        for row, key in enumerate(("position", "velocity")):
            axis = axes[row, column]
            for vehicle in range(8):
                axis.plot(trace["time"][shown], trace[key][vehicle, shown],
                          color=VEHICLE_COLORS[vehicle], linewidth=1.05, alpha=0.86,
                          label=f"Agent {vehicle + 1}" if column == 0 and row == 0 else None)
            axis.axvspan(0, float(config["disturbance_duration_s"]), color="#D8C7A7", alpha=0.22)
            axis.axhline(0, color="#3F4D47", linewidth=0.8)
            peak = float(np.abs(trace[key][:, shown]).max())
            axis.set_ylim(-1.08 * peak, 1.08 * peak)
            axis.set_xlim(0, end_time)
            axis.grid()
            unit = "m" if key == "position" else "m/s"
            axis.text(0.97, 0.94, f"峰值 {peak:.2f} {unit}", transform=axis.transAxes,
                      ha="right", va="top", fontsize=8.6, color="#596660")
            if row == 0:
                axis.set_title(f"{LABELS[method]}\n{'稳定' if stable else '失稳'}")
            if column == 0:
                axis.set_ylabel("位置误差 / m" if row == 0 else r"速度误差 / (m·s$^{-1}$)", labelpad=5)
            if row == 1:
                axis.set_xlabel("时间 / s")
    axes[0, 0].legend(loc="lower right", ncol=2, fontsize=7.6, columnspacing=0.8, handlelength=1.8)
    fig.suptitle(f"公共对齐时延压力测试：$\\tau={1000*delay:.0f}$ ms（阴影为初始扰动；各子图采用独立线性纵轴）",
                 fontsize=14, color="#3F4D47")
    save(fig, output, "G4_Fig2_stress_tracking_errors")


def plot_realistic(traces: dict, delay: float, config: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    for method in LABELS:
        trace = traces[(method, delay)]
        axes[0].plot(trace["time"], trace["position_rms"], color=COLORS[method], linewidth=2.2, label=LABELS[method])
        axes[1].plot(trace["time"], trace["velocity_rms"], color=COLORS[method], linewidth=2.2, label=LABELS[method])
    for axis, ylabel in zip(axes, ("位置误差RMS / m", "速度误差RMS / (m/s)"), strict=True):
        axis.axvspan(0, float(config["disturbance_duration_s"]), color="#D8C7A7", alpha=0.22)
        axis.set(xlabel="时间 / s", ylabel=ylabel)
        axis.grid()
    axes[0].set_title(f"现实网络上界：$\\tau={1000*delay:.0f}$ ms")
    axes[1].set_title("三种权重均位于稳定域内")
    axes[1].legend(fontsize=9)
    save(fig, output, "G4_Fig3_realistic_delay_transient")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/v2_1_g4.json"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/v2_1/g4"))
    parser.add_argument("--figure-root", type=Path, default=Path("figures/v2_1/g4"))
    args = parser.parse_args()
    style()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    g3 = json.loads((Path(config["g3_source"]) / "summary.json").read_text(encoding="utf-8"))
    edges = [tuple(edge) for edge in g3["topology"]["edges"]]
    matrices = {item["method"]: laplacian(8, edges, item["weights"]) for item in g3["methods"]}
    margins = {item["method"]: float(item["delay_margin_s"]) for item in g3["methods"]}

    delays = np.linspace(float(config["stability_sweep_start_s"]), float(config["stability_sweep_end_s"]),
                         int(config["stability_sweep_points"]))
    sweep = []
    for method, matrix in matrices.items():
        eigenvalues = np.linalg.eigvalsh(matrix)[1:]
        for delay in delays:
            abscissa = max(spectral_abscissa(float(delay), float(lam), float(config["k_p"]),
                                             float(config["k_v"]), int(config["pade_order"])) for lam in eigenvalues)
            sweep.append({"method": method, "delay_s": float(delay), "spectral_abscissa": float(abscissa)})

    trace_delays = [float(value) for value in config["trace_delays_s"]]
    traces = {}
    trace_rows = []
    metrics = []
    for method, matrix in matrices.items():
        eigenvalues = np.linalg.eigvalsh(matrix)[1:]
        for delay in trace_delays:
            trace = simulate_network(delay, matrix, config)
            abscissa = max(spectral_abscissa(delay, float(lam), float(config["k_p"]),
                                             float(config["k_v"]), int(config["pade_order"])) for lam in eigenvalues)
            trace["spectral_abscissa"] = float(abscissa)
            traces[(method, delay)] = trace
            settle = settling_time(trace["time"], trace["position_rms"], float(config["settling_threshold_m"]),
                                   float(config["settling_hold_s"]))
            metrics.append({
                "method": method, "delay_s": delay, "stable_by_roots": bool(abscissa < 0),
                "spectral_abscissa": float(abscissa), "peak_position_rms_m": float(trace["position_rms"].max()),
                "final_position_rms_m": float(trace["position_rms"][-1]), "settling_time_s": settle,
            })
            stride = 10
            for index in range(0, len(trace["time"]), stride):
                trace_rows.append({
                    "method": method, "delay_s": delay, "time_s": float(trace["time"][index]),
                    "position_rms_m": float(trace["position_rms"][index]),
                    "velocity_rms_mps": float(trace["velocity_rms"][index]),
                })

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "stability_sweep.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep[0])); writer.writeheader(); writer.writerows(sweep)
    with (args.output_root / "rms_traces.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0])); writer.writeheader(); writer.writerows(trace_rows)
    result = {"gate": "V2.1-G4", "config": config, "topology_hash": g3["topology"]["graph_sha256"],
              "theoretical_margins_s": margins, "metrics": metrics,
              "stress_classification_pass": all(
                  next(row for row in metrics if row["method"] == method and row["delay_s"] == float(config["stress_delay_s"]))["stable_by_roots"] == (method == "S2_delay_constrained")
                  for method in LABELS),
              "claim_boundary": config["claim_boundary"]}
    (args.output_root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_stability(sweep, margins, args.figure_root)
    plot_stress(traces, float(config["stress_delay_s"]), config, args.figure_root)
    plot_realistic(traces, float(config["realistic_delay_s"]), config, args.figure_root)
    print(json.dumps({"summary": str(args.output_root / "summary.json"), "figures": str(args.figure_root),
                      "stress_classification_pass": result["stress_classification_pass"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
