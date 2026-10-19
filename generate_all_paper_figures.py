from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
INTERMEDIATE = ROOT / "figures" / "paper_v21"
FINAL = ROOT / "figures" / "paper_final"


def run(script: str) -> None:
    print(f"[RUN] {script}")
    subprocess.run([sys.executable, str(SCRIPTS / script)], cwd=ROOT, check=True)


def copy_pair(source_stem: str, target_stem: str) -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        source = INTERMEDIATE / f"{source_stem}{suffix}"
        target = FINAL / f"{target_stem}{suffix}"
        shutil.copy2(source, target)


def main() -> None:
    # Remove obsolete manuscript aliases from earlier revisions. The underlying
    # E6 simultaneous-dropout result remains in the intermediate directory.
    for stale_name in ("fig8.pdf", "fig8.png", "fig9.pdf", "fig9.png"):
        (FINAL / stale_name).unlink(missing_ok=True)

    run("plot_v21_paper_single_panel_figures.py")
    # The wide script also writes the old combined recovery figure. It is kept
    # for backward compatibility; the manuscript uses the split panels below.
    run("plot_v21_paper_wide_figures.py")
    run("plot_v21_paper_recovery_panels_final.py")

    mapping = {
        "E1_edge_budget_residual_connectivity": "fig3",
        "E2_paired_cross_layer_effects": "fig4",
        "E3_selected_coordination_weights": "fig5",
        "E4_delay_stress_tracking_errors": "fig6",
        "E7_scale_resilience_gain": "fig8_scale_gain",
    }
    for source_stem, target_stem in mapping.items():
        copy_pair(source_stem, target_stem)

    expected = (
        "fig3.pdf", "fig4.pdf", "fig5.pdf", "fig6.pdf",
        "fig7a_f1_trajectory.pdf", "fig7b_f2_trajectory.pdf",
        "fig7c_fv_trajectory.pdf", "fig8a_f1_service.pdf",
        "fig8b_f2_service.pdf", "fig8c_fv_service.pdf",
        "fig8_scale_gain.pdf",
    )
    missing = [name for name in expected if not (FINAL / name).exists()]
    if missing:
        raise RuntimeError(f"Missing expected figures: {missing}")
    print("\nAll manuscript experimental figures were generated successfully:")
    for name in expected:
        print(f"  {FINAL / name}")


if __name__ == "__main__":
    main()
