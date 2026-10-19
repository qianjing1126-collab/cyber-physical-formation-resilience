# Manuscript figure and archived-data mapping

All paths below are relative to the repository root after restoring the archive.

| Manuscript output | Generator | Primary source |
|---|---|---|
| Fig. 3 | `scripts/plot_v21_paper_single_panel_figures.py`, `plot_e1` | `runs/v2_1/g1/V21_G1_stage1_structural_N8_f2_retry01/` |
| Fig. 4 | same script, `plot_e2` | `runs/v2_1/g2/V21_G2_analysis_v5/` |
| Fig. 5 | same script, `plot_e3` | `runs/v2_1/g3/V21_G3_stage2_weights_N8_f2_retry03/` |
| Fig. 6 | `scripts/plot_v21_paper_wide_figures.py`, `plot_e4` | `configs/v2_1_g4.json`, G3 selected weights, and configured constant-delay dynamics |
| Fig. 7(a,c,e) | `scripts/plot_v21_paper_recovery_panels_final.py` | `runs/v2_2_clearance/g6/V21_G6_sequential_recovery_N8/representative_states.csv` |
| Fig. 7(b,d,f) | same script | same run directory, `representative_metrics.csv`; gain annotations use `run_records.csv` |
| Fig. 8 | `scripts/plot_v21_paper_single_panel_figures.py`, `plot_e7` | `runs/v2_4/g8/V24_G8_certified_stage12_inputs/` |

The E6 simultaneous-failure plot is auxiliary. It is not an additional formal
manuscript figure. `fig8a_f1_service` and similarly named files are panels of
manuscript Fig. 7; the formal manuscript Fig. 8 is `fig8_scale_gain.pdf`.

The table of offline designs uses
`runs/v2_1/g9/V21_G9_offline_stage12_scale/offline_scale_summary.csv` and the
corresponding `offline_scale_details.json`. Packet/control results are archived
under `runs/v2_1/g5/V21_G5_packet_closed_loop_N8/`.

The package intentionally retains earlier run directories as historical evidence.
Use the paths above for the current manuscript figures rather than selecting
a directory solely by its version number.
