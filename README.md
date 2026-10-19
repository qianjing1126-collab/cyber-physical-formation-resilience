# Cyber-Physical Resilience in Networked Cooperative Formation Systems

Research code accompanying **Cyber-Physical Resilience in Networked Cooperative
Formation Systems: Topology Optimization and Formation Recovery**, by Ke Sun,
Zhonghao Qian, and Zhiwei Xing.

This repository contains the simulation engine, configurations, tests, archived
experiment records, and generators for the manuscript's experimental figures.
The airport snow-removal example validates an ordered cooperative formation task.
It does not provide a general-purpose autonomous-vehicle implementation.

The code is available under the MIT License. Installation, figure reproduction,
verification status, and known manuscript/archive differences are documented below
and in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Quick start

Use Python 3.11 or newer. In a terminal at the repository root:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
python tools/restore_archived_runs.py
python generate_all_paper_figures.py
```

The 11 experimental PDF panels and their PNG previews are written to
`figures/paper_final/`. The generators use archived records; the constant-delay
figure also evaluates its configured dynamics. This is figure reproduction,
not a fresh rerun of every Monte Carlo study or ns-3 calibration.

The experiment archive is distributed as small binary files in `data/archive_parts/`.
The restoration helper automatically rejoins them into `data/archived_runs.zip`,
verifies the original archive checksum, and recreates the original `runs/` layout.
The parts index is `data/ARCHIVE_PARTS.json`. Its complete experiment-file SHA-256
manifest is `data/ARCHIVE_MANIFEST.json`. Three workstation-path metadata fields
were made relative; numerical records were preserved. See
[docs/ARCHIVE_REDACTIONS.json](docs/ARCHIVE_REDACTIONS.json).

## Tests and data checks

```bash
python -m unittest discover -s tests -v
python tools/restore_archived_runs.py --verify-only
python tools/summarize_recovery_gains.py
```

## Repeating simulations

The CLI exposes validation, simulation, topology design, weight design, and
recovery experiments:

```bash
python -m tnsm_sim.cli --help
python -m tnsm_sim.cli validate configs/base.yaml
python -m tnsm_sim.cli v21-g9 configs/v2_1_g9_offline_scale.json --output-root runs/new_offline_scale
python -m tnsm_sim.cli v21-g8 configs/v2_1_g8.json --output-root runs/new_recovery_scale
```

Offline optimization and full Monte Carlo experiments can be substantially slower
than replaying archived results. Keep the supplied configs and random seeds to
preserve the experimental conditions. `uv.lock` retains the original locked
dependency versions; `requirements-verified.txt` records the separate packaging
check environment. Numerical results can depend on solver versions and tolerance.

The `ns3/` directory contains the ns-3.47 calibration source. Rebuilding the full
packet-trace calibration requires a separate ns-3 installation and its original
simulation protocol. Multi-gigabyte raw ns-3 traces are not included. ns-3 is not
required to regenerate the manuscript figures from the supplied archived data.

## Paper mapping and limitations

- [Figure/data mapping](docs/FIGURE_DATA_MAP.md)
- [Baseline and metric labels](README_paper_labels.md)
- [Reproduction status and known discrepancies](docs/REPRODUCIBILITY.md)
- [Citation metadata](CITATION.cff)

Legacy run IDs are retained to preserve traceability. `J_u` is quadratic control
effort, not measured electrical or fuel energy. The analytical delay margin
concerns fixed symmetric residual graphs with a common constant delay; sampled
clearance checks do not prove continuous-time collision avoidance.

The `reference_outputs/` directory contains author-supplied manuscript plot
snapshots. Rendering and fonts may differ from regenerated plots, and those
snapshots are not a substitute for validating the archived numeric records.

## License

MIT; see [LICENSE](LICENSE). Dependency software retains its respective license.
