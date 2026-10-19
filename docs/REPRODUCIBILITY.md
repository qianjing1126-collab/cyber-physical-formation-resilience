# Reproduction status and archived results

## What is supplied

The package includes simulation source, configurations, unit tests, raw CSV/JSON
experiment records, per-run checksums, plotting scripts, and ns-3 C++ calibration
source. The original `uv.lock` specifies the author's environment. This repository
additionally records the environment used for its packaging checks.

Restoring the archive and generating figures is distinct from independently
repeating the complete optimization, Monte Carlo, and network-calibration studies.
The original multi-gigabyte ns-3 packet traces are not part of the supplied ZIP.
The existing PowerShell helpers retain their documented Windows/WSL assumptions;
Linux/macOS users must adapt those installation paths before regenerating traces.

## Archived recovery gains and manuscript version

For the 90 paired runs used by the final clearance-checked recovery plots,
compute each run's relative improvement as
`100 * R_mission_gain / R_mission_no_recovery`, then average within each layout.

| Layout | Archived paired runs | Mean relative mission-resilience gain |
|---|---:|---:|
| Formation 1, `F1_parallel` | 30 | 17.5813950743% |
| Formation 2, `F2_oblique` | 30 | 17.4243411826% |
| Formation 3, `FV_vshape` | 30 | 17.5812549759% |
| Equal-weight mean across layouts | 90 total | 17.5289970776% |

These round to **17.58%, 17.42%, 17.58%**, with an overall average of **17.53%**.
A manuscript draft associated with these materials states 17.44% for Formation 2.
That value is not the rounding of these archived paired results. For reproduction
of this archive, the corresponding value is 17.42%. The intended experiment version
must be checked when aligning a manuscript with this repository. No simulation
records were changed to match the manuscript draft.

The recovery plot's former fixed percentage annotations now use the archived
paired CSV records directly. This changes no simulation algorithm or data.
Run `python tools/summarize_recovery_gains.py` to inspect the calculation.

## Archive provenance

The original ZIP was retained separately. All 16 original per-run SHA-256
manifests passed before preparation. Three original Windows workstation paths
were replaced with repository-relative provenance paths; their corresponding
manifest entries were refreshed. Original and public file digests are listed in
`docs/ARCHIVE_REDACTIONS.json`. The CSV numerical records and experiment
configuration values were not modified.

`data/ARCHIVE_MANIFEST.json` checks the compressed archive and every restored
file. The archive is distributed as small binary parts indexed in
`data/ARCHIVE_PARTS.json`; the helper rejoins them and checks each part and the
original whole-archive SHA-256 before extraction. No experimental data are changed
by this packaging step. The restoration helper refuses to overwrite differing local files unless
explicitly invoked with `--force`.

## Repository version

The repository uses the MIT License. For reproducibility, identify the exact
commit used when citing these materials or comparing them with a manuscript.
No archived software DOI or accepted-paper DOI is asserted by this repository.

## Packaging checks performed on 2026-10-19

- 38 selected existing unit tests passed, covering metrics, information age,
  network calibration helpers, geometry, fault sets, delay behavior, and a small
  offline-design pipeline. This is not a claim that the entire supplied test
  suite or every full experiment was rerun.
- The manuscript figure entry point generated all 11 expected experimental PDF
  panels successfully. The Formation 2 figure annotation was checked against
  the archived 17.42% result.
- All 16 original per-run manifests passed; the restored public archive and all
  its 139 files were verified after the documented metadata-only redactions.
- Editable installation and the CLI help/configuration-validation entry points
  were checked. Detailed environment versions are in `VERIFICATION.json` and
  `requirements-verified.txt`.
