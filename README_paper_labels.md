# Paper-facing terminology and reproducibility mapping

The manuscript uses descriptive labels so that each comparison is readable
without knowledge of the internal experiment numbering. Existing raw files,
configuration keys, and run-directory names retain their legacy identifiers so
that all reported values remain traceable to the archived runs.

## Communication conditions

| Paper label | Legacy key | Background sources | Rate per source |
|---|---|---:|---:|
| C1 | `light` | 0 | 0 Hz |
| C2 | `moderate` | 16 | 20 Hz |
| C3 | `heavy` | 48 | 50 Hz |

C1-C3 are controlled offered-load configurations, not standardized V2X
severity classes.

## Method labels

| Paper label | Legacy key | Meaning |
|---|---|---|
| B1 | `B1_random_connected` | connected random graph under the same edge budget |
| B2 | `B3_nominal_algebraic` | nominal-connectivity topology comparator |
| B3 | `B7_uniform` | uniform coordination weights |
| B4 | `B8_convergence_priority` | convergence-priority weights without a delay constraint |
| B5 | `B9` | sequential local predecessor contraction |
| B6 | `B10` | infrastructure-relayed implementation of the S3 constructor |
| B7 | `B11` | complete assignment executed at an edge controller |
| S1/S2/S3 | unchanged | proposed topology, weighting, and recovery stages |

## Metric labels

The raw field `position_iae_m_s` is reported in the paper as integrated RMS
position error, `J_p`. The raw field `control_energy_m2ps3` is the quadratic
control-effort integral, `J_u`; it is not fuel, battery, or physical energy.
Settling time uses a 0.05-m RMS threshold held continuously for 1 s.

## Figure policy

`generate_all_paper_figures.py` exports only the figures cited by the revised
manuscript. The simultaneous-dropout run and its legacy E6 plot are retained
as an auxiliary reproducibility check, but the paper relies on Corollary 1 for
the general batched-dropout statement and does not include a separate figure.
Manuscript Fig. 8 reads the paired Stage 3 results from
`runs/v2_4/g8/V24_G8_certified_stage12_inputs`; every fleet size in that run
uses the corresponding Stage 1 topology and Stage 2 weights reported by the
offline scale study underlying Table IV.
