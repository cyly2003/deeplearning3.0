# v1.2.31 Molecular-Size Descriptor Sensitivity Summary

## Purpose

This sensitivity check tested whether pTox model performance depends materially
on molecular weight or strongly molecular-size-related descriptors. The concern
is that some `pTox` targets are converted from `mg/L` to `mol/L` using
molecular weight:

```text
pTox = -log10(mg/L) + 3 + log10(MW)
```

The new deep-model ablation `no_molecular_size_descriptors` masks `MolWt`,
`TPSA`, `HeavyAtomCount`, `NumHAcceptors`, `NumHDonors`, `RingCount`, and
`RotatableBonds`, while keeping `MolLogP`, Morgan fingerprints, species/context
features, source weighting, toxicity binning, and censored loss unchanged.

## Run Status

- Remote launcher: `scripts/run_v1_2_31_molecular_size_sensitivity_remote.sh priority`
- Started: 2026-07-07 01:35 (+08:00)
- Finished: 2026-07-07 03:15 (+08:00)
- Formal runs: 4/4 completed, all exit_code `0`
- Summary directory: `outputs/experiments/v1_2_31_molecular_size_sensitivity_remote_summary`
- Runtime table: `outputs/logs/run_v1_2_31_molecular_size_sensitivity_times.csv`

## Test Metrics

| Split policy | Ablation | n | R2 | RMSE | MAE | Huber loss |
|---|---:|---:|---:|---:|---:|---:|
| no-metal random 8:2 | full | 2608 | 0.7635 | 0.9397 | 0.6724 | 0.3446 |
| no-metal random 8:2 | no_molecular_size_descriptors | 2608 | 0.7658 | 0.9350 | 0.6656 | 0.3399 |
| scaffold/similarity-cluster holdout | full | 2493 | 0.1204 | 1.4574 | 1.1114 | 0.7177 |
| scaffold/similarity-cluster holdout | no_molecular_size_descriptors | 2493 | 0.0988 | 1.4752 | 1.1308 | 0.7301 |

## Delta vs Full

| Split policy | Delta R2 | Delta RMSE | Delta MAE | Delta Huber loss |
|---|---:|---:|---:|---:|
| no-metal random 8:2 | +0.0023 | -0.0046 | -0.0068 | -0.0047 |
| scaffold/similarity-cluster holdout | -0.0217 | +0.0178 | +0.0194 | +0.0125 |

## Interpretation

The priority result does not support the idea that current pTox performance is
mainly driven by direct molecular-weight/size descriptor coupling.

In the no-metal random 8:2 interpolation setting, removing molecular-size
descriptors slightly improved all primary metrics. This suggests these
descriptors are not required for the strong random-split performance and may
even add small redundant or noisy signal when fingerprints and context are
available.

In the scaffold/similarity-cluster holdout setting, removing molecular-size
descriptors caused a small performance loss: MAE increased by `0.0194` log
units and R2 decreased by `0.0217`. This indicates molecular-size descriptors
carry some useful information for chemical-family extrapolation, but the effect
is modest relative to the total scaffold-holdout error.

## Decision

Keep molecular-size descriptors in the main model, but interpret their
importance cautiously. `MolWt` and related descriptors should be described as
chemically meaningful size, polarity/surface, and structural-complexity proxies
that are also partly coupled to the pTox unit-conversion scale. They should not
be written as an independent mechanistic proof that molecular weight itself
causes toxicity.

No immediate full 5-fold expansion is required for internal decision-making.
If this sensitivity check becomes a manuscript claim, expand to scaffold
5-fold and/or multiple seeds before treating the small scaffold-holdout loss as
a stable quantitative estimate.
