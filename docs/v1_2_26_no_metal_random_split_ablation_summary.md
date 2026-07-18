# v1.2.26 No-Metal/Inorganic Random-Split Targeted Ablation Summary

Date checked: 2026-07-06 12:28 (+08:00)

## Scope

This run repeats the v1.2.21 targeted ablation set on the v1.2.22
no-metal/inorganic random transfer splits, using only the best single seed
from v1.2.22 (`2042`).

- Database: `outputs/derived/modeling_dataset_v2_0_0_rebuild_no_metal_inorganic.sqlite`
- Source table: `aggregated_task_records_aquatic_soil_ptox_qc_no_metal_inorganic`
- Launcher: `scripts/run_v1_2_26_no_metal_random_split_ablation_remote.sh`
- Summary: `outputs/experiments/v1_2_26_no_metal_random_split_ablation_remote_summary`
- Runtime table: `outputs/logs/run_v1_2_26_no_metal_random_split_ablation_times.csv`

## Completion

The remote queue completed on 2026-07-03 22:36:54 (+08:00).

- Formal runs: 36/36
- Exit codes: all `0`
- Total training time: 15.18 h
- Mean time per run: 25.30 min
- Range: 20.90-29.30 min per run

## Baseline

Baseline is the v1.2.22 no-metal full model with seed `2042`.

| Split policy | n | RMSE | MAE | Huber |
|---|---:|---:|---:|---:|
| random 8:2 | 2608 | 0.9320 | 0.6597 | 0.3376 |
| random 5-fold | 13063 | 0.9324 | 0.6554 | 0.3312 |

For random 5-fold, MAE and Huber are fold-test-size weighted; RMSE is computed
as `sqrt(sum(n * RMSE^2) / sum(n))`. Fold-level R2 is not used as the primary
combined metric because exact combined R2 cannot be reconstructed from
fold-level summaries alone.

## Main Result

Positive `delta_MAE` means the ablation is worse than the seed2042 full
baseline.

| Ablation | random 8:2 MAE | random 8:2 delta_MAE | random 5-fold MAE | random 5-fold delta_MAE |
|---|---:|---:|---:|---:|
| no_context | 0.8727 | +0.2131 | 0.8920 | +0.2366 |
| no_species_lifestage | 0.7636 | +0.1040 | 0.7505 | +0.0952 |
| no_toxicity_binning | 0.6911 | +0.0315 | 0.6945 | +0.0391 |
| no_source_weighting | 0.6688 | +0.0091 | 0.6686 | +0.0133 |
| no_censored_loss | 0.6461 | -0.0136 | 0.6601 | +0.0047 |
| no_molecular_residual | 0.6505 | -0.0092 | 0.6579 | +0.0025 |

## Interpretation

The no-metal/inorganic targeted ablation reproduces the main v1.2.21 pattern:

1. Context features remain the largest contributor under random interpolation.
2. Species/lifestage features remain the second-largest contributor.
3. Toxicity binning has a moderate regularization benefit.
4. Source weighting and censored loss are small under random interpolation.
5. Molecular residual features do not show a robust positive contribution in
   this random-split no-metal setting.

This should be interpreted as an organic-chemical descriptor-applicability
sensitivity result under random interpolation. It does not replace the
scaffold/cluster chemical-family extrapolation evidence from v1.2.24.
