# v1.2.30 Aquatic Species-Endpoint Traditional ML Expanded Baseline

Date: 2026-07-06

## Purpose

This run expands the aquatic species-endpoint traditional ML baseline after the v1.2.28 top-30 aquatic run was confirmed to be only a high-sample representative subset.

## Scope

- Source table: `aggregated_task_records_aquatic_ptox_qc`
- Domain: aquatic only
- Subtask rule: all aquatic species-endpoint combinations with `n >= 200`
- Subtasks modeled: 199
- Species covered: 97
- Split: row-random 8:2 train/validation within each subtask, seed 42
- Minimum split thresholds: `min_train=160`, `min_validation=40`
- Models: XGBoost, LightGBM, Random Forest, KNN, PLS
- HPO: Optuna TPE, `n_trials=3`, objective validation RMSE
- Features: same fixed 121-feature descriptor/effect-level input set as v1.2.28

## Outputs

- Raw output root: `outputs/experiments/v1_2_30_aquatic_species_endpoint_ml_descriptor_effect_n200`
- Summary root: `实验汇总/机器学习基线_分子描述符效应水平_水相扩展n200`
- Figures: 995 model scatter plots, PNG and SVG, total 1,990 figure files
- Descriptor tables: 199

## Completion

- Expected model fits: 995
- Completed model fits: 995
- Skipped: 0

## Model Comparison

Weighted validation metrics:

| model | subtasks | mean val R2 | median val R2 | weighted val RMSE | weighted val MAE |
|---|---:|---:|---:|---:|---:|
| LightGBM | 199 | 0.6406 | 0.6927 | 0.8778 | 0.6202 |
| XGBoost | 199 | 0.6359 | 0.6920 | 0.8798 | 0.6157 |
| Random Forest | 199 | 0.6233 | 0.6723 | 0.9101 | 0.6598 |
| KNN | 199 | 0.5576 | 0.6296 | 0.9762 | 0.6770 |
| PLS | 199 | 0.0657 | 0.4856 | 1.3223 | 0.9094 |

Best-model counts:

| model | best subtasks |
|---|---:|
| XGBoost | 91 |
| LightGBM | 64 |
| Random Forest | 24 |
| KNN | 13 |
| PLS | 7 |

## Interpretation

This run is much more complete for aquatic plotting than v1.2.28 while keeping a conservative enough sample threshold for stable validation estimates. It covers many additional fish, algae, crustacean, rotifer, mollusk, insect, amphibian, and plant endpoints.

The performance pattern is consistent with the top-30 run: nonlinear tree/boosting models are strongest, KNN is moderate, and PLS is weaker on average. XGBoost wins the largest number of individual subtasks, while LightGBM has the best weighted RMSE across all validation records.

This remains a within-species and within-endpoint row-random interpolation baseline. It should not be interpreted as cross-species, cross-endpoint, or new-chemical extrapolation evidence.
