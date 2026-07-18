# v1.2.28 Species-Endpoint Traditional ML Baseline Summary

Date: 2026-07-06

## Purpose

This run builds local traditional machine-learning baselines for each species-endpoint subtask. The inputs are restricted to a fixed literature-guided RDKit 2D descriptor union plus effect-level numeric features. Species identity, taxonomy, medium/domain labels, endpoint labels, and target/concentration fields are excluded from model inputs.

## Scope

- Source tables:
  - `aggregated_task_records_aquatic_ptox_qc`
  - `aggregated_task_records_soil_ptox_qc`
- Subtasks:
  - Aquatic: top 30 species-endpoint combinations.
  - Soil: top 30 species-endpoint combinations.
- Split: row-random 8:2 train/validation within each subtask, seed 42.
- Models: XGBoost, LightGBM, Random Forest, KNN, PLS.
- HPO: Optuna TPE, `n_trials=3`, objective validation RMSE.
- Feature count: 121 total input features after descriptor selection.

## Outputs

- Raw output root: `outputs/experiments/v1_2_28_species_endpoint_ml_descriptor_effect_baselines_stable_pls`
- Curated summary root: `实验汇总/机器学习基线_分子描述符效应水平`
- Global metrics:
  - `指标表/traditional_ml_descriptor_effect_all_metrics.csv`
  - `指标表/traditional_ml_descriptor_effect_model_comparison_summary.csv`
  - `指标表/traditional_ml_descriptor_effect_best_by_subtask.csv`
  - `指标表/traditional_ml_descriptor_effect_hpo_trials.csv`
  - `指标表/traditional_ml_descriptor_effect_skipped_subtasks.csv`
- Per-task folders:
  - `水生/<species>/<endpoint>/图表`
  - `水生/<species>/<endpoint>/指标表`
  - `水生/<species>/<endpoint>/描述符表`
  - `水生/<species>/<endpoint>/预测值表`
  - same structure under `土壤/`.
- Descriptor-selection evidence:
  - `特征选择依据/descriptor_selection_rationale.md`
  - `特征选择依据/descriptor_selection_reference.csv`

## Completion

- Expected model fits: 300.
- Completed model fits: 299.
- Skipped: 1 PLS fit for `Oryza sativa / ECx_GeneticDamage`.
- Figures: 299 model scatter plots, each exported as PNG and SVG, total 598 figure files.
- Compound descriptor tables: 60.

The skipped PLS fit was kept as an audit record. The task matrix and target vector were finite, but sklearn PLS produced internal NaN loadings even after conservative one-component fallback. Substituting a different linear method under the PLS label was not used.

## Model Comparison

Weighted validation metrics by domain:

| domain | model | subtasks | mean val R2 | median val R2 | weighted val RMSE | weighted val MAE |
|---|---:|---:|---:|---:|---:|---:|
| aquatic | LightGBM | 30 | 0.6692 | 0.7450 | 0.8374 | 0.5911 |
| aquatic | XGBoost | 30 | 0.6681 | 0.7528 | 0.8465 | 0.6007 |
| aquatic | Random Forest | 30 | 0.6570 | 0.7275 | 0.8684 | 0.6295 |
| aquatic | KNN | 30 | 0.5962 | 0.6406 | 0.9428 | 0.6525 |
| aquatic | PLS | 30 | -1.9421 | 0.4729 | 1.4661 | 0.9610 |
| soil | XGBoost | 30 | 0.3234 | 0.4301 | 0.9084 | 0.6623 |
| soil | Random Forest | 30 | 0.3216 | 0.3563 | 0.9416 | 0.7200 |
| soil | KNN | 30 | 0.2482 | 0.2461 | 0.9813 | 0.7082 |
| soil | LightGBM | 30 | 0.0951 | -0.0072 | 1.0459 | 0.8129 |
| soil | PLS | 29 | 0.1825 | 0.3835 | 1.1060 | 0.8174 |

Best-model counts:

| domain | model | best subtasks |
|---|---:|---:|
| aquatic | LightGBM | 15 |
| aquatic | XGBoost | 10 |
| aquatic | Random Forest | 5 |
| soil | XGBoost | 13 |
| soil | Random Forest | 6 |
| soil | PLS | 6 |
| soil | KNN | 5 |

## Eisenia fetida

`Eisenia fetida / ECx_Mortality` had 117 rows and was modeled successfully.

| model | val R2 | val RMSE | val MAE |
|---|---:|---:|---:|
| XGBoost | 0.5704 | 1.5956 | 1.1495 |
| Random Forest | 0.5002 | 1.7212 | 1.4056 |
| LightGBM | 0.4335 | 1.8324 | 1.5529 |
| PLS | 0.3994 | 1.8868 | 1.3828 |
| KNN | 0.2209 | 2.1489 | 1.4395 |

Other `Eisenia fetida` endpoints such as `ECx_Growth` had too few rows for the configured minimum train/validation thresholds. They are preserved in descriptor/skipped audit outputs where applicable.

## Interpretation Boundary

This is a descriptor-only same-subtask interpolation baseline. It is useful for judging whether classic QSAR descriptors plus effect level can recover within-species and within-endpoint variation. It should not be used as evidence of cross-species transfer, cross-endpoint transfer, or chemical-family/scaffold extrapolation.

Ecotoxicological interpretation should focus on the contrast between:

- ECx mortality and immobilization tasks, where descriptor-only nonlinear models often perform strongly.
- NOEC/LOEC biochemical, genetic damage, and plant growth tasks, where performance is more variable and often weaker, consistent with smaller effective sample sizes, heterogeneous experimental designs, and noisier endpoint definitions.

The selected descriptor families support mechanistic interpretation through hydrophobic partitioning, molecular size, polarity/H-bonding, ring/flexibility, topology/branching, electronic state, charge distribution, and VSA descriptors.
