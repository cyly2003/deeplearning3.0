# v1.2.29 Soil Species-Endpoint Traditional ML Expanded Baseline

Date: 2026-07-06

## Purpose

This run expands the soil species-endpoint traditional ML baseline after the v1.2.28 top-30 soil run was judged too narrow for species coverage.

## Scope

- Source table: `aggregated_task_records_soil_ptox_qc`
- Domain: soil only
- Subtask rule: all soil species-endpoint combinations with `n >= 30`
- Subtasks modeled: 73
- Species covered: 24
- Split: row-random 8:2 train/validation within each subtask, seed 42
- Minimum split thresholds: `min_train=24`, `min_validation=6`
- Models: XGBoost, LightGBM, Random Forest, KNN, PLS
- HPO: Optuna TPE, `n_trials=3`, objective validation RMSE
- Features: same fixed 121-feature descriptor/effect-level input set as v1.2.28

## Outputs

- Raw output root: `outputs/experiments/v1_2_29_soil_species_endpoint_ml_descriptor_effect_n30`
- Summary root: `实验汇总/机器学习基线_分子描述符效应水平_土壤扩展n30`
- Figures: 364 model scatter plots, PNG and SVG, total 728 figure files
- Descriptor tables: 73

## Completion

- Expected model fits: 365
- Completed model fits: 364
- Skipped: 1 PLS fit for `Oryza sativa / ECx_GeneticDamage`

The skipped PLS fit is the same numerical-degeneracy issue seen in v1.2.28. The input matrix and target vector are finite, but sklearn PLS produces internal NaN loadings even with a one-component fallback.

## Soil Species Covered

Acanthamoeba castellanii; Allium cepa; Arabidopsis thaliana; Avena sativa; Caenorhabditis elegans; Cucumis sativus; Daucus carota; Eisenia andrei; Eisenia fetida; Glomus intraradices; Glycine max; Hordeum vulgare; Lactuca sativa; Lumbricus rubellus; Medicago sativa; Nicotiana tabacum; Oryza sativa; Phaseolus vulgaris; Pisum sativum; Raphanus sativus; Setaria viridis; Solanum lycopersicum var. lycopersicum; Triticum aestivum; Zea mays.

## Model Comparison

Weighted validation metrics:

| model | subtasks | mean val R2 | median val R2 | weighted val RMSE | weighted val MAE |
|---|---:|---:|---:|---:|---:|
| XGBoost | 73 | -0.0043 | 0.1951 | 1.0195 | 0.7568 |
| Random Forest | 73 | 0.1001 | 0.1779 | 1.0334 | 0.7969 |
| KNN | 73 | 0.0237 | 0.1495 | 1.0567 | 0.7839 |
| LightGBM | 73 | -0.1102 | -0.0296 | 1.1536 | 0.9113 |
| PLS | 72 | -0.9329 | 0.1141 | 1.2353 | 0.8954 |

## Interpretation

This expanded run is more complete for soil species coverage than v1.2.28, but the lower `n >= 30` threshold makes many subtasks statistically fragile. It should be used for figure completeness and exploratory species-endpoint comparison, while conclusions should prioritize larger subtasks and avoid overinterpreting small validation sets.

The drop in average performance compared with the top-30 soil run is expected: the expanded set includes many lower-sample and more heterogeneous NOEC/LOEC/ICx endpoints.
