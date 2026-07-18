# Paper Figure Design Plan

Date: 2026-07-06

This note records the planned paper-facing figure set, data sources, panel
layout, and terminology for the QSAR transfer-learning manuscript figures.

## External terminology

Avoid the internal term "mainline" in paper figures, legends, and captions.
Use the following paper-facing terms instead:

- selected metal- and inorganic-excluded transfer-learning model
- selected modeling strategy
- metal- and inorganic-excluded dataset
- random 80:20 split
- random five-fold cross-validation
- traditional descriptor/effect-level ML baselines
- targeted ablation study
- model interpretation or feature contribution

## Shared visual style

- English journal figures: Arial.
- Chinese thesis figures: Microsoft YaHei or SimHei.
- Detailed typography, color, marker, and dense-panel hierarchy rules are centralized in `docs/paper_figure_visual_hierarchy_20260712.md`; follow that document for subsequent manuscript figures.
- Figure background: white.
- Export formats: SVG plus PNG at 600 dpi.
- Color scheme: Van Gogh's *The Starry Night* inspired, colorblind-aware manuscript palette.
- Unified manuscript palette source: `outputs/paper_figures/fig_model_results_integrated_20260709/performance/figure_palette_starry_night_v1.yaml`.
- Fixed metric colors for all subsequent performance/result figures:
  - `R²`: star yellow `#F2C94C`;
  - `RMSE`: warm orange `#D8892B`;
  - `MAE`: cobalt blue `#2F6DB3`.
- Metric colors must not change between figures. When model identity also needs encoding, use marker shape and line style rather than changing the metric color.
- Fixed model marker convention:
  - selected transfer-learning model: circle `o`;
  - XGBoost: pentagon/filled plus `P`;
  - LightGBM: triangle `^`;
  - Random Forest: diamond `D`;
  - KNN: star `*`;
  - PLS: downward triangle `v`;
  - ExtraTrees: square `s` when v1.2.27 endpoint baselines are plotted.
- Domain/support colors may use Starry-Night blues, cypress green, muted gray, and ivory callouts from the palette file.
- Use 30-50% alpha for dense points, bars, or filled bands. Add a thin light-gray
  edge (`#D0D5DD` or similar) when colored marks merge into one large color block.
- Heatmaps should avoid large connected white missing-value blocks. If a heatmap
  becomes mostly empty or visually dominated by white cells, replace it with a
  bubble/dot plot using size for sample count and color for performance.
- Axes should use `Observed pTox` and `Predicted pTox` for prediction plots.
- Metrics displayed in result panels may include `R²`, `RMSE`, `MAE`, `Huber loss`, and `n`; when plotted as colored series, use the fixed metric colors above.
- Composite panel labels use aligned lowercase parenthetical labels, e.g.,
  `(a)`, `(b)`, `(c)`. Panel labels are added only at the composite assembly
  layer, not inside the reusable panel drawing functions.
- Standalone panel exports must not contain `(a)`, `(b)`, or similar panel
  labels. The file name and caption notes identify each panel slot.
- Use bold font weight for figure text intended for manuscript submission,
  including axis labels, tick labels, legends, titles, and schematic labels.
- Keep plot text minimal. Axis labels should contain only variable names and
  necessary units. Legends should only define color, marker, or line mappings.
  Do not use long plot titles to explain results. Split names such as `80:20`
  and `5-fold CV` should be omitted from panel titles when the caption and file
  names already identify the split.
- Each composite and each standalone panel may have one short figure title.
  Full title explanations, bilingual captions, and interpretation notes should
  be centralized in `figure_titles_and_captions.md` for the relevant figure
  directory.
- Figure architecture references are structural only:
  - SHAP: horizontal feature-importance/strip layout with subtle row bands and a
    compact colorbar or category legend only when supported by real SHAP data.
  - PDP: trend line with uncertainty band, observed-value distribution at the
    bottom, sparse threshold/median reference lines only when scientifically
    defined.
  - Prediction diagnostics: observed-predicted scatter with marginal
    distributions and a residual panel, using transparent points and light-gray
    point edges for high-density regions.
  - Radar plots may be used for ablation summaries and subtask-performance
    overviews, but subtask labels should be moved to supporting tables rather
    than printed around the radar.

## Figure GA: Graphical Abstract

### Purpose

Summarize the study logic for editors/readers:

metal- and inorganic-excluded toxicity data -> transfer-learning QSAR ->
random 80:20 and random five-fold validation -> interpretable pTox prediction
-> ecological risk relevance.

### Data sources

- `PROJECT_STATUS.md`
- `docs/experiment_registry.csv`
- `docs/experiment_decision_log.md`
- `docs/v1_2_26_no_metal_random_split_ablation_summary.md`

### Composite output

- `outputs/paper_figures/fig1_workflow_and_graphical_abstract/graphical_abstract_composite.svg`
- `outputs/paper_figures/fig1_workflow_and_graphical_abstract/graphical_abstract_composite.png`
- `outputs/paper_figures/fig1_workflow_and_graphical_abstract/figure_titles_and_captions.md`

### Required components

- GA-A: metal- and inorganic-excluded data curation.
- GA-B: transfer-learning QSAR model.
- GA-C: validation using random 80:20 and random five-fold split.
- GA-D: pTox prediction and interpretation.
- GA-E: ecological risk relevance.

## Figure 1: Modeling Workflow

### Purpose

Explain the complete computational workflow without exposing internal
experiment names.

### Data sources

- `PROJECT_STATUS.md`
- `docs/experiment_registry.csv`
- `docs/experiment_decision_log.md`
- `docs/v1_2_26_no_metal_random_split_ablation_summary.md`

### Composite output

- `outputs/paper_figures/fig1_workflow_and_graphical_abstract/fig1_modeling_workflow_composite.svg`
- `outputs/paper_figures/fig1_workflow_and_graphical_abstract/fig1_modeling_workflow_composite.png`
- `outputs/paper_figures/fig1_workflow_and_graphical_abstract/figure_titles_and_captions.md`

### Required panels

- Fig. 1A: data curation and exclusion of inorganic and metal/metalloid records.
- Fig. 1B: molecular, species/context, effect-level, and quality/source inputs.
- Fig. 1C: transfer-learning QSAR architecture.
- Fig. 1D: validation, performance metrics, ablation, and interpretation outputs.

## Figure 2: Predictive Performance of the Selected Model

### Purpose

Show overall predictive reliability, validation-boundary sensitivity, and
task-level heterogeneity for the selected metal- and inorganic-excluded
transfer-learning model.

### Data sources

- `outputs/experiments/v1_2_22_no_metal_random_split_remote_summary/split_policy_ensemble_combined_summary.csv`
- `outputs/experiments/v1_2_22_no_metal_random_split_remote_summary/split_policy_ensemble_fold_summary.csv`
- `outputs/experiments/v1_2_22_no_metal_random_split_remote_summary/split_policy_ensemble_random_8_2_holdout_prediction_rows.csv`
- `outputs/experiments/v1_2_22_no_metal_random_split_remote_summary/split_policy_ensemble_random_5fold_fold1_prediction_rows.csv`
- `outputs/experiments/v1_2_22_no_metal_random_split_remote_summary/split_policy_ensemble_random_5fold_fold2_prediction_rows.csv`
- `outputs/experiments/v1_2_22_no_metal_random_split_remote_summary/split_policy_ensemble_random_5fold_fold3_prediction_rows.csv`
- `outputs/experiments/v1_2_22_no_metal_random_split_remote_summary/split_policy_ensemble_random_5fold_fold4_prediction_rows.csv`
- `outputs/experiments/v1_2_22_no_metal_random_split_remote_summary/split_policy_ensemble_random_5fold_fold5_prediction_rows.csv`
- `outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_remote_summary/split_policy_ensemble_combined_summary.csv`
- `outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_remote_summary/split_policy_ensemble_fold_summary.csv`
- `outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_remote_summary/split_policy_ensemble_scaffold_cluster_8_2_holdout_prediction_rows.csv`
- `outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_remote_summary/split_policy_ensemble_scaffold_cluster_5fold_fold1_prediction_rows.csv`
- `outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_remote_summary/split_policy_ensemble_scaffold_cluster_5fold_fold2_prediction_rows.csv`
- `outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_remote_summary/split_policy_ensemble_scaffold_cluster_5fold_fold3_prediction_rows.csv`
- `outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_remote_summary/split_policy_ensemble_scaffold_cluster_5fold_fold4_prediction_rows.csv`
- `outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_remote_summary/split_policy_ensemble_scaffold_cluster_5fold_fold5_prediction_rows.csv`
- `outputs/paper_figures/fig2_selected_model_performance/tables/fig2_task_radar_data.csv`
- `outputs/paper_figures/fig2_selected_model_performance/tables/fig2_task_code_mapping.csv`

### Confirmed summary metrics

- Random five-fold: `n=13063`, `R²=0.7920`, `MAE=0.6200`,
  `Huber loss=0.3057`.
- Random 80:20: `n=2608`, `R²=0.7895`, `MAE=0.6243`,
  `Huber loss=0.3098`.
- Scaffold/similarity-cluster five-fold: `n=11459`, `R²=0.3664`,
  `MAE=1.1178`, `Huber loss=0.7224`.
- Scaffold/similarity-cluster 80:20: `n=2493`, `R²=0.1810`,
  `MAE=1.0664`, `Huber loss=0.6746`.

### Composite output

- `outputs/paper_figures/fig2_selected_model_performance/fig2_selected_model_performance_composite.svg`
- `outputs/paper_figures/fig2_selected_model_performance/fig2_selected_model_performance_composite.png`
- `outputs/paper_figures/fig2_selected_model_performance/figure_titles_and_captions.md`

### Required panels

- Fig. 2A: random holdout `Observed pTox` vs `Predicted pTox` with marginal
  distributions and residual density.
- Fig. 2B: random cross-validation `Observed pTox` vs `Predicted pTox` with
  marginal distributions and residual density.
- Fig. 2C: scaffold/similarity-cluster holdout `Observed pTox` vs
  `Predicted pTox` with marginal distributions and residual density.
- Fig. 2D: scaffold/similarity-cluster cross-validation `Observed pTox` vs
  `Predicted pTox` with marginal distributions and residual density.
- Fig. 2E-H: task-wise clipped `R²` radar summaries for the same four
  validation policies using the shared task set; task labels are omitted from
  the plotted radar panels and provided in the source table.

## Figure 3: Model Interpretation

### Purpose

Show which features or feature groups explain the selected model's pTox
predictions.

### Data sources to verify

- `scripts/explain_deep_model.py`
- Candidate output files, if present:
  - `shap_feature_importance.csv`
  - `permutation_importance.csv`
  - `pdp_duration.csv`
- Candidate model run files, if present:
  - `best_model.pt`
  - `preprocessing.json`
  - `manifest.json`

### Current caveat

SHAP outputs were not yet confirmed in the first local scan. Do not draw SHAP
panels unless real SHAP or explainability files exist or are generated from a
real model checkpoint.

### Composite output

- `outputs/paper_figures/fig3_explainability/fig3_explainability_composite.svg`
- `outputs/paper_figures/fig3_explainability/fig3_explainability_composite.png`
- `outputs/paper_figures/fig3_explainability/figure_titles_and_captions.md`

### Required panels

- Fig. 3A: SHAP or permutation top feature importance.
- Fig. 3B: feature-group contribution summary.
- Fig. 3C: selected dependence or PDP plots for interpretable variables.
- Fig. 3D: optional task/family-level explanation panel.
- Fig. 3E: molecular-descriptor-only SHAP summary, if real target-model SHAP
  output can be generated.
- Fig. 3F: molecular-descriptor-only PDP for selected descriptors, if real
  target-model PDP output can be generated.

## Figure 4: Targeted Ablation Study

### Purpose

Show which input/learning components contribute most to prediction accuracy
after removing inorganic and metal/metalloid records.

### Data sources

- `outputs/experiments/v1_2_26_no_metal_random_split_ablation_remote_summary/best_by_split.csv`
- `outputs/experiments/v1_2_26_no_metal_random_split_ablation_remote_summary/common_task_summary.csv`
- `outputs/experiments/v1_2_26_no_metal_random_split_ablation_remote_summary/common_task_comparison.csv`
- `outputs/experiments/v1_2_26_no_metal_random_split_ablation_remote_summary/focus_summary.csv`
- `outputs/experiments/v1_2_26_no_metal_random_split_ablation_remote_summary/audit_summary.csv`
- `docs/v1_2_26_no_metal_random_split_ablation_summary.md`

### Baseline definition

The full-model reference is the v1.2.22 metal- and inorganic-excluded full
model with seed `2042`.

### Confirmed MAE deltas

Positive `delta_MAE` means worse than the full model.

- `no_context`: `+0.2131` for random 80:20, `+0.2366` for random five-fold.
- `no_species_lifestage`: `+0.1040` for random 80:20, `+0.0952` for random five-fold.
- `no_toxicity_binning`: `+0.0315` for random 80:20, `+0.0391` for random five-fold.
- `no_source_weighting`: `+0.0091` for random 80:20, `+0.0133` for random five-fold.
- `no_censored_loss`: `-0.0136` for random 80:20, `+0.0047` for random five-fold.
- `no_molecular_residual`: `-0.0092` for random 80:20, `+0.0025` for random five-fold.

### Composite output

- `outputs/paper_figures/fig4_ablation_study/fig4_ablation_study_composite.svg`
- `outputs/paper_figures/fig4_ablation_study/fig4_ablation_study_composite.png`
- `outputs/paper_figures/fig4_ablation_study/figure_titles_and_captions.md`

### Required panels

- Fig. 4A: `ΔMAE` relative to the full model.
- Fig. 4B: `ΔHuber loss` relative to the full model.
- Fig. 4C: `ΔR²` relative to the full model.

## Figure 5: Baselines and Subtask-Level Performance

### Purpose

Compare the selected transfer-learning model with traditional
descriptor/effect-level ML baselines while preserving validation-boundary
differences. Also show subtask-level behavior.

### Data sources

Selected transfer-learning model:

- `outputs/experiments/v1_2_22_no_metal_random_split_remote_summary/split_policy_ensemble_combined_summary.csv`
- `outputs/experiments/v1_2_22_no_metal_random_split_remote_summary/common_task_summary.csv`
- `outputs/experiments/v1_2_22_no_metal_random_split_remote_summary/common_task_comparison.csv`
- `outputs/experiments/v1_2_22_no_metal_random_split_remote_summary/effect_level_summary.csv`
- `outputs/experiments/v1_2_22_no_metal_random_split_remote_summary/family_summary.csv`
- `outputs/experiments/v1_2_22_no_metal_random_split_remote_summary/toxicity_bin_summary.csv`

Traditional ML baselines:

- `outputs/experiments/v1_2_27_traditional_ml_descriptor_effect_baselines/tables/traditional_ml_descriptor_effect_best_scope_summary.csv`
- `outputs/experiments/v1_2_27_traditional_ml_descriptor_effect_baselines/tables/traditional_ml_descriptor_effect_all_metrics.csv`
- `outputs/experiments/v1_2_28_species_endpoint_ml_descriptor_effect_baselines_stable_pls/tables/traditional_ml_descriptor_effect_model_comparison_summary.csv`
- `outputs/experiments/v1_2_29_soil_species_endpoint_ml_descriptor_effect_n30/tables/traditional_ml_descriptor_effect_model_comparison_summary.csv`
- `outputs/experiments/v1_2_30_aquatic_species_endpoint_ml_descriptor_effect_n200/tables/traditional_ml_descriptor_effect_model_comparison_summary.csv`

### Validation-boundary caveat

Traditional ML baseline results are descriptor/effect-level, within-task
row-random validation. They are useful baseline and subtask references, but
they are not direct evidence of cross-domain transfer or chemical-family
extrapolation.

### Composite output

- `outputs/paper_figures/fig5_baseline_and_subtasks/fig5_baseline_comparison_composite.svg`
- `outputs/paper_figures/fig5_baseline_and_subtasks/fig5_baseline_comparison_composite.png`
- `outputs/paper_figures/fig5_baseline_and_subtasks/figure_titles_and_captions.md`

### Required panels

- Fig. 5A: selected model vs traditional ML baselines by `R²`.
- Fig. 5B: selected model vs traditional ML baselines by `MAE`.
- Fig. 5C: Huber loss, only for sources where the metric is available.
- Fig. 5D: subtask-level performance heatmap or ranked dot plot for the selected model.

### Supplemental subtask panels

- Fig. SxA: task-family performance.
- Fig. SxB: toxicity-bin performance.
- Fig. SxC: species-endpoint baseline coverage.
- Fig. SxD: ranked traditional ML model comparison by domain.
