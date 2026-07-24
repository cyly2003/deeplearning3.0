# Record-level applicability-domain analysis

This directory implements the locked handoff contract for the v1.2.44 M10 model and the strict M00 baseline. It is an analysis-only workflow: it reuses saved splits, preprocessing contracts and row-wise predictions and does not retrain the model.

## Execution order

1. `00_discover_inputs.py`
2. `01_reproduce_locked_metrics.py`
3. `02_build_chemical_space.py`
4. `03_build_support_features.py`
5. `04_calibrate_ad_on_validation.py`
6. `06_evaluate_ad_on_test.py`
7. `07_transfer_gain_by_support.py`
8. `08_make_main_figures.py`
9. `09_make_supplementary_figures.py`
10. `10_export_tables_and_reports.py`
11. `python -m pytest analysis/applicability_domain/tests -q`

The workflow must stop if the discovery identity hashes or locked M10/M00 test metrics fail.

## Locked statistical choices

- Stable row identity is the strict `stage_sample_v1` composite, not the display `sample_id`.
- The formal taxonomy rule uses same-task taxonomic support; global taxonomic support is descriptive because the shared trunk can transfer across heads.
- Chemical-neighbor counts use unique canonical parents. Joint local support counts Stage-3 training records, as required by the handoff formula.
- Similarity thresholds use `>=` consistently.
- Four-seed prediction SD uses `ddof=1` and is only a disagreement diagnostic.
- Numeric context missingness uses distance 0 when both values are missing and 1 when only one is missing. Semantic missing categories such as `NR` are retained as an explicit state and also contribute to `context_missing_fraction`.
- `effect_level_x` is included conditionally within task. For tasks where it is structurally absent in both records, it contributes zero pair distance rather than excluding those tasks.
- `primary_medium` is excluded from B_exp because it is a derived summary redundant with the included media/habitat fields; `target_basis` is constant and `effect_family` is determined by the task head.

## Final scientific status

No one of 384 validation-only candidates met the predeclared coverage, tier-size and task-composition criteria. This prevents use as a calibrated rejection rule, but it does not identify unreliable records or tasks. Intermediate and Lower-measured support cover 98.2% of the outer test and differ from the full-test MAE by only −0.030 and +0.033, respectively. The strict High subset contains 54 records and is interpreted as a small favorable-support subset. The final label is therefore **training-support stratification**, with support labels treated as density descriptors rather than pass/fail reliability classes. See `AD_RESULTS_SUMMARY_CN.md` and `AD_TEST_EVALUATION_REPORT.md`.

## Figure contract

Core conclusion: prediction performance is broadly similar across the two large support strata; joint support provides a modest ranking signal, while the small strict-High subset does not justify a reliable/unreliable boundary.

- Archetype: schematic-led quantitative composite.
- Backend: Python only.
- Target: ES&T-style double-column figure, 183 mm wide, editable SVG/PDF plus 600-dpi PNG/TIFF.
- Hero panel: one C-B-T concept cube; joint local density is color, not a fourth axis.
- Validation panels: conditional error heatmaps, coverage-error curves and tier-wise error distributions.
- Palette: low-saturation blue-teal-gold surface colors with translucent gray threshold planes, based on the user-supplied reference.
- Reviewer risks: random-split interpolation, high exact-parent coverage, task-composition confounding, structure-unavailable records, and any weak validation-to-test support-error trend must remain visible.
