# v1.2.42 E-series OOF summary

## Evaluation boundary

This experiment evaluates prediction-level adaptation inside the locked
v1.2.40 outer stage-3 training population. Direct and Transfer base learners
were trained in five OOF folds for screening seeds 42 and 3407. The held-out
OOF fold was used only for prediction, each training stage used an internal
training-fold monitor, and the outer test was excluded from all OOF splits.

The meta-selection population came from the fixed v1.2.40 seed42 stage-3
validation identities. Of 2,433 requested identities, 2,325 had paired Direct
and Transfer OOF predictions. The 108 excluded identities all belonged to task
routes absent from the complete paired OOF task space because the outer-train-
only sample count was below the model's minimum-support threshold:

- ECx_Population__solid_neglog_mol_kg: 32
- LOEC_GeneticDamage__solid_neglog_mol_kg: 38
- NOEC_Physiology__solid_neglog_mol_kg: 38

The summary implementation records the excluded identity hash and exact route
counts. It still fails closed if an identity is missing from a task route that
is represented in OOF predictions.

## Validation-only result

| Candidate | Description | n | R2 | RMSE | MAE | Delta R2 vs E0 | Delta MAE vs E0 | Eligible |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E0 | frozen Transfer ensemble anchor | 2325 | 0.656425 | 0.821628 | 0.582290 | 0 | 0 | reference |
| E1 | constrained Direct-Transfer blend | 2325 | 0.655596 | 0.822619 | 0.581000 | -0.000830 | -0.001290 | no |
| E2 | contextual sigmoid gate | 2325 | 0.656414 | 0.821642 | 0.582328 | -0.000012 | +0.000038 | no |
| E3 | bounded residual head | 2325 | 0.656228 | 0.821863 | 0.580657 | -0.000197 | -0.001634 | no |

No E1-E3 candidate improved both native `-log10(mol/kg)` R2 and MAE relative
to E0. E1 and E3 slightly reduced MAE but also slightly reduced R2; E2 was
effectively unchanged and marginally worse on both criteria. The locked result
is therefore `selected_candidate=null`.

## Decision and interpretation

The matrix stopped without generating expansion seeds 2042 and 8417. The
outer test was not read, so there are intentionally no E-series final-test or
common-mg/kg metrics. The result does not show a useful prediction-level gain
over the frozen Transfer anchor under the simultaneous R2-and-MAE decision
rule.

The OOF validation R2 of about 0.656 must not be compared directly with the
v1.2.40 four-seed full-fit outer-test R2 of about 0.697. OOF base learners use
only four fifths of the stage-3 training population, and the evaluation
populations are different. The v1.2.40 outer-test result remains the current
full-fit random-8:2 result.

## Traceability

- Remote selection JSON: `outputs/experiments/v1_2_42_e_series_summary/selected_candidate.json`
- Remote validation CSV: `outputs/experiments/v1_2_42_e_series_summary/validation_candidate_summary.csv`
- Tracked metric CSV: `docs/v1_2_42_e_series_validation_metrics.csv`
- Controller completion marker: `[matrix_complete_no_winner]` at
  `2026-07-19T13:53:17+08:00`
- Selection identity SHA256:
  `34d37e781a9c8db58da1e1e823026afb2c728706f940ee14ee339efd937c7cf9`
- Excluded identity SHA256:
  `0d00019d3ffe08cd24016242cf18041881b9922eb1a4e26dfeda5e8a9f1c6aaa`
