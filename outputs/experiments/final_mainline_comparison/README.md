# Final Mainline Comparison Summary

This directory harmonizes the existing final result tables for the current QSAR transfer mainline.

## Sources

- Chemical-holdout mainline: `outputs\experiments\v1_2_12_f100_5seed_confirmation_remote_summary`.
- Random split reference: `outputs\experiments\v1_2_15_random_split_policy_formal_remote_summary`.

## Files

- `existing_final_overall_summary.csv`: overall ECx/LOEC/NOEC 30-task metrics.
- `existing_final_family_summary_30task.csv`: endpoint-family metrics for ECx, LOEC, and NOEC.
- `existing_final_task_summary_30task.csv`: task-level main-scope metrics.
- `existing_final_task_summary_35task.csv`: task-level full trained-head metrics, including ICx/LDx when present.
- `chemical_holdout_single_seed_summary.csv`: fixed-test single-seed anchor metrics for comparison with the 5-seed ensemble.

## Interpretation Boundary

- `chemical_holdout_f100_5seed` is the current mainline external-generalization result.
- Random split rows are same-distribution/interpolation references; the seed count is encoded in each `evaluation_policy` label.
- Ensemble metrics are computed by averaging aligned prediction rows across seeds; they should be reported separately from single-model mean +/- SD.

## Current Overall Metrics

| evaluation_policy           | n     | r2     | rmse   | mae    | huber_loss | ensemble_seed_count |
| --------------------------- | ----- | ------ | ------ | ------ | ---------- | ------------------- |
| chemical_holdout_f100_5seed | 2594  | 0.5388 | 1.2368 | 0.9277 | 0.5563     | 5                   |
| random_5fold_5seed          | 15630 | 0.7878 | 0.8512 | 0.5939 | 0.2852     | 5                   |
| random_8_2_5seed            | 3165  | 0.7853 | 0.8521 | 0.5925 | 0.2858     | 5                   |

## Main Task Count

- Main-scope task rows: 90.
