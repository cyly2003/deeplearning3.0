# v1.2.48 Random 8:2 structure-similarity coverage analysis

## Scientific question

Within the locked v1.2.44 random 8:2 evaluation boundary, how does the
selected model's predictive reliability vary with the structural coverage of a
test chemical? This is a coverage/reliability analysis, not a replacement for
the strict v1.2.47 scaffold/similarity-family external holdout.

## Locked model and boundary

- Model: v1.2.44 M10 (aquatic pTox pretraining then direct full-parameter soil
  mol/kg fine-tuning), seed 42.
- Selection: minimum validation MAE among the completed comparable candidates;
  outer test predictions were not used to choose the model or seed.
- Evaluation: the same locked v1.2.44 random 8:2 outer test (`n=3,042` before
  structure-resolution exclusions).
- Scale: `neg_log10_mol_kg`.

## Coverage scores

Each test chemical receives the maximum Morgan Tanimoto similarity (`radius=2`,
`2,048` bits) to two reference spaces:

1. **Primary, all-model-fit coverage:** every chemical actually fitted in M10
   Stage 1 or Stage 3. This is the relevant axis for a transfer model because
   a chemical encountered in aquatic pretraining is not chemically unseen by
   the complete learning chain.
2. **Sensitivity, Stage-3 target-domain coverage:** Stage-3 soil training
   chemicals only. This distinguishes target-domain proximity from the broader
   transferable source-domain knowledge.

Murcko scaffold identity is retained as an audit field, but the continuous
score is Morgan similarity because it is the measure used by the existing
`0.65` structure-family rule.

## Analysis matrix

No model is retrained. All rows below use the same fixed M10 seed-42 outer-test
predictions.

| Module | Coverage condition | Scientific role |
|---|---|---|
| Reference | all test rows | reproduces the random 8:2 reporting boundary |
| Nested threshold comparison | `Smax < 0.80`, `< 0.65`, `< 0.40`, `< 0.20` | progressively restricts analysis to less covered chemicals |
| Exclusive coverage curve | `>=0.80`, `0.65–0.80`, `0.40–0.65`, `0.20–0.40`, `<0.20` | reveals the performance pattern without overlapping strata |

Before calculation, any exclusive bin with fewer than 100 prediction rows or
10 canonical parents is deterministically merged only with an adjacent
similarity interval. This rule uses sample support alone, never predictions or
performance values. Nested threshold subsets below the same support gate are
marked descriptive-only.

If this support rule collapses the primary all-model-fit axis to one interval,
the analysis is formally marked infeasible for a coverage-decay curve. It may
still be retained as an audit showing that the random split is dominated by
chemicals already seen during fitting, but it must not be presented as evidence
of chemical extrapolation.

## Outcomes and statistics

Every supported condition reports `n`, canonical-parent count, task count,
MAE, RMSE, ordinary R2, within-task centred R2, and macro task-MAE. The latter
two guard against different task composition across coverage strata.

Uncertainty is estimated by 5,000 percentile bootstrap resamples of test
canonical parents. This keeps repeated measurements of the same chemical in
the same resampling unit. The resulting intervals are descriptive uncertainty
intervals for coverage strata; overlapping nested thresholds are not treated as
independent significance tests.

## Reporting boundary

The random 8:2 `Smax < 0.65` subset must be described as a low-coverage
interpolation subset. It is not equivalent to v1.2.47, where test structure
families and all upstream source structures were actively isolated before
training.
