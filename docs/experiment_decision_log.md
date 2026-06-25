# Experiment Decision Log

This file is the human-readable decision layer for the v1.2 transfer-learning
experiments. Use it with `docs/experiment_registry.csv`: the registry stores
where each result lives, while this log records why each strategy was kept or
stopped.

## Current Mainline

The current defensible mainline is:

- Data scope: soil pTox, same target scale, from
  `outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite`.
- Transfer split: `M_v2_aquatic_to_soil_ptox_adapt_C_f100`.
- Source weighting: `tanimoto_to_finetune`, `alpha=1.0`.
- Auxiliary target: authority-based toxicity bin CE, loss weight `0.025`.
- Censored loss: enabled, loss weight `0.01`.
- Finetune validation: keep `finetune_validation_fraction=0.2`.
- Robustness: report both 5-seed single-model mean and 5-seed ensemble.

Best current reporting result:

- `outputs/experiments/v1_2_12_f100_5seed_confirmation_remote_summary`
- Anchor 5-seed ensemble: `n=2594`, MAE `0.9277`, RMSE `1.2368`, R2 `0.5388`.
- Anchor 5-seed single-model mean: MAE `0.9883 +/- 0.0168`, RMSE
  `1.3091 +/- 0.0204`, R2 `0.4875 +/- 0.0158`.

Important interpretation boundary: the v1.2.12 improvement is not because the
test set changed. The fixed test predictions have the same `n=2594`. The
improvement mainly comes from seed averaging/ensemble reducing training
variance and extreme prediction noise.

## Decision Timeline

### v1.2.1 corrected rerun

Question: after medium audit, effect-level handling, and feature z-score
correction, is the previous best transfer result reliable?

Result: test MAE remained high at about `1.2311` with R2 `0.1700`, while
finetune validation was much stronger.

Decision: keep only as corrected baseline. The validation-test gap showed that
the next stage needed stronger source selection, scheduling, and evaluation
rather than more epochs alone.

### v1.2.2 source Tanimoto transfer optimization

Question: which first-round transfer mechanism improves the corrected baseline?

Result: `source_tanimoto alpha=1.0` improved test MAE to `1.1634` and R2 to
`0.2785`.

Decision: keep `source_tanimoto alpha=1.0` as the next anchor. CORAL did not
become the main strategy.

### v1.2.3 effect-level weighting

Question: can effect-level weighting stabilize endpoint-specific behavior?

Result: effects were mixed and not robust enough to become the mainline.

Decision: leave effect-level weighting default-off. Use it as a diagnostic, not
as a default training ingredient.

### v1.2.4 and v1.2.5 soil-only controls

Question: how strong is soil-only learning under low and fuller soil pTox data?

Result: soil-only was competitive when enough soil data were available. Low
soil sample settings still favored transfer or at least made transfer useful
for task coverage.

Decision: all transfer claims must be compared against soil-only on shared
tasks. Do not claim transfer wins from aggregate totals alone.

### v1.2.6 authority-bin HPO

Question: does authority-based toxicity binning help training?

Result: transfer f20 best was CE-bin loss weight `0.05` with MAE `1.1478` and
R2 `0.2978`; transfer f100 best was loss weight `0.025` with MAE `0.9768` and
R2 `0.4978`.

Decision: keep CE authority-bin as a transfer regularizer. It is not a
universal soil-only improvement.

### v1.2.7 ordinal, censored audit, proxy audit, AD audit

Question: should binning remain in training, and should ordinal loss replace CE?

Result: ordinal improved no-bin slightly, but did not beat CE authority-bin.
Censored records and proxy descriptors were large enough to justify controlled
follow-up. AD audit made species/task-family extrapolation visible.

Decision: keep CE authority-bin. Do not replace it with ordinal loss. Carry
chemical and species AD gates forward.

### v1.2.8 censored loss

Question: does censored loss help the CE-bin anchor?

Result: f100 with censored weight `0.01` improved to MAE `0.9692` and R2
`0.5072`. f20 improved most in MAE at censored weight `0.03`, but the gain was
modest.

Decision: keep censored loss for f100 at weight `0.01`. Treat stronger weights
carefully because they shift endpoint-family balance.

### v1.2.9 and v1.2.10 proxy source rules

Question: can environmental-behavior proxy weighting improve source selection?

Result: f100 proxy candidates had signal, but gains were not clean. Pure proxy
improved MAE in one view but not R2/family stability; Tanimoto+proxy improved
R2 in one view but hurt species extrapolation. Alpha `0.25` and `0.75` did not
resolve the tradeoff.

Decision: stop proxy grid expansion. Keep `tanimoto_proxy alpha=0.5` only as a
secondary endpoint/in-domain diagnostic.

### v1.2.11 seed stability

Question: was the strong seed42 result stable?

Result: no. The 3-seed single-model mean showed seed42 was optimistic. Proxy
mean gains were tiny and not scientifically decisive.

Decision: do not choose the final model from one seed.

### v1.2.12 5-seed confirmation

Question: does 5-seed confirmation stabilize the f100 story?

Result: the anchor 5-seed ensemble was the best overall result and was more
stable across AD gates than proxy alternatives.

Decision: set v1.2.12 anchor 5-seed ensemble as the current mainline. Report
single-model mean and ensemble separately.

### v1.2.13 validation policy

Question: does releasing the 20 percent finetune-validation rows into training
improve final locked performance?

Result: `finetune_validation_fraction=0.0` nearly matched control on R2/RMSE
but was slightly worse on MAE and Huber loss. Single-model mean also did not
beat the `0.2` control.

Decision: keep `finetune_validation_fraction=0.2`. Treat val0 as a sensitivity
check only.

## Paper-Level Story So Far

The result is stronger than a pure performance story. The useful scientific
story is:

- Target-scale control matters. Water-to-soil transfer is not interpretable if
  pTox, mg/kg, and other scales are mixed.
- Zero-shot water-to-soil transfer is weak; soil finetuning is necessary.
- Source Tanimoto weighting gives the first robust transfer improvement.
- Authority-bin auxiliary learning adds risk-level prior information and acts
  as a migration regularizer.
- Censored loss improves the f100 mainline when weighted modestly.
- Species/task-family extrapolation is a major risk layer and should be
  reported separately from chemical AD.
- Seed ensembles improve robustness, but the single-model mean must remain in
  the report to avoid overstating deployable single-model performance.

## How To Add A New Experiment

Before running:

1. Add a proposed row to `docs/experiment_registry.csv` with `status=planned`.
2. Record the fixed data version, source table, split, config, script, seed set,
   and expected output directory.
3. State the decision rule in one sentence, for example: "continue only if MAE
   improves by at least 0.02 and AD species-extrapolation does not worsen."

After running:

1. Update the registry row to `completed`.
2. Add a short decision note in this file.
3. Link the exact summary directory and key CSV files.
4. If the script or source table changed, update
   `docs/code_organization_and_experiment_scripts.md`.
