# Experiment Decision Log

This file is the human-readable decision layer for the v1.2 transfer-learning
experiments. Use it with `docs/experiment_registry.csv`: the registry stores
where each result lives, while this log records why each strategy was kept or
stopped.

## Fixed-boundary traditional-ML comparison (2026-07-24)

- v1.2.53 compares RF, XGBoost, LightGBM, PLS, and KNN with the current deep
  routes on the locked v1.2.44 random-interpolation boundary. Traditional
  models receive no species/task embedding and do not pool records across
  species: every model is fitted independently for one
  `latin_name x model_head` species-endpoint subtask.
- A support contract of at least 35/10/10 train/validation/test records was
  locked before fitting. Forty-one subtasks pass, yielding the same 1,135
  outer-test rows for every model and feature contract (37.31% of the complete
  3,042-row test set). Unsupported subtasks are audited rather than silently
  removed.
- Layer 1 asks about the whole-framework advantage: molecule-only traditional
  models are compared with M11U. The best traditional baseline is XGBoost
  (R2=0.6039, MAE=0.5895), versus M11U R2=0.7358 and MAE=0.4935.
  The paired deltas are +0.1319 R2 (95% CI 0.1001 to 0.1655) and -0.0960 MAE
  (95% CI -0.1201 to -0.0726).
- Layer 2 asks about architecture/training-system advantage under matched
  molecule-plus-context inputs. RF has the best traditional R2 (0.6407) and
  XGBoost the best traditional MAE (0.5517), versus M00 R2=0.7107 and
  MAE=0.5226. M00 improves R2 over RF by 0.0700 (95% CI 0.0386 to 0.1023) and
  MAE over XGBoost by 0.0291 in the beneficial direction (95% CI 0.0043 to
  0.0547).
- Inference uses 20,000 record-paired bootstrap resamples stratified within
  species-endpoint subtasks, followed by Holm adjustment across the five
  traditional baselines. Reference and test identifiers are provenance only;
  no reference/test aggregation enters model fitting, splitting, tuning, or
  uncertainty estimation. Both metrics favor the deep route significantly
  against all five baselines in both layers after correction.
- The single four-panel publication figure combines absolute R2/MAE bars with
  R2-gain and MAE-reduction heatmaps. Stars appear only for favorable,
  Holm-adjusted significant cells; nonsignificant labels are omitted.
- These results support random-interpolation claims only for the 41 adequately
  supported species-endpoint subtasks. They do not establish generalization to
  unseen species, unseen endpoints, or unseen chemical structure families.

## Scaffold/similarity-family chemical-extrapolation follow-up (2026-07-21)

- v1.2.47 is a separate boundary test for unseen chemical structure families.
  It reuses the locked v1.2.44 Stage-3 parent pool but does not reuse the
  random or reference-group target assignment. References, species, and tasks
  may cross partitions; chemical structure components may not.
- Structures are normalized to the largest carbon-containing uncharged parent,
  then linked by exact canonical identity, shared non-empty Murcko scaffold, or
  any direct Morgan radius-2/2048-bit Tanimoto >=0.65 edge. After excluding
  invalid/non-organic structures, the target pool contains 11,267 rows, 581
  canonical chemicals, and 316 transitive structure components.
- The locked Stage-3 boundary contains 7,322/1,651/2,294
  train/validation/test rows and 189/61/66 components. All canonical,
  scaffold, component, aggregate, result, and test-identifier overlaps are
  zero; the largest cross-part Tanimoto is 0.6364.
- None of 4,096 raw prediction-blind hash candidates met every hard task gate
  because ICx_Growth has only 51 structure-resolved rows. Before any model was
  trained, the best 64 candidates were subjected to deterministic whole-
  component constraint repair using only support deficits, ratio and identity
  gates, and target-distribution balance. Fifty-four became feasible. The
  selected seed is 20261729, and its four component moves are recorded in the
  immutable split contract; no threshold was relaxed and no component split.
- Stage-1/2 source rows are independently removed when their canonical parent
  or non-empty scaffold overlaps target validation/test, or their Morgan
  similarity to either target holdout is >=0.65. The retained source has
  130,045 aquatic rows and 6,455 soil-pTox rows. Its maximum similarity to the
  held-out target structures is 0.6471; exact aggregate/result/test and
  canonical/scaffold overlaps are zero.
- The contract hash is
  `5acc398a4bc9215fa4972511156106e1f0461f9ff1b7a6ef13cd25fd884cc755`.
  Local and remote protocol/regression tests passed (20 tests), and all M00,
  M10, and M11U one-epoch smoke cells passed at 2026-07-21 10:40:41 (+08:00).
- The formal M00/M10/M11U x four-seed matrix started at 10:40:50 (+08:00)
  with two concurrent jobs. Stage 3 is full-network finetuning with explicit
  validation; test is report-only. On completion, 20,000 paired bootstrap
  resamples will use target-test structure components, with a +/-0.01 MAE
  practical-equivalence margin for M10 versus M11U. The two predeclared
  low-support tasks report task MAE/RMSE but not task-level R2.

## Reference-group boundary follow-up (2026-07-20)

- The paired v1.2.45 reference-component bootstrap is complete on the same
  3,042 v1.2.44 test rows, using 917 transitive reference components and 20,000
  resamples. M10 improved MAE over strict M00 by 0.03345 (95% CI 0.02427 to
  0.04289 in the beneficial direction).
- M11U minus M10 had delta MAE=-0.00111 with 95% CI [-0.00717, 0.00496]. The
  full interval lies inside the predeclared +/-0.01 practical-equivalence
  margin. Therefore the default scientific interpretation is that soil-pTox
  intermediate adaptation adds no practically meaningful performance after
  aquatic pretraining on the locked random boundary.
- v1.2.46 is the single boundary validation selected after v1.2.44. All records
  connected through shared reference identifiers are assigned to only one of
  Stage-3 train, validation, or test. The locked counts are 9,809/2,516/2,874
  records and 854/217/291 reference components across 18 tasks. Source-domain
  Stage-1/2 records sharing a reference with target validation or test were
  removed; no CAS-, SMILES-, scaffold-, species-, or task-disjoint claim is
  introduced.
- The v1.2.46 matrix contains only M00, M10, and M11U with seeds 42, 2042,
  3407, and 8417. Stage 3 uses full-network finetuning (`freeze=none`) with an
  explicit validation split. The test partition remains report-only and is not
  used for stopping, route selection, or formal-run monitoring.
- Local and remote protocol tests both reported 34 passed. All three 1-epoch
  smoke cells passed at 2026-07-20 19:39 (+08:00), after which the formal
  12-cell matrix started automatically. All 12 cells completed at 22:30:34
  (+08:00), with 12 manifests and 12 prediction exports. The controller exited
  normally; the log is
  `outputs/logs/v1_2_46_reference_group_gate_20260720_185905.log`.
- On the 2,874-row reference-group test ensemble, M00/M10/M11U obtained R2
  0.1710/0.2302/0.2243, RMSE 1.2092/1.1652/1.1697, and MAE
  0.8986/0.8649/0.8668. M10 improved MAE over M00 by 0.0337 and was better in
  14 of 18 task heads. M11U was slightly worse than M10 overall (delta R2
  -0.0059; delta MAE +0.0019) and improved only 8 of 18 task-head MAEs.
- The boundary result therefore preserves the qualitative causal conclusion:
  aquatic pretraining supplies useful transferable initialization, whereas
  soil-pTox intermediate adaptation does not add measurable value after that
  pretraining. The sharp absolute-performance decline relative to v1.2.44 also
  identifies study-source and experimental-condition heterogeneity as an
  important generalization limitation. Because the random and reference-group
  test identities differ, their numerical gap is descriptive rather than a
  paired causal estimate.
- This experiment estimates robustness to new study/reference sources and
  associated experimental-condition heterogeneity. It is not evidence of new
  chemical-scaffold or new-species extrapolation and will be reported
  separately from the v1.2.44 random-interpolation result.

## Current soil mol/kg three-stage mainline (2026-07-20)

- The active target reference is `v1.2.40 X0_molar`, not the E- or G-series
  explorations. The latter are excluded from v1.2.44 splits, checkpoints,
  thresholds, model initialization, and final aggregation.
- The v1.2.40 Stage-3 trunk was not frozen: its effective setting was
  `finetune_mgkg_freeze=none` with one unified learning rate. Therefore the
  new `M11U` cell reproduces the current v1.2.40 strategy, while `M11F` is the
  newly added true heads-only frozen-trunk control.
- The causal matrix is locked to the same 18-task Stage-3 boundary for every
  cell: 9,724 target-train, 2,433 validation, and 3,042 test records. M00, M10,
  M01, and M11 have the identical Stage-3 boundary hash.
- M10 and M01 use the task set already admitted by the v1.2.40 full route.
  Their internal task thresholds are disabled only after that admission step,
  preventing removal of a pretraining stage from also changing the task set.
  M10 reproduces the exact M11 Stage-1 identities; M01 reproduces the exact
  M11 Stage-2 identities.
- The historical v1.2.40 `D_molar` run was rejected as strict M00 evidence:
  it merged 12,157 Stage-3 development rows and reconstructed validation with
  seed 17073, instead of the locked 9,724/2,433 boundary with seed 42. The
  training engine was corrected to preserve explicit `epochs=0` and to permit
  an empty Stage 1 only for a requested downstream-only route. Strict M00 was
  then retrained on the fixed boundary.
- All 28 formal cells (seven trained cells times four seeds) completed on
  2026-07-20 at 16:41 (+08:00). There are 28 validated manifests and prediction
  exports, and both local and remote targeted suites report 28 passed tests.
- B1 uses the newly reproduced M11U as the within-matrix causal reference and
  separately reports agreement with the completed v1.2.40 X0_molar runs. B2
  reports both ordinary R2 and within-task centered R2. B3 compares molar and
  mass training only after conversion to the same reporting scale and includes
  error-versus-logMW diagnostics.
- On the four-seed rowwise test ensemble, M11U reached R2=0.7153 and MAE=0.5447.
  M10 was effectively tied (R2=0.7154, MAE=0.5458), so soil-pTox adaptation has
  little overall independent gain. M01 (R2=0.6701, MAE=0.5879) and strict M00
  (R2=0.6801, MAE=0.5792) show that aquatic pretraining is the main transfer
  source. M11F strongly degraded (R2=0.3521, MAE=0.8830); full Stage-3
  finetuning is retained.
- Full-model within-task R2=0.6817, versus 0.2997 for Context-only and 0.4924
  for Molecule-only, supporting genuine within-task molecular signal. On the
  common mg/kg scale, Scale-Mol modestly outperformed Scale-Mass
  (R2 0.6855 vs 0.6794; MAE 0.5441 vs 0.5498), but the row-level association
  of its gain with logMW was weak (r=0.0424).

## Current Mainline

The current reporting boundary is split-dependent. Do not answer the current
best version as a single number without naming the evaluation boundary first.

Primary random-interpolation reporting:

- Current version: `v1.2.22`.
- Data scope: no-metal/no-inorganic soil pTox transfer data from
  `outputs/derived/modeling_dataset_v2_0_0_rebuild_no_metal_inorganic.sqlite`.
- Split scope: no-metal random 8:2 and no-metal random 5-fold adaptation
  splits.
- Main evidence:
  `outputs/experiments/v1_2_22_no_metal_random_split_remote_summary`.
- 5-seed ensemble random 8:2: `n=2608`, MAE `0.6243`, RMSE `0.8865`,
  R2 `0.7895`.
- 5-seed ensemble random 5-fold: `n=13063`, MAE `0.6200`, RMSE `0.8899`,
  R2 `0.7920`.
- Interpretation: this is the strongest current interpolation/reporting
  result after removing metal/metalloid and inorganic chemicals. It should not
  be described as new-structure extrapolation.

Chemical-family extrapolation pressure test:

- Current version: `v1.2.24`.
- Data scope: same no-metal/no-inorganic data boundary as v1.2.22.
- Split scope: scaffold/similarity-cluster holdout and scaffold/similarity-
  cluster 5-fold.
- Main evidence:
  `outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_remote_summary`.
- 5-seed ensemble scaffold/similarity-cluster holdout: `n=2493`, MAE `1.0664`,
  RMSE `1.4063`, R2 `0.1810`.
- 5-seed ensemble scaffold/similarity-cluster 5-fold: `n=11459`, MAE `1.1178`,
  RMSE `1.4808`, R2 `0.3664`.
- Interpretation: this is the current structure-family extrapolation boundary.
  The performance drop relative to random splits is expected and should be
  reported as extrapolation difficulty, not as a failed random-split model.

Historical fixed-CAS holdout anchor:

- Historical version: `v1.2.12` / `v1.2.17`.
- Evidence:
  `outputs/experiments/v1_2_12_f100_5seed_confirmation_remote_summary` and
  `outputs/experiments/final_mainline_comparison`.
- Anchor 5-seed ensemble: `n=2594`, MAE `0.9277`, RMSE `1.2368`, R2 `0.5388`.
- Interpretation: keep this as a historical fixed-CAS comparison and seed-
  ensemble anchor. It is no longer the preferred single current mainline
  because the CAS-number holdout is weaker as a scientific structure-
  extrapolation design than v1.2.24.

Mechanism and diagnostic branches:

- `v1.2.31` supports that random performance is not mainly driven by molecular
  weight or molecular-size descriptor coupling.
- `v1.2.33` is the current RDKit descriptor/fingerprint molecular-signal
  diagnostic.
- `v1.2.34-v1.2.36` are PaDEL/graph molecular-input diagnostics. They show
  measurable molecular signal but do not replace the full RDKit/Morgan
  mainline.

The shared modeling strategy retained across the current transfer mainline is:

- Source weighting: `tanimoto_to_finetune`, `alpha=1.0`.
- Auxiliary target: authority-based toxicity bin CE, loss weight `0.025`.
- Censored loss: enabled, loss weight `0.01`.
- Finetune validation: keep `finetune_validation_fraction=0.2`.
- Robustness: report both 5-seed single-model mean and 5-seed ensemble.

## Decision Timeline

### v1.2.31 molecular-size descriptor sensitivity

Question: does the current pTox model performance depend materially on
molecular weight or strongly molecular-size-related descriptors, given that
some `pTox` targets are converted from `mg/L` to `mol/L` using molecular
weight?

Design: add a deep-model ablation named `no_molecular_size_descriptors`. It
masks `MolWt`, `TPSA`, `HeavyAtomCount`, `NumHAcceptors`, `NumHDonors`,
`RingCount`, and `RotatableBonds`, while leaving Morgan fingerprints,
`MolLogP`, species/context fields, medium adapters, source weighting,
toxicity-bin auxiliary loss, and censored loss unchanged. Also add a traditional
ML descriptor-sensitivity switch,
`--descriptor-sensitivity drop_molecular_size_related`, which removes a broader
set of RDKit molecular-size, surface-area, VSA, Chi, BCUT, ring/flexibility,
and atom-count proxy descriptors.

Execution: `scripts/run_v1_2_31_molecular_size_sensitivity_remote.sh priority`
was launched on 2026-07-07 01:35 (+08:00) and completed on 2026-07-07 03:15
(+08:00). The priority matrix compared `full` vs
`no_molecular_size_descriptors` at seed `2042` on no-metal random 8:2 and
scaffold/similarity-cluster holdout. All 4/4 formal runs completed with
exit_code `0`. The summary lives in
`outputs/experiments/v1_2_31_molecular_size_sensitivity_remote_summary`, with
interpretive notes in `docs/v1_2_31_molecular_size_sensitivity_summary.md`.

Result: in the random 8:2 interpolation split, masking molecular-size
descriptors slightly improved test performance (MAE `0.6724 -> 0.6656`, R2
`0.7635 -> 0.7658`). In the scaffold/similarity-cluster holdout, masking these
descriptors caused a small loss (MAE `1.1114 -> 1.1308`, R2 `0.1204 ->
0.0988`).

Decision: keep molecular-size descriptors in the main model. The priority
result does not support the concern that model skill is mainly driven by direct
molecular-weight/unit-conversion coupling. However, `MolWt` and related
descriptor importance must be interpreted cautiously as molecular size,
surface/polarity, and structural-complexity signal that is partly coupled to
the pTox scale. Do not present molecular weight as an independent mechanistic
proof of toxicity. Full 5-fold expansion is optional and only needed if this
sensitivity analysis becomes a manuscript-level quantitative claim.

### v1.2.27 traditional ML descriptor-effect baselines

Question: how strong are local traditional machine-learning baselines when the
model is intentionally restricted to molecular descriptors plus effect level,
without species/context fields?

Design: use `aggregated_task_records_aquatic_ptox_qc` and
`aggregated_task_records_soil_ptox_qc`. Features are RDKit 2D descriptors and
`effect_level_x` derivatives only. Species identity, taxonomy, medium labels,
task labels, and target/concentration fields are excluded. The local validation
split is row-random 8:2 with seed `42`; Optuna TPE tunes LightGBM and
ExtraTrees for domain, endpoint, species, and species-endpoint subtasks.

Result: the run completed locally under
`outputs/experiments/v1_2_27_traditional_ml_descriptor_effect_baselines` and
was copied to
`实验汇总/12_传统机器学习基线_分子描述符效应水平`. Best-by-subtask validation
metrics show aquatic overall R2 `0.5542` / MAE `0.9601`, soil overall R2
`0.6098` / MAE `0.8091`, aquatic `ECx_Immobilization` R2 `0.8341`, and soil
`ECx_Growth` R2 `0.7151`.

Decision: keep v1.2.27 as a descriptor-only interpolation baseline for
paper/report comparison and subtask visualization. Do not use it as evidence
for chemical-family extrapolation or cross-species generalization, because the
split is row-random and species-level models are trained after filtering to a
specific species rather than predicting unseen species.

### v1.2.26 no-metal/inorganic random-split targeted ablation

Question: after removing inorganic and metal/metalloid chemicals, do the
random-split ablation conclusions still rank the same modules and training
strategies as v1.2.21?

Design: reuse the v1.2.21 targeted ablation set on the v1.2.22 no-metal random
transfer splits. The ablations are `no_context`, `no_species_lifestage`,
`no_molecular_residual`, `no_source_weighting`, `no_toxicity_binning`, and
`no_censored_loss`. To control cost, run only seed `2042`, selected from
v1.2.22 because it has the best MAE on both no-metal random 8:2 and
fold-weighted no-metal random 5-fold single-model summaries.

Result: the remote queue completed 36/36 formal runs with exit_code `0` and
summarized on 2026-07-03 22:36:54 (+08:00). Against the v1.2.22 seed2042 full
baseline, `no_context` caused the largest MAE increase (`+0.2131` random 8:2;
`+0.2366` random 5-fold), followed by `no_species_lifestage` (`+0.1040`;
`+0.0952`). `no_toxicity_binning` had a moderate loss (`+0.0315`; `+0.0391`),
while `no_source_weighting`, `no_censored_loss`, and `no_molecular_residual`
stayed close to the full baseline.

Decision: the no-metal/inorganic sensitivity check preserves the v1.2.21
random-split ablation ordering. Context and species/lifestage remain the main
contributors under random interpolation after removing metals and inorganics.
Treat source weighting, censored loss, and molecular residual as weak or
setting-dependent contributors in this random-split setting. This result is an
organic-chemical descriptor-applicability sensitivity check and should not
replace v1.2.24 scaffold/cluster chemical-family extrapolation evidence.

### v1.2.24 scaffold/cluster chemical-family holdout

Question: does the current no-metal mainline remain robust when chemical
generalization is defined by structure families rather than CAS identifiers?

Design: build new split assignments on
`modeling_dataset_v2_0_0_rebuild_no_metal_inorganic.sqlite` using canonical
organic parent SMILES, Bemis-Murcko scaffolds, and Butina/Morgan fingerprint
clusters at Tanimoto `0.65`. Records without an organic parent are excluded
from this structure-extrapolation split and reported in
`missing_structure_report.csv`. The training strategy is otherwise unchanged:
`tanimoto_to_finetune alpha=1.0`, authority CE-bin loss weight `0.025`,
censored loss weight `0.01`, and `finetune_validation_fraction=0.2`.

Result: the formal 30-run `ensemble_all` completed and was synchronized to
`outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_remote_summary`.
The 5-seed ensemble metrics show a substantial drop relative to random
validation: scaffold/similarity-cluster holdout reached `n=2493`,
R² `0.1810`, MAE `1.0664`, and Huber loss `0.6746`; scaffold/similarity-cluster
cross-validation reached `n=11459`, R² `0.3664`, MAE `1.1178`, and Huber loss
`0.7224`.

Decision rule: interpret v1.2.24 as the current chemical-family extrapolation
pressure test. It should be compared against v1.2.22 no-metal random split and
the older CAS-holdout result, but it should not be expected to match random
interpolation performance. If performance drops materially while leakage
audits remain clean, the drop should be reported as the cost of structural
extrapolation rather than as a training failure.

### v1.2.22 no-metal/inorganic random-split sensitivity

Question: how much of the random 8:2 and random 5-fold performance depends on
records whose chemicals are classified as inorganic or metal/metalloid?

Design: keep the existing v1.2.15/v1.2.16 random-split result set intact, build
a separate derived SQLite subset
`modeling_dataset_v2_0_0_rebuild_no_metal_inorganic.sqlite`, and rerun the same
best random-transfer strategy on filtered `_no_metal_inorganic` tables.

Decision rule: compare the no-metal 5-seed random 8:2 and random 5-fold
ensemble metrics against the existing full-data v1.2.17 random rows. If
performance changes materially, report the sensitivity explicitly and avoid
claiming that the all-data QSAR behavior is purely organic-chemical structure
learning.

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

### v1.2.23 random mainline CST-AD

Question: can the current random 8:2 and random 5-fold mainline predictions be
described by Chemical-Species-Task AD without using seed-ensemble uncertainty
as an extra tier?

Result: yes. The analysis directly used the already pulled v1.2.15 remote
5-seed ensemble prediction rows and did not retrain. Under the C+S+T variant,
random 8:2 had AD-A coverage 96.49 percent with MAE 0.5896, while AD-B had MAE
0.6400. Random 5-fold had AD-A coverage 96.26 percent with MAE 0.5904, while
AD-B had MAE 0.6832. High-error detection remained weak, with AUROC near 0.50.

Decision: use random 8:2 and random 5-fold as the current mainline basis for
interpolation reliability reporting. Do not use CAS-number chemical holdout as
the current sensitivity or pressure test, because CAS grouping is not a
scientific scaffold/structure-extrapolation design. If a new-chemical
extrapolation claim is needed, create a separate scaffold, cluster, or
Tanimoto-distance split.

### v1.2.28 species-endpoint traditional ML baseline

Question: within each fixed species and toxicity endpoint, how strong are
traditional ML baselines when species/context labels are excluded and inputs
are restricted to a literature-guided RDKit descriptor union plus effect level?

Result: 60 species-endpoint subtasks were run locally, 30 aquatic and 30 soil.
XGBoost, LightGBM, Random Forest, KNN, and PLS were tuned with Optuna TPE
(`n_trials=3`). A total of 299/300 model fits completed and 598 PNG/SVG
scatter plots were exported. Aquatic tasks favored LightGBM by weighted
validation RMSE (0.8374), closely followed by XGBoost (0.8465) and Random
Forest (0.8684). Soil tasks favored XGBoost by weighted validation RMSE
(0.9084), followed by Random Forest (0.9416) and KNN (0.9813). PLS was
generally weaker and one `Oryza sativa / ECx_GeneticDamage` PLS fit was
skipped because sklearn PLS internally produced NaN loadings despite finite
input features and finite targets.

Decision: use v1.2.28 as the traditional ML species-endpoint interpolation
baseline for per-species/per-endpoint figures and model comparisons. Do not
interpret it as cross-species, cross-endpoint, or chemical-family extrapolation
evidence. Keep the skipped PLS audit instead of substituting another linear
model under the PLS label.

### v1.2.29 expanded soil species-endpoint baseline

Question: was the v1.2.28 soil top-30 species-endpoint matrix too narrow for
the user's requested soil species coverage?

Result: yes. v1.2.28 soil covered 30 combinations and 12 species. Querying the
soil table showed 73 combinations and 24 species at `n >= 30`, so an expanded
soil-only matrix was run with the same descriptor/effect feature policy and
the same five model families. The expanded run completed 364/365 model fits,
exported 728 PNG/SVG plots, and produced 73 descriptor tables. XGBoost had the
best weighted validation RMSE (1.0195), followed by Random Forest (1.0334) and
KNN (1.0567). One PLS fit for `Oryza sativa / ECx_GeneticDamage` remained a
skipped numerical-degeneracy audit.

Decision: use v1.2.29 for broader soil plotting and exploratory species
coverage. Treat it as lower-confidence than high-n subtasks because the
threshold was reduced to `n >= 30`; do not use small validation-set results as
strong model-ranking evidence.

### v1.2.30 expanded aquatic species-endpoint baseline

Question: was the v1.2.28 aquatic top-30 species-endpoint matrix also too
narrow for the user's requested aquatic species coverage?

Result: yes. v1.2.28 aquatic covered 30 combinations and 16 species, while the
database contains 199 aquatic species-endpoint combinations and 97 species at
`n >= 200`. The expanded aquatic run completed 995/995 model fits, exported
1,990 PNG/SVG plots, and produced 199 descriptor tables. LightGBM had the best
weighted validation RMSE (0.8778), closely followed by XGBoost (0.8798);
XGBoost won the most individual subtasks (91), followed by LightGBM (64).

Decision: use v1.2.30 as the main aquatic traditional ML figure/comparison set
for species-endpoint baselines. Keep `n >= 200` as the current practical
coverage threshold because it expands coverage substantially while preserving
more stable validation sets than `n >= 100`, `50`, or `30`.

### v1.2.32 molecular signal strength evidence boundary and side branch

Question: does the current model rely so strongly on species/context embeddings
that molecular descriptors are masked, and can PaDEL/graph molecular inputs be
prepared without disturbing the current mainline?

Result: the current evidence package was built under
`实验汇总/分子信号强度探索_20260707`. The user explicitly excluded the
fixed chemical holdout / CAS-number `v1.2.18` project from future consideration,
so `v1.2.18` is recorded only as `excluded_historical` in
`source_manifest.csv`. Under the current boundary, `no_context` and
`no_species_lifestage` remain the strongest random-interpolation ablations,
while `no_molecular_size_descriptors` does not materially reduce random
performance and only mildly worsens scaffold-cluster 8:2 performance. After
excluding `v1.2.18`, valid current-boundary evidence is missing for
`no_descriptors`, `no_fingerprint`, and `descriptors_only`.

Decision: do not use `v1.2.18` to fill current molecular-signal ablation gaps.
Before drawing a conclusion about molecular descriptors versus fingerprints,
run seed2042 `no_descriptors`, `no_fingerprint`, and `descriptors_only` on
no-metal `random_8_2` and `scaffold_cluster_8_2`. Keep best-performance
modeling eligible for ensembles, but keep mechanism/diagnostic ablations at
seed2042 unless a clear effect threshold justifies expansion. The PaDEL and
graph work remains a side branch: PaDEL caches and descriptor heads are
implemented, graph cache/interface is reserved, and no default mainline
behavior changes until those branches show value.

### v1.2.33-v1.2.36 molecular signal follow-up queue

Question: can the missing molecular-signal contrasts be started while PaDEL and
graph inputs are completed as side branches?

Status: v1.2.33 is running remotely for seed2042 no-metal `random8_2` and
`scaffold_cluster_8_2` with `full`, `no_descriptors`, `no_fingerprint`, and
`descriptors_only`. As of 2026-07-07 15:44 (+08:00), all eight priority runs
completed successfully and the v1.2.33 summary directory was generated. The
graph-only follow-up queue then started v1.2.36 at 2026-07-07 15:47 (+08:00),
beginning with `random8_2_graph_only_molecule_seed2042_graph_only_molecule`.
As of 2026-07-07 16:44 (+08:00), both graph-only priority runs completed with
exit code 0. PaDEL raw was then started; the first attempt exposed two
PaDEL-specific integration issues before training metrics were produced:
cache-miss descriptor width fell back to the 8-dimensional RDKit/fallback
schema, and extreme PaDEL descriptor values could overflow preprocessing
statistics. Both were fixed by schema-aligned cache-miss handling and bounded
raw numeric preprocessing. The fixed v1.2.34 PaDEL raw queue restarted at
2026-07-07 16:57 (+08:00), completed both valid priority runs, and then
v1.2.35 PaDEL prior clustered completed both priority runs by
2026-07-07 19:54 (+08:00).
PaDEL is prepared as a 2D
descriptor cache path
because the initial `padelpy.from_smiles` route invoked 3D conversion and was
too slow for this diagnostic branch. The full PaDEL cache is complete and synced
to remote (`6353/6353` rows, `1444` descriptors, zero descriptor-generation and
Morgan-fingerprint failures). Graph-only is no longer just reserved: a pure
PyTorch message-passing graph encoder is connected through
`graph_only_molecule`, the graph cache covers 6353/6353 unique SMILES, and a
remote queue waited until v1.2.33 finished before launching. A second remote
queue completed v1.2.34 PaDEL raw and v1.2.35 PaDEL prior clustered.

Decision: keep v1.2.33 as the immediate RDKit/fingerprint ablation evidence.
PaDEL raw is not a mainline replacement: random8_2 test MAE/R2 was
0.7270/0.7157 and scaffold_cluster_8_2 was 1.2350/-0.0731. PaDEL prior
clustered recovered random8_2 performance close to RDKit full
(MAE/R2 0.6670/0.7646), but scaffold_cluster_8_2 remained weaker
(1.1464/0.0726). Graph-only retained signal without descriptor or Morgan input
(random8_2 0.7987/0.6717; scaffold_cluster_8_2 1.1254/0.0779), but did not
match full. Interpret the model as context-aware/species-informed QSAR rather
than descriptor-only QSAR: species/context signals are strong, but molecular
signals remain measurable and split-dependent.

### Remote result preservation before server expiry

Question: before the remote server expires, are completed valuable results locally preserved and logically organized under `实验汇总`?

Status: completed on 2026-07-12. A fresh remote/local comparison found three missing formal lightweight result directories: `runtime_summaries`, `v1_2_7_censored_ordinal_ad_first_batch_remote_summary`, and `v1_2_15_random_split_policy_remote_summary`. These were pulled to `outputs/experiments`. Key completion logs and runtime files were also pulled or refreshed, including the full v1.2.24 scaffold-cluster log, v1.2.31/v1.2.33 molecular-signal logs, and v1.2.34-v1.2.36 queue nohup logs.

Decision: package evidence by interpretation boundary, not by remote folder order. v1.2.24 complete scaffold-cluster summary was added under `实验汇总/13_结构骨架聚类外推审计_v1_2_24/远端训练结果_summary`; v1.2.31-v1.2.36 molecular-signal summaries were added under `实验汇总/分子信号强度探索_20260707/远端摘要明细`; v1.2.7/v1.2.15 were added under `实验汇总/18_历史策略与方法筛选_正式摘要_v1_2_7_v1_2_15`; runtime evidence was added under `实验汇总/10_运行时间记录`.

Interpretation: the current mainline boundary is unchanged. v1.2.22 remains the no-metal random-interpolation reporting boundary, v1.2.24 is the scaffold/similarity-cluster extrapolation boundary, v1.2.31-v1.2.36 are molecular-signal diagnostics, and v1.2.7/v1.2.15 are historical method-screening evidence. No remote deletion or cleanup was performed.

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

### v1.2.38 soil mg/kg single-domain random split local补跑

Question: can the current deep CST framework directly model the traditional
soil `mg/kg` target scale when the source table is switched from soil pTox to
`aggregated_task_records_soil_mg_kg_qc`?

Planned design: run locally because the previous remote server has expired.
The source table is `aggregated_task_records_soil_mg_kg_qc`, the target is
`neg_log10_mg_kg`, and the split names are isolated under `SoilMgkgQC2_`.
The initial补跑 uses seed 42, random 8:2 plus random 5-fold, and the existing
single-domain deep training settings: full model, per-task-target
standardization, authority-bin auxiliary classification loss, censored hinge
loss weight 0.01, no source weighting, and no finetuning stage. The local
launcher is `scripts/run_v1_2_38_soil_mgkg_random_local.ps1` and defaults to
`.venv-cuda\Scripts\python.exe`. It uses
`configs\experiment.local.cuda3050ti.yaml` so Windows training runs with
single-process DataLoader workers.

Decision rule: treat this as an exploratory target-scale feasibility run. If
random 8:2 and 5-fold results are both usable and task-level behavior is not
dominated by a few high-count species or endpoints, expand to multi-seed and
then scaffold/similarity-cluster splits before making soil-risk-limit claims.

Status: completed locally on 2026-07-18. The first smoke attempt failed because
the remote config used `num_workers=4`, which triggered a Windows
multiprocessing `PermissionError`; switching the launcher to
`configs\experiment.local.cuda3050ti.yaml` fixed the issue by using
single-process workers. Smoke then passed, and the formal seed42 random 8:2
plus five random 5-fold runs all completed with exit code 0.

Result: for all test prediction rows, random 8:2 produced `n=3054`, `R2=0.6582`,
`RMSE=0.7888`, and `MAE=0.5704` on the `neg_log10_mg_kg` scale. The combined
five test folds produced `n=15236`, `R2=0.6526`, `RMSE=0.7859`, and
`MAE=0.5637`. The summary focus rows, which follow the standard task-valid
summary filter, gave random 8:2 `n=2967`, `R2=0.6370`, `RMSE=0.7974`,
`MAE=0.5787`; individual 5-fold focus R2 values ranged from `0.6211` to
`0.6487`.

Decision: the current deep CST framework can directly train the soil `mg/kg`
target scale as a separate soil-only model without pTox-to-mg/kg matching. This
is positive random-split feasibility evidence, not yet external-generalization
or risk-limit evidence. The next defensible expansion is multi-seed stability
and scaffold/similarity-cluster validation on `aggregated_task_records_soil_mg_kg_qc`.

### v1.2.40 paired mass/molar result and v1.2.41 stage-3 optimization

Evaluation boundary: for the present model, the primary empirical evaluation
is the locked random 8:2 split because the prediction function combines a broad
chemical space with species, life-stage, endpoint, duration, and medium
context. Scaffold or similarity-cluster holdout remains a separate optional
question about unseen structural families; it is not a gate for interpreting
the primary heterogeneous chemical-species model.

Scale interpretation: the v1.2.40 three-stage mol/kg branch reached a four-seed
mean native test R2 of 0.6973. This is a valid native mol/kg result, not an
insufficient result. The paired conversion adds the sample-specific term
`3 + log10(MW)`, so back-conversion leaves prediction residuals, MAE, and RMSE
unchanged but changes the target variance and therefore R2. Native mol/kg and
back-converted mg/kg R2 must consequently be reported side by side rather than
treated as the same metric.

Seed decision: v1.2.22 used `42, 1042, 2042, 3042, 4042`, but its first two
stages differ materially from the current pipeline. The current stage 2 uses
20 epochs, LR 1e-4, and cosine scheduling, whereas v1.2.22 used 60 epochs,
approximately 3.08e-4, and ReduceLROnPlateau; the current split also includes
the exact-source routing correction and new target-head/data-table contract.
Therefore v1.2.41 retains the completed v1.2.40 seeds
`42, 2042, 3407, 8417`. Fixed validation ranks `3407` and `42` highest, so they
are locked as screening seeds without consulting challenger test metrics.

Matrix: S1 keeps stage 1/2 at 30/20 epochs and changes only stage 3 to 40
epochs with full unfreezing, head LR 5e-4, trunk LR 1e-4, and a new cosine
cycle. S2 adds stage-3-only SWA across epochs 31-40. S3 adds a standardized
target loss of 0.7 Huber plus 0.3 MSE. Stage-3 early stopping is disabled so
all candidates receive the same 40-epoch budget. A challenger is eligible only
if its two-seed mean validation native R2 increases and validation native MAE
decreases relative to S0. The eligible model with the lowest validation MAE is
locked, then expanded to seeds 2042 and 8417; test metrics are report-only.

### v1.2.41 no-winner result and v1.2.42 E-series OOF decision

The v1.2.41 full-unfreeze screen completed without an eligible winner. On the
locked two-seed validation identities, S0 achieved native mol/kg R2 0.6907 and
MAE 0.5536. S1, S2, and S3 achieved R2 0.6171, 0.6164, and 0.6175 and MAE
0.6339, 0.6346, and 0.6381, respectively. Since every challenger degraded both
metrics, no expansion run was triggered and the outer test was not used to
select a configuration. This is evidence of stage-3 catastrophic forgetting
under full trunk unfreezing, not evidence that the 30-epoch frozen-trunk anchor
was undertrained.

The next matrix is therefore v1.2.42 E-series. It leaves the Direct and S0
Transfer base architectures frozen and tests only prediction-level adaptation:
E0 is the Transfer ensemble anchor; E1 is a constrained Direct-Transfer linear
blend; E2 is a contextual sigmoid gate using base predictions, disagreement,
seed dispersion, molecular weight, effect level, task family, and top-level
taxon; E3 is a small bounded residual head on the same features.

Each base architecture is retrained in five OOF folds inside the locked outer
stage-3 training population. The outer test is omitted from every OOF split,
and the OOF held-out fold cannot be used for early stopping: Direct and each
Transfer stage use only an internal fraction of their current training fold for
model selection. Seeds 42 and 3407 are used for screening. The fixed v1.2.40
seed42 stage-3 validation identities form a reserved meta-selection set and are
excluded from meta fitting. A candidate is eligible only if it improves native
mol/kg R2 and MAE simultaneously on that reserved set. Only after this lock may
seeds 2042 and 8417 be generated, the winning head refitted on four-seed OOF
predictions, and the outer test read once for report-only native mol/kg and
paired common mg/kg metrics.

Remote QA and launch: the related remote protocol suite passed 15 tests. The
first smoke run failed closed before training because stage-3 OOF assignments
did not retain the strict group-key contract; the split builder was corrected
to emit strict contracts and the regression test was expanded. The second
Direct/Transfer smoke completed, with both validators confirming 2,323 fold-1
OOF rows and no outer-test rows. Formal screening started at
2026-07-19T11:03:24+08:00 with three concurrent jobs on the 32 GB RTX 4080
SUPER. The controller log is `outputs/logs/v1_2_42_e_series.log`.

### v1.2.42 E-series OOF no-winner decision

The 20 screening units completed, but the first summary attempt failed closed
because 108 identities from the locked v1.2.40 validation set had no paired OOF
prediction. Raw inspection showed that all 108 belonged to three exact task
routes that the outer-train-only OOF learners consistently skipped below the
minimum-support threshold: ECx_Population (32), LOEC_GeneticDamage (38), and
NOEC_Physiology (38). The summary contract was corrected to exclude only task
routes absent from the complete paired OOF prediction space, record their IDs,
hashes and route counts, and continue to fail on a missing identity from any
OOF-eligible route. This retained 2,325 of 2,433 locked validation identities.

On the retained validation-only set, E0/E1/E2/E3 native mol/kg R2 values were
0.656425/0.655596/0.656414/0.656228 and MAE values were
0.582290/0.581000/0.582328/0.580657. E1 and E3 produced negligible MAE gains
but slightly lower R2, while E2 did not improve either criterion. Therefore no
candidate met the preregistered simultaneous R2-and-MAE rule. The matrix ended
with `selected_candidate=null`; expansion seeds 2042 and 8417 were not run and
the outer test remained unread. The result rejects these prediction-level
adaptation heads as replacements for the frozen Transfer anchor under the
current gate, but does not invalidate the v1.2.40 full-fit outer-test result.

### 2026-07-23 v1.2.44 M10 applicability-domain decision

The locked v1.2.44 random Stage-3 boundary was audited without retraining. The
primary route was the four-seed M10 prediction-level ensemble and M00 was the
fixed comparator; all results remain on the native `neg_log10_mol_kg` scale.
Strict record identities, the 9,724/2,433/3,042 train/validation/test manifests,
and the registered M10/M00 metrics were reproduced before support analysis.

Chemical support used radius-2, 2,048-bit Morgan Tanimoto similarity to unique
canonical parents in Stage-3 training (`C_target`) and Stage-1 fitted source
records (`C_source`). Same-task taxonomy, a six-field same-task experimental
context distance, relative task-head support, and joint local Stage-3 training
record counts completed the C/B/T/L description. Structure-unavailable records
remained an explicit state rather than being assigned similarity zero.

No one of 384 validation-only candidate rules met the preregistered coverage,
tier-size and task-composition requirements. Candidate High coverage did not
exceed 18.29%, and combined High+Moderate coverage did not exceed 49.98%. The
strict locked descriptive rule retained 54 validation and 54 outer-test High
records. Aggregate validation and test MAE decreased from Low/outside through
Moderate to High, but only two test tasks supported a within-task High-versus-Low
comparison of at least 15 rows per side and only one showed the expected
direction. The joint support ranking also did not consistently outperform seed
disagreement for low-coverage error selection.

This is a limitation of the rejection-rule calibration, not evidence that most
tasks are unreliable. Intermediate and Lower-measured support together cover
98.2% of test records; their MAEs (0.516 and 0.579) differ from the full-test MAE
(0.546) by only -0.030 and +0.033. The absence of enough strict-High rows in most
tasks prevents within-task contrast estimation but does not invalidate those
tasks. Figures therefore use neutral density labels and show deviations from the
full-test benchmark rather than a trusted/untrusted visual split.

Decision: report this result as `training-support stratification` or
`record-level support analysis`, not as a calibrated applicability-domain
rejection rule. These labels describe measured training support, not model
pass/fail reliability. The random boundary is chemically interpolation dominated:
98.3% of structure-available test records have an exact Stage-3 training parent,
while 24.75% of all test records have no usable organic molecular structure.
These results do not support new-scaffold, new-family, or universal in-domain
reliability claims. The immutable rule hash was identical before and after
outer-test evaluation; the implementation and manuscript-ready outputs are in
`analysis/applicability_domain/` and its test suite reports 12 passed checks.

The publication presentation was locked on 2026-07-23 without changing any rule,
record assignment, or metric. Main Figure 5 now uses a two-dimensional
chemical-versus-bio-context projection with task support encoded by point size,
occupied-cell conditional-error maps, outer-test-only coverage curves, and
overlapping stratum error distributions. All panels use parenthesized labels
`(a)`–`(d)`. Validation curves and seed-disagreement diagnostics were moved to a
separate supplementary figure. The corresponding bilingual Methods, Results,
Discussion, Conclusions, Figure 5 legend, Table 2, detailed SI methods/results,
and production brief were synchronized in
`三阶段版/论文初稿_v0.4_20260723/`. This change is interpretive and presentational:
the locked quantitative conclusion remains continuous, modest reliability
enrichment rather than a binary trusted/untrusted boundary.

### 2026-07-23 revised multidimensional-support statistics and figure package

A stricter, task-stratified reanalysis was completed in `三阶段版/AD/`. It
retains the locked M10 prediction route and outer-test identities but replaces
presentation-only reliability tiers with prespecified support cells,
within-task risk–coverage ranking, validation-derived tied-quantile strata,
reference-component cluster bootstrap, and within-task permutation tests. The
legacy `ad_rule_locked.json` was not used for selection and was not modified.

The audit partitioned the 3,042 outer-test records into 2,289 support-computable
and 753 support-unavailable records. Chemical support for the inferential panels
is the nearest non-identical Stage-3 training canonical-parent analogue;
exact-parent occurrence remains a separate composition result. The primary
joint score is the bottleneck `S_joint_min=min(C,B,T)` and the geometric mean is
reported only as a sensitivity analysis.

On the task-balanced test risk–coverage analysis, the relative AURC
(`AURC_support - AURC_paired_random`) was 0.005127 for `S_joint_min`
(reference-component cluster-bootstrap 95% CI -0.026898 to 0.036651). The
highest-minus-lowest validation-quantile stratum difference was -0.026105
task-IQR units (95% CI -0.101137 to 0.037490), and continuous Spearman rho was
0.011972 (bootstrap 95% CI -0.038461 to 0.063141). These results do not establish
a stable joint-support prediction-error boundary. Bio-context support alone
showed lower-error ranking enrichment, with relative AURC -0.036222 (95% CI
-0.065377 to -0.001298), but this is an association under the current random
boundary and not a causal or joint-domain effect.

Support-unavailable test records had lower descriptive task-balanced normalized
MAE than support-computable records (0.258052 versus 0.307797). Missing support
therefore cannot be merged with measured low support or interpreted as automatic
unreliability. The reporting decision is to describe continuous multidimensional
training coverage, disclose the bio-context-only association, and avoid binary
inside/outside or reliable/unreliable language. The revised main figure uses
parenthesized `(a1)`, `(a2)`, `(b)`, `(c)`, and `(d)` labels; Figures S1–S10 and
all figure-ready statistics are part of the same package.

The corresponding Chinese and English main text, Supporting Information, and
bilingual figure-production brief were backfilled as a versioned `v0.5 AD
revision` under `三阶段版/AD/manuscript_backfill/`. The previous v0.4 DOCX
files were preserved. Word-to-PDF rendering of all five revised documents
produced 46 pages and passed page-by-page visual inspection without clipping,
overlap, or anomalous page breaks.

### 2026-07-24 M10 random-fivefold and paired target-data learning-curve decision

The v1.2.52 expansion completed all 15 added seed-fold cells and combined them
with the five existing seed-3407 cells. The fixed task-stratified row-random
fivefold assignments were not regenerated. Exact-once OOF coverage was verified
for each seed over 15,199 target records. The four-seed rowwise OOF ensemble
achieved R2 0.690203, RMSE 0.780621, and MAE 0.561950. Across the four
single-seed OOF predictions, the mean±SD values were R2 0.669906±0.010273,
RMSE 0.805715±0.012529, and MAE 0.582087±0.009648. This supports random
resampling and initialization robustness; it is not scaffold extrapolation
evidence and is not merged numerically with the fixed 8:2 main score.

The v1.2.54 paired M10/M00 learning curve completed all 32 new cells. The
10%, 25%, 50%, and 75% Stage-3 train/validation subsets are strictly nested
within the locked v1.2.44 row-random 8:2 boundary, while all fractions retain
the same 3,042-row outer test. The 100% M10/M00 anchors reuse the validated
v1.2.44 four-seed predictions. M10 outperformed M00 in ensemble R2 and MAE at
every target-data fraction. Task-stratified row-paired bootstrap with 20,000
replicates gave positive M00-minus-M10 MAE benefits at all five fractions, and
all five 95% intervals excluded zero.

The inferential decision is deliberately narrower than “enhanced data
efficiency.” The M10 MAE benefit at 10%, 25%, 50%, 75%, and 100% was
0.020277, 0.024839, 0.030073, 0.023561, and 0.033448, respectively. None of
the low-data fractions showed a statistically supported larger benefit than
the full-data contrast; the 10% and 75% excess-benefit intervals were
significantly below zero. The manuscript may retain aquatic-assisted or aquatic
pretraining as a robust incremental improvement that persists in low-data
settings, but it must not claim that aquatic pretraining becomes stronger as
soil labels decrease or that enhanced target-data efficiency has been proven.
