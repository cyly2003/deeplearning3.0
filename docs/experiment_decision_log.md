# Experiment Decision Log

This file is the human-readable decision layer for the v1.2 transfer-learning
experiments. Use it with `docs/experiment_registry.csv`: the registry stores
where each result lives, while this log records why each strategy was kept or
stopped.

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
