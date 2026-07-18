# Code Organization And Experiment Scripts

This note explains how to treat the current scripts and code files when
rerunning or extending experiments. It is intentionally practical: the goal is
to prevent old scripts from being mistaken for current mainline code.

## Current Problem

The repository now contains several generations of experiment launch scripts.
Many scripts encode a specific data version, source table, split name,
hyperparameter choice, and output directory. Newer scripts are not necessarily
compatible with older data versions, and older scripts may reproduce an older
question rather than the current best method.

The current branch is `codex/authority-toxicity-binning-matrix`, but it also
contains later work through v1.2.13. That means branch name alone is not enough
to reconstruct the experiment history. Use the registry and decision log first.

## Script Categories

### Stable pipeline entry points

These are reusable entry points or wrappers:

- `qsar_tl/training/train.py`
- `qsar_tl/training/deep_train.py`
- `qsar_tl/training/deep_experiment.py`
- `scripts/train_remote.ps1`
- `scripts/train_remote_batch.ps1`
- `scripts/sync_to_server.ps1`
- `scripts/sync_from_server.ps1`

Expected behavior:

- Training reads a SQLite database and split assignments.
- Training writes model artifacts and CSV outputs under `outputs/experiments`.
- Sync scripts can change remote files and should be treated as operational
  scripts, not as harmless analysis scripts.

### Versioned experiment launchers

These scripts reproduce specific experiment windows:

- `scripts/run_v1_2_6_authority_binning_matrix_remote.sh`
- `scripts/run_v1_2_7_censored_ordinal_ad_first_batch_remote.sh`
- `scripts/run_v1_2_8_censored_loss_matrix_remote.sh`
- `scripts/run_v1_2_9_proxy_source_rule_remote.sh`
- `scripts/run_v1_2_10_f100_proxy_alpha_remote.sh`
- `scripts/run_v1_2_11_f100_seed_stability_remote.sh`
- `scripts/run_v1_2_12_f100_5seed_confirmation_remote.sh`
- `scripts/run_v1_2_13_anchor_validation_policy_remote.sh`

Expected behavior:

- They usually do not modify the original raw ECOTOX data.
- They read `outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite`.
- They write new experiment folders, summary folders, audit folders, and timing
  CSV files.
- They may skip existing output directories if required files already exist.
- They encode fixed assumptions. Do not use an old script as a template without
  checking its source table, split, seeds, and loss settings.

Current mainline launcher:

- `scripts/run_v1_2_12_f100_5seed_confirmation_remote.sh`

Current sensitivity-only launcher:

- `scripts/run_v1_2_13_anchor_validation_policy_remote.sh`

Current soil mg/kg target-scale local补跑 launcher:

- `scripts/run_v1_2_38_soil_mgkg_random_local.ps1`

### Summary and audit scripts

These scripts are mostly read-only with respect to the source database and write
derived CSV summaries:

- `scripts/summarize_deep_runs.py`
- `scripts/summarize_ad_gate.py`
- `scripts/summarize_seed_ensembles.py`
- `scripts/audit_prediction_application_domain.py`
- `scripts/audit_prediction_medium_counts.py`
- `scripts/audit_proxy_bins.py`
- `scripts/audit_censored_records.py`

Expected behavior:

- They read predictions, manifests, config/preprocessing files, or SQLite tables.
- They write summary CSVs, audit CSVs, or diagnostic files under
  `outputs/experiments` or a specified output path.
- They should not be used to build or overwrite core data tables.

### Database and split builders

These scripts can modify SQLite tables and should be treated as data-building
steps:

- `scripts/build_modeling_tables.py`
- `scripts/build_task_tables.py`
- `scripts/build_medium_domain_tables.py`
- `scripts/build_aquatic_soil_adaptation_split.py`
- CLI commands such as `python -m qsar_tl.cli generate-split`

Expected behavior:

- They may create, drop, delete, or insert SQLite tables.
- `build_aquatic_soil_adaptation_split.py` deletes and reinserts rows in
  `split_assignments` for the requested split names.
- `build_medium_domain_tables.py` drops and recreates medium-domain derived
  tables.
- These should not be run casually against the current main database unless the
  target database and output split/table names are explicit and backed up.

## Answer To The Reproducibility Question

It is only partly correct that "new experiments can be run just by executing the
corresponding test script."

For most v1.2 remote experiment launchers, yes: they are designed to read the
current v2 derived database and write new outputs under `outputs/experiments`.
They should not change the original raw data or the core source scripts.

But there are important exceptions:

- Split-building scripts can modify `split_assignments` in the SQLite database.
- Medium-domain and task/modeling table builders can drop and recreate derived
  tables.
- Sync scripts can overwrite remote code, config, feature cache, database
  copies, or output folders depending on flags.
- Training scripts can overwrite an existing run directory if the same output
  directory is reused and the script does not skip it.
- Old scripts may target older database versions or older assumptions, even if
  they still run.

Therefore the safe rule is:

1. Treat `run_v1_2_*.sh` scripts as reproducible experiment launchers, not unit
   tests.
2. Treat `build_*`, `generate-split`, and sync scripts as state-changing.
3. Before rerunning, confirm database path, source table, split name, output
   root, seed set, and current Git commit.
4. Prefer creating a new output root for every new experiment.
5. Add the experiment to `docs/experiment_registry.csv` before launch.

## Recommended Folder Policy

Keep current files where they are for now, but use the following policy going
forward:

- `qsar_tl/`: reusable library and pipeline code only.
- `scripts/run_v1_2_*.sh`: frozen experiment launchers. Do not edit after a run
  has been used for conclusions; create a new versioned script instead.
- `scripts/summarize_*.py`: reusable summary scripts.
- `scripts/audit_*.py`: reusable audit scripts.
- `scripts/build_*.py`: state-changing data construction scripts.
- `outputs/experiments/<experiment_id>_...`: raw experiment artifacts.
- `outputs/experiments/<experiment_id>_..._summary`: compact summary artifacts.
- `docs/experiment_registry.csv`: first lookup table.
- `docs/experiment_decision_log.md`: final decision narrative.

## Git Policy For Future Runs

Before a new experiment family:

1. Create a descriptive branch, for example
   `codex/v1-2-14-risk-aware-eval`.
2. Commit code and script changes before launching remote runs.
3. Record the commit hash in the registry.
4. Do not edit a completed experiment launcher in place.

After a run finishes:

1. Commit summary scripts, launch scripts, registry updates, and decision log
   updates.
2. Do not commit large SQLite databases or model weights unless intentionally
   using Git LFS or a separate artifact store.
3. Push the branch so the exact code state is backed up before analysis
   continues.
