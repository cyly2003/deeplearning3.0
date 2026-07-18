#!/usr/bin/env bash
set -u -o pipefail

MODE="${1:-ensemble_all}"

cd /home/easyai/DL1/ecotox_qsar_transfer || exit 1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
CONFIG="${CONFIG:-configs/experiment.remote.easyai.yaml}"
SOURCE_DB="${SOURCE_DB:-outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite}"
DB="${DB:-outputs/derived/modeling_dataset_v2_0_0_rebuild_no_metal_inorganic.sqlite}"
TRANSFER_TABLE="${TRANSFER_TABLE:-aggregated_task_records_aquatic_soil_ptox_qc_no_metal_inorganic}"
SOIL_TABLE="${SOIL_TABLE:-aggregated_task_records_soil_ptox_qc_no_metal_inorganic}"
SOIL_SPLIT_PREFIX="${SOIL_SPLIT_PREFIX:-NoMetalSoilPtoxQC2_}"
OUT_ROOT="${OUT_ROOT:-outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_remote}"
SUMMARY_OUT="${SUMMARY_OUT:-outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_remote_summary}"
AUDIT_DIR="${AUDIT_DIR:-outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_audit}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
RUN_TIMES="${RUN_TIMES:-${LOG_DIR}/run_v1_2_24_scaffold_cluster_holdout_times.csv}"
FORCE_REBUILD_DB="${FORCE_REBUILD_DB:-0}"
FORCE_REBUILD_SPLITS="${FORCE_REBUILD_SPLITS:-0}"

SEEDS="${SEEDS:-42 1042 2042 3042 4042}"
SPLIT_SEED="${SPLIT_SEED:-42}"
ENSEMBLE_SEEDS="${ENSEMBLE_SEEDS:-$SEEDS}"
FOLDS="${FOLDS:-1 2 3 4 5}"
RUN_HOLDOUT="${RUN_HOLDOUT:-1}"
RUN_5FOLD="${RUN_5FOLD:-1}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-30}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-60}"
F100_CENSORED_WEIGHT="${F100_CENSORED_WEIGHT:-0.01}"
TANIMOTO_THRESHOLD="${TANIMOTO_THRESHOLD:-0.65}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-}"

HOLDOUT_SOIL_SPLIT="${SOIL_SPLIT_PREFIX}G_scaffold_cluster_8_2"
HOLDOUT_TRANSFER_SPLIT="M_v2_aquatic_to_soil_ptox_no_metal_adapt_G_scaffold_cluster_8_2_f100"

ensure_outputs() {
  mkdir -p "$OUT_ROOT" "$SUMMARY_OUT" "$AUDIT_DIR" "$LOG_DIR" "$(dirname "$DB")"
  if [[ ! -s "$RUN_TIMES" ]]; then
    echo "kind,run_name,split_name,start_iso,end_iso,duration_seconds,exit_code" > "$RUN_TIMES"
  fi
}

ensure_outputs

COMMON_ARGS=(
  --config "$CONFIG"
  --db "$DB"
  --run-version "v1.2.24"
  --ablation full
  --batch-size 512
  --scheduler cosine
  --target-standardization per_task_target
  --device cuda:0
  --metric-min-n 5
  --early-stopping
  --early-stopping-patience 15
  --early-stopping-min-delta 0.0
  --source-table "$TRANSFER_TABLE"
  --finetune-learning-rate 0.0003082636455810776
  --finetune-scheduler reduce_on_plateau
  --finetune-validation-fraction 0.2
  --toxicity-binning
  --toxicity-binning-mode aux_classification
  --toxicity-binning-scheme authority_v1
  --toxicity-binning-loss-weight 0.025
  --no-effect-level-weighting
  --censored-loss
  --censored-loss-weight "$F100_CENSORED_WEIGHT"
  --censored-loss-margin 0.0
)

time_command() {
  local kind="$1"
  local run_name="$2"
  local split="$3"
  shift 3
  local start_epoch end_epoch duration status start_iso end_iso
  start_iso="$(date -Is)"
  start_epoch="$(date +%s)"
  echo "[run-start] ${start_iso} kind=${kind} run=${run_name} split=${split}"
  "$@"
  status=$?
  end_epoch="$(date +%s)"
  end_iso="$(date -Is)"
  duration=$((end_epoch - start_epoch))
  echo "${kind},${run_name},${split},${start_iso},${end_iso},${duration},${status}" >> "$RUN_TIMES"
  echo "[run-done] ${end_iso} kind=${kind} run=${run_name} split=${split} duration=${duration}s status=${status}"
  return "$status"
}

weight_label() {
  local value="$1"
  echo "${value/./p}"
}

fold_soil_split() {
  local fold="$1"
  echo "${SOIL_SPLIT_PREFIX}H_scaffold_cluster_5fold_fold${fold}"
}

fold_transfer_split() {
  local fold="$1"
  echo "M_v2_aquatic_to_soil_ptox_no_metal_adapt_H_scaffold_cluster_5fold_fold${fold}_f100"
}

prepare_db() {
  if [[ -s "$DB" && "$FORCE_REBUILD_DB" != "1" ]]; then
    echo "[db-skip-existing] ${DB}"
    return 0
  fi
  echo "[db-build] source=${SOURCE_DB} out=${DB}"
  "$PYTHON_BIN" scripts/build_no_metal_inorganic_dataset.py \
    --source-db "$SOURCE_DB" \
    --out-db "$DB" \
    --force
}

build_soil_splits() {
  if [[ "$FORCE_REBUILD_SPLITS" != "1" ]] && soil_splits_exist; then
    echo "[scaffold-splits-skip-existing] prefix=${SOIL_SPLIT_PREFIX}"
    return 0
  fi
  echo "[scaffold-splits] source_table=${SOIL_TABLE} threshold=${TANIMOTO_THRESHOLD} audit=${AUDIT_DIR}"
  "$PYTHON_BIN" scripts/build_scaffold_cluster_splits.py \
    --db "$DB" \
    --source-table "$SOIL_TABLE" \
    --split-name-prefix "$SOIL_SPLIT_PREFIX" \
    --audit-dir "$AUDIT_DIR" \
    --tanimoto-threshold "$TANIMOTO_THRESHOLD" \
    --invalid-policy exclude \
    --seed "$SPLIT_SEED"
}

soil_splits_exist() {
  "$PYTHON_BIN" - "$DB" "$SOIL_TABLE" "$HOLDOUT_SOIL_SPLIT" "$(fold_soil_split 1)" "$(fold_soil_split 2)" "$(fold_soil_split 3)" "$(fold_soil_split 4)" "$(fold_soil_split 5)" <<'PY'
import sqlite3
import sys

db, source_table, *split_names = sys.argv[1:]
with sqlite3.connect(db) as con:
    row = con.execute(
        "select name from sqlite_master where type='table' and name='split_assignments'"
    ).fetchone()
    if row is None:
        raise SystemExit(1)
    for split_name in split_names:
        count = con.execute(
            """
            select count(*)
            from split_assignments
            where split_name = ? and source_table = ?
            """,
            (split_name, source_table),
        ).fetchone()[0]
        if int(count) == 0:
            raise SystemExit(1)
raise SystemExit(0)
PY
}

build_one_transfer_split() {
  local soil_split="$1"
  local transfer_split="$2"
  echo "[split-build] soil=${soil_split} transfer=${transfer_split}"
  "$PYTHON_BIN" scripts/build_aquatic_soil_adaptation_split.py \
    --db "$DB" \
    --source-table "$TRANSFER_TABLE" \
    --soil-source-table "$SOIL_TABLE" \
    --soil-split-name "$soil_split" \
    --split-name "$transfer_split" \
    --soil-finetune-fraction 1.0 \
    --seed "$SPLIT_SEED"
}

build_transfer_splits() {
  if [[ "$RUN_HOLDOUT" == "1" ]]; then
    build_one_transfer_split "$HOLDOUT_SOIL_SPLIT" "$HOLDOUT_TRANSFER_SPLIT" || return $?
  fi
  if [[ "$RUN_5FOLD" == "1" ]]; then
    local fold
    for fold in $FOLDS; do
      build_one_transfer_split "$(fold_soil_split "$fold")" "$(fold_transfer_split "$fold")" || return $?
    done
  fi
}

run_transfer() {
  local run_name="$1"
  local split="$2"
  local run_seed="$3"
  local out_dir="${OUT_ROOT}/v1.2.24_${run_name}/deep/full/${split}"
  if [[ -s "${out_dir}/predictions.csv" && -s "${out_dir}/manifest.json" && -s "${out_dir}/history.csv" ]]; then
    echo "[skip-existing] transfer ${run_name} ${split}"
    return 0
  fi
  time_command "transfer" "$run_name" "$split" \
    "$PYTHON_BIN" -m qsar_tl.training.train \
      "${COMMON_ARGS[@]}" \
      --out-dir "$OUT_ROOT" \
      --seed "$run_seed" \
      --run-name-zh "$run_name" \
      --split-name "$split" \
      --epochs "$PRETRAIN_EPOCHS" \
      --finetune-epochs "$FINETUNE_EPOCHS" \
      --learning-rate 0.0005 \
      --dropout 0.10 \
      --weight-decay 0.000009856751793848817 \
      --source-weighting-method tanimoto_to_finetune \
      --source-weighting-alpha 1.0
}

run_matrix() {
  local cw fold run_name split run_seed
  cw="$(weight_label "$F100_CENSORED_WEIGHT")"
  for run_seed in $SEEDS; do
    if [[ "$RUN_HOLDOUT" == "1" ]]; then
      run_name="${RUN_NAME_PREFIX}no_metal_scaffold8_2_seed${run_seed}_cebin_lw0025_censored_w${cw}"
      run_transfer "$run_name" "$HOLDOUT_TRANSFER_SPLIT" "$run_seed" || return $?
    fi
    if [[ "$RUN_5FOLD" == "1" ]]; then
      for fold in $FOLDS; do
        run_name="${RUN_NAME_PREFIX}no_metal_scaffold5fold_fold${fold}_seed${run_seed}_cebin_lw0025_censored_w${cw}"
        split="$(fold_transfer_split "$fold")"
        run_transfer "$run_name" "$split" "$run_seed" || return $?
      done
    fi
  done
}

summarize() {
  "$PYTHON_BIN" scripts/summarize_deep_runs.py \
    --root "$OUT_ROOT" \
    --out-dir "$SUMMARY_OUT" \
    --run-times "$RUN_TIMES" || return $?
  "$PYTHON_BIN" scripts/summarize_split_policy_pilot.py \
    --root "$OUT_ROOT" \
    --out-dir "$SUMMARY_OUT" || return $?
}

summarize_ensemble() {
  "$PYTHON_BIN" scripts/summarize_split_policy_ensembles.py \
    --root "$OUT_ROOT" \
    --out-dir "$SUMMARY_OUT" \
    --seeds $ENSEMBLE_SEEDS \
    --write-prediction-rows || return $?
}

run_smoke() {
  PRETRAIN_EPOCHS="${SMOKE_PRETRAIN_EPOCHS:-1}"
  FINETUNE_EPOCHS="${SMOKE_FINETUNE_EPOCHS:-1}"
  FOLDS="${SMOKE_FOLDS:-1}"
  SEEDS="${SMOKE_SEEDS:-42}"
  ENSEMBLE_SEEDS="$SEEDS"
  OUT_ROOT="${OUT_ROOT}_smoke"
  SUMMARY_OUT="${SUMMARY_OUT}_smoke"
  AUDIT_DIR="${AUDIT_DIR}_smoke"
  RUN_TIMES="${LOG_DIR}/run_v1_2_24_scaffold_cluster_holdout_smoke_times.csv"
  RUN_NAME_PREFIX="smoke_"
  ensure_outputs
  echo "[smoke-config] pretrain=${PRETRAIN_EPOCHS} finetune=${FINETUNE_EPOCHS} folds=${FOLDS}"
  prepare_db || return $?
  build_soil_splits || return $?
  build_transfer_splits || return $?
  run_matrix || return $?
  summarize || return $?
}

echo "[start] $(date -Is) v1.2.24 scaffold-cluster holdout mode=${MODE} seeds=${SEEDS} folds=${FOLDS}"
case "$MODE" in
  prepare_db)
    prepare_db
    ;;
  splits)
    prepare_db && build_soil_splits && build_transfer_splits
    ;;
  matrix)
    run_matrix
    ;;
  summarize)
    summarize
    ;;
  ensemble)
    summarize_ensemble
    ;;
  ensemble_all)
    prepare_db && build_soil_splits && build_transfer_splits && run_matrix && summarize && summarize_ensemble
    ;;
  smoke)
    run_smoke
    ;;
  all)
    prepare_db && build_soil_splits && build_transfer_splits && run_matrix && summarize
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use prepare_db, splits, matrix, summarize, ensemble, ensemble_all, smoke, or all." >&2
    exit 2
    ;;
esac
status=$?
echo "[done] $(date -Is) v1.2.24 scaffold-cluster holdout mode=${MODE} status=${status}"
exit "$status"
