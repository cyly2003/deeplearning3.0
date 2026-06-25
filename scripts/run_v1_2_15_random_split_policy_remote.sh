#!/usr/bin/env bash
set -u -o pipefail

MODE="${1:-all}"

cd /home/easyai/DL1/ecotox_qsar_transfer || exit 1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
CONFIG="${CONFIG:-configs/experiment.remote.easyai.yaml}"
DB="${DB:-outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite}"
TRANSFER_TABLE="${TRANSFER_TABLE:-aggregated_task_records_aquatic_soil_ptox_qc}"
SOIL_TABLE="${SOIL_TABLE:-aggregated_task_records_soil_ptox_qc}"
OUT_ROOT="${OUT_ROOT:-outputs/experiments/v1_2_15_random_split_policy_formal_remote}"
SUMMARY_OUT="${SUMMARY_OUT:-outputs/experiments/v1_2_15_random_split_policy_formal_remote_summary}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
RUN_TIMES="${RUN_TIMES:-${LOG_DIR}/run_v1_2_15_random_split_policy_times.csv}"

SEED="${SEED:-42}"
FOLDS="${FOLDS:-1 2 3 4 5}"
RUN_RANDOM8="${RUN_RANDOM8:-1}"
RUN_5FOLD="${RUN_5FOLD:-1}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-30}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-60}"
F100_CENSORED_WEIGHT="${F100_CENSORED_WEIGHT:-0.01}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-}"

RANDOM8_SOIL_SPLIT="SoilPtoxQC2_B_random_8_2"
RANDOM8_TRANSFER_SPLIT="M_v2_aquatic_to_soil_ptox_adapt_B_random_8_2_f100"

ensure_outputs() {
  mkdir -p "$OUT_ROOT" "$SUMMARY_OUT" "$LOG_DIR"
  if [[ ! -s "$RUN_TIMES" ]]; then
    echo "kind,run_name,split_name,start_iso,end_iso,duration_seconds,exit_code" > "$RUN_TIMES"
  fi
}

ensure_outputs

COMMON_ARGS=(
  --config "$CONFIG"
  --db "$DB"
  --run-version "v1.2.15"
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

fold_transfer_split() {
  local fold="$1"
  echo "M_v2_aquatic_to_soil_ptox_adapt_E_random_5fold_fold${fold}_f100"
}

fold_soil_split() {
  local fold="$1"
  echo "SoilPtoxQC2_E_random_5fold_fold${fold}"
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
    --seed "$SEED"
}

build_splits() {
  if [[ "$RUN_RANDOM8" == "1" ]]; then
    build_one_transfer_split "$RANDOM8_SOIL_SPLIT" "$RANDOM8_TRANSFER_SPLIT" || return $?
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
  local out_dir="${OUT_ROOT}/v1.2.15_${run_name}/deep/full/${split}"
  if [[ -s "${out_dir}/predictions.csv" && -s "${out_dir}/manifest.json" && -s "${out_dir}/history.csv" ]]; then
    echo "[skip-existing] transfer ${run_name} ${split}"
    return 0
  fi
  time_command "transfer" "$run_name" "$split" \
    "$PYTHON_BIN" -m qsar_tl.training.train \
      "${COMMON_ARGS[@]}" \
      --out-dir "$OUT_ROOT" \
      --seed "$SEED" \
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
  local cw fold run_name split
  cw="$(weight_label "$F100_CENSORED_WEIGHT")"
  if [[ "$RUN_RANDOM8" == "1" ]]; then
    run_name="${RUN_NAME_PREFIX}transfer_f100_anchor_random8_2_seed${SEED}_cebin_lw0025_censored_w${cw}"
    run_transfer "$run_name" "$RANDOM8_TRANSFER_SPLIT" || return $?
  fi
  if [[ "$RUN_5FOLD" == "1" ]]; then
    for fold in $FOLDS; do
      run_name="${RUN_NAME_PREFIX}transfer_f100_anchor_random5fold_fold${fold}_seed${SEED}_cebin_lw0025_censored_w${cw}"
      split="$(fold_transfer_split "$fold")"
      run_transfer "$run_name" "$split" || return $?
    done
  fi
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

run_smoke() {
  local old_pretrain="$PRETRAIN_EPOCHS"
  local old_finetune="$FINETUNE_EPOCHS"
  local old_folds="$FOLDS"
  local old_out_root="$OUT_ROOT"
  local old_summary_out="$SUMMARY_OUT"
  local old_run_times="$RUN_TIMES"
  local old_run_name_prefix="$RUN_NAME_PREFIX"
  PRETRAIN_EPOCHS="${SMOKE_PRETRAIN_EPOCHS:-1}"
  FINETUNE_EPOCHS="${SMOKE_FINETUNE_EPOCHS:-1}"
  FOLDS="${SMOKE_FOLDS:-1}"
  OUT_ROOT="${old_out_root}_smoke"
  SUMMARY_OUT="${old_summary_out}_smoke"
  RUN_TIMES="${LOG_DIR}/run_v1_2_15_random_split_policy_smoke_times.csv"
  RUN_NAME_PREFIX="smoke_"
  ensure_outputs
  echo "[smoke-config] pretrain=${PRETRAIN_EPOCHS} finetune=${FINETUNE_EPOCHS} folds=${FOLDS}"
  build_splits || return $?
  run_matrix || return $?
  summarize || return $?
  PRETRAIN_EPOCHS="$old_pretrain"
  FINETUNE_EPOCHS="$old_finetune"
  FOLDS="$old_folds"
  OUT_ROOT="$old_out_root"
  SUMMARY_OUT="$old_summary_out"
  RUN_TIMES="$old_run_times"
  RUN_NAME_PREFIX="$old_run_name_prefix"
}

echo "[start] $(date -Is) v1.2.15 random split policy mode=${MODE} seed=${SEED} folds=${FOLDS}"
case "$MODE" in
  splits)
    build_splits
    ;;
  matrix)
    run_matrix
    ;;
  summarize)
    summarize
    ;;
  smoke)
    run_smoke
    ;;
  all)
    build_splits && run_matrix && summarize
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use splits, matrix, summarize, smoke, or all." >&2
    exit 2
    ;;
esac
status=$?
echo "[done] $(date -Is) v1.2.15 random split policy mode=${MODE} status=${status}"
exit "$status"
