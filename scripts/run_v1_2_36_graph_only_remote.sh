#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-priority}"

PROJECT_DIR="${PROJECT_DIR:-/home/easyai/DL1/ecotox_qsar_transfer}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
CONFIG="${CONFIG:-configs/experiment.remote.easyai.graph_only.yaml}"
DB="${DB:-outputs/derived/modeling_dataset_v2_0_0_rebuild_no_metal_inorganic.sqlite}"
SOURCE_TABLE="${SOURCE_TABLE:-aggregated_task_records_aquatic_soil_ptox_qc_no_metal_inorganic}"
RUN_VERSION="${RUN_VERSION:-v1.2.36}"
EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-graph_only_molecule}"
OUT_ROOT="${OUT_ROOT:-outputs/experiments/v1_2_36_graph_only_remote}"
SUMMARY_OUT="${SUMMARY_OUT:-outputs/experiments/v1_2_36_graph_only_remote_summary}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
RUN_TIMES="${RUN_TIMES:-${LOG_DIR}/run_v1_2_36_graph_only_times.csv}"

RANDOM8_SPLIT="${RANDOM8_SPLIT:-M_v2_aquatic_to_soil_ptox_no_metal_adapt_B_random_8_2_f100}"
RANDOM5_SPLIT_PREFIX="${RANDOM5_SPLIT_PREFIX:-M_v2_aquatic_to_soil_ptox_no_metal_adapt_E_random_5fold_fold}"
RANDOM5_SPLIT_SUFFIX="${RANDOM5_SPLIT_SUFFIX:-_f100}"
SCAFFOLD8_SPLIT="${SCAFFOLD8_SPLIT:-M_v2_aquatic_to_soil_ptox_no_metal_adapt_G_scaffold_cluster_8_2_f100}"
SCAFFOLD5_SPLIT_PREFIX="${SCAFFOLD5_SPLIT_PREFIX:-M_v2_aquatic_to_soil_ptox_no_metal_adapt_H_scaffold_cluster_5fold_fold}"
SCAFFOLD5_SPLIT_SUFFIX="${SCAFFOLD5_SPLIT_SUFFIX:-_f100}"

SEEDS="${SEEDS:-2042}"
FOLDS="${FOLDS:-1 2 3 4 5}"
ABLATIONS="${ABLATIONS:-graph_only_molecule}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-30}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-60}"
F100_CENSORED_WEIGHT="${F100_CENSORED_WEIGHT:-0.01}"
SOURCE_WEIGHTING_METHOD="${SOURCE_WEIGHTING_METHOD:-none}"
SOURCE_WEIGHTING_ALPHA="${SOURCE_WEIGHTING_ALPHA:-0.0}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-}"

mkdir -p "$OUT_ROOT" "$SUMMARY_OUT" "$LOG_DIR"
if [[ ! -s "$RUN_TIMES" ]]; then
  echo "kind,split_policy,run_name,ablation,split_name,start_iso,end_iso,duration_seconds,exit_code" > "$RUN_TIMES"
fi

COMMON_ARGS=(
  --config "$CONFIG"
  --db "$DB"
  --out-dir "$OUT_ROOT"
  --run-version "$RUN_VERSION"
  --batch-size 512
  --scheduler cosine
  --target-standardization per_task_target
  --device cuda:0
  --metric-min-n 5
  --early-stopping
  --early-stopping-patience 15
  --early-stopping-min-delta 0.0
  --source-table "$SOURCE_TABLE"
  --finetune-learning-rate 0.0003082636455810776
  --finetune-scheduler reduce_on_plateau
  --finetune-validation-fraction 0.2
  --no-effect-level-weighting
)

MAINLINE_STRATEGY_ARGS=(
  --toxicity-binning
  --toxicity-binning-mode aux_classification
  --toxicity-binning-scheme authority_v1
  --toxicity-binning-loss-weight 0.025
  --censored-loss
  --censored-loss-weight "$F100_CENSORED_WEIGHT"
  --censored-loss-margin 0.0
)
if [[ "$SOURCE_WEIGHTING_METHOD" != "none" && "$SOURCE_WEIGHTING_METHOD" != "off" ]]; then
  MAINLINE_STRATEGY_ARGS=(
    --source-weighting-method "$SOURCE_WEIGHTING_METHOD"
    --source-weighting-alpha "$SOURCE_WEIGHTING_ALPHA"
    "${MAINLINE_STRATEGY_ARGS[@]}"
  )
fi

time_command() {
  local kind="$1"
  local split_policy="$2"
  local run_name="$3"
  local ablation="$4"
  local split="$5"
  shift 5
  local start_epoch end_epoch duration status start_iso end_iso
  start_iso="$(date -Is)"
  start_epoch="$(date +%s)"
  echo "[run-start] ${start_iso} kind=${kind} policy=${split_policy} run=${run_name} ablation=${ablation} split=${split}"
  set +e
  "$@"
  status=$?
  set -e
  end_epoch="$(date +%s)"
  end_iso="$(date -Is)"
  duration=$((end_epoch - start_epoch))
  echo "${kind},${split_policy},${run_name},${ablation},${split},${start_iso},${end_iso},${duration},${status}" >> "$RUN_TIMES"
  echo "[run-done] ${end_iso} kind=${kind} policy=${split_policy} run=${run_name} ablation=${ablation} split=${split} duration=${duration}s status=${status}"
  return "$status"
}

random5_split() {
  local fold="$1"
  echo "${RANDOM5_SPLIT_PREFIX}${fold}${RANDOM5_SPLIT_SUFFIX}"
}

scaffold5_split() {
  local fold="$1"
  echo "${SCAFFOLD5_SPLIT_PREFIX}${fold}${SCAFFOLD5_SPLIT_SUFFIX}"
}

run_exists() {
  local run_name="$1"
  local ablation="$2"
  local split="$3"
  local out_dir="${OUT_ROOT}/${RUN_VERSION}_${run_name}/deep/${ablation}/${split}"
  [[ -s "${out_dir}/predictions.csv" && -s "${out_dir}/manifest.json" && -s "${out_dir}/history.csv" ]]
}

run_train() {
  local kind="$1"
  local split_policy="$2"
  local run_name="$3"
  local ablation="$4"
  local split="$5"
  local seed="$6"
  shift 6
  if run_exists "$run_name" "$ablation" "$split"; then
    echo "[skip-existing] ${kind} ${split_policy} ${run_name} ${ablation} ${split}"
    return 0
  fi
  time_command "$kind" "$split_policy" "$run_name" "$ablation" "$split" \
    "$PYTHON_BIN" -m qsar_tl.training.train \
      "${COMMON_ARGS[@]}" \
      "${MAINLINE_STRATEGY_ARGS[@]}" \
      "$@" \
      --run-name-zh "$run_name" \
      --split-name "$split" \
      --seed "$seed" \
      --epochs "$PRETRAIN_EPOCHS" \
      --finetune-epochs "$FINETUNE_EPOCHS" \
      --learning-rate 0.0005 \
      --dropout 0.10 \
      --weight-decay 0.000009856751793848817 \
      --ablation "$ablation"
}

run_split() {
  local kind="$1"
  local split_policy="$2"
  local split="$3"
  local fold_label="${4:-}"
  local seed ablation run_name
  for seed in $SEEDS; do
    for ablation in $ABLATIONS; do
      run_name="${RUN_NAME_PREFIX}${split_policy}${fold_label}_${ablation}_seed${seed}_${EXPERIMENT_LABEL}"
      run_train "$kind" "$split_policy" "$run_name" "$ablation" "$split" "$seed"
    done
  done
}

run_priority() {
  run_split "priority" "random8_2" "$RANDOM8_SPLIT" ""
  run_split "priority" "scaffold_cluster_8_2" "$SCAFFOLD8_SPLIT" ""
}

run_random() {
  local fold
  run_split "random" "random8_2" "$RANDOM8_SPLIT" ""
  for fold in $FOLDS; do
    run_split "random" "random5fold" "$(random5_split "$fold")" "_fold${fold}"
  done
}

run_scaffold() {
  local fold
  run_split "scaffold" "scaffold_cluster_8_2" "$SCAFFOLD8_SPLIT" ""
  for fold in $FOLDS; do
    run_split "scaffold" "scaffold_cluster_5fold" "$(scaffold5_split "$fold")" "_fold${fold}"
  done
}

run_smoke() {
  local old_seeds="$SEEDS"
  local old_folds="$FOLDS"
  local old_ablations="$ABLATIONS"
  local old_pretrain="$PRETRAIN_EPOCHS"
  local old_finetune="$FINETUNE_EPOCHS"
  local old_prefix="$RUN_NAME_PREFIX"
  local old_out_root="$OUT_ROOT"
  local old_summary_out="$SUMMARY_OUT"
  local old_run_times="$RUN_TIMES"
  SEEDS="${SMOKE_SEEDS:-2042}"
  FOLDS="${SMOKE_FOLDS:-1}"
  ABLATIONS="${SMOKE_ABLATIONS:-full}"
  PRETRAIN_EPOCHS="${SMOKE_PRETRAIN_EPOCHS:-1}"
  FINETUNE_EPOCHS="${SMOKE_FINETUNE_EPOCHS:-1}"
  RUN_NAME_PREFIX="${SMOKE_RUN_NAME_PREFIX:-smoke_}"
  OUT_ROOT="${old_out_root}_smoke"
  SUMMARY_OUT="${old_summary_out}_smoke"
  RUN_TIMES="${LOG_DIR}/run_v1_2_36_graph_only_smoke_times.csv"
  mkdir -p "$OUT_ROOT" "$SUMMARY_OUT" "$LOG_DIR"
  if [[ ! -s "$RUN_TIMES" ]]; then
    echo "kind,split_policy,run_name,ablation,split_name,start_iso,end_iso,duration_seconds,exit_code" > "$RUN_TIMES"
  fi
  run_priority
  SEEDS="$old_seeds"
  FOLDS="$old_folds"
  ABLATIONS="$old_ablations"
  PRETRAIN_EPOCHS="$old_pretrain"
  FINETUNE_EPOCHS="$old_finetune"
  RUN_NAME_PREFIX="$old_prefix"
  OUT_ROOT="$old_out_root"
  SUMMARY_OUT="$old_summary_out"
  RUN_TIMES="$old_run_times"
}

summarize() {
  "$PYTHON_BIN" scripts/summarize_deep_runs.py \
    --root "$OUT_ROOT" \
    --out-dir "$SUMMARY_OUT" \
    --run-times "$RUN_TIMES"
}

echo "[start] $(date -Is) ${RUN_VERSION} ${EXPERIMENT_LABEL} mode=${MODE} config=${CONFIG} seeds=${SEEDS} folds=${FOLDS} ablations=${ABLATIONS}"
case "$MODE" in
  smoke)
    run_smoke
    ;;
  priority)
    run_priority
    ;;
  random)
    run_random
    ;;
  scaffold)
    run_scaffold
    ;;
  all|matrix)
    run_random
    run_scaffold
    ;;
  summarize)
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use smoke, priority, random, scaffold, all, matrix, or summarize." >&2
    exit 2
    ;;
esac
summarize
echo "[done] $(date -Is) ${RUN_VERSION} ${EXPERIMENT_LABEL} mode=${MODE}"
