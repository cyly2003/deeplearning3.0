#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-priority}"

PROJECT_DIR="${PROJECT_DIR:-/home/easyai/DL1/ecotox_qsar_transfer}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
CONFIG="${CONFIG:-configs/experiment.remote.easyai.yaml}"
DB="${DB:-outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite}"
SOURCE_TABLE="${SOURCE_TABLE:-aggregated_task_records_aquatic_soil_ptox_qc}"
OUT_ROOT="${OUT_ROOT:-outputs/experiments/v1_2_21_random_split_ablation_remote}"
SUMMARY_OUT="${SUMMARY_OUT:-outputs/experiments/v1_2_21_random_split_ablation_remote_summary}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
RUN_TIMES="${RUN_TIMES:-${LOG_DIR}/run_v1_2_21_random_split_ablation_times.csv}"

RANDOM8_SPLIT="${RANDOM8_SPLIT:-M_v2_aquatic_to_soil_ptox_adapt_B_random_8_2_f100}"
FIVEFOLD_SPLIT_PREFIX="${FIVEFOLD_SPLIT_PREFIX:-M_v2_aquatic_to_soil_ptox_adapt_E_random_5fold_fold}"
FIVEFOLD_SPLIT_SUFFIX="${FIVEFOLD_SPLIT_SUFFIX:-_f100}"
RANDOM8_SEEDS="${RANDOM8_SEEDS:-42 1042 2042 3042 4042}"
FIVEFOLD_SEEDS="${FIVEFOLD_SEEDS:-42}"
FOLDS="${FOLDS:-1 2 3 4 5}"
MODULE_ABLATIONS="${MODULE_ABLATIONS:-no_context no_species_lifestage no_molecular_residual}"
STRATEGY_ABLATIONS="${STRATEGY_ABLATIONS:-no_source_weighting no_toxicity_binning no_censored_loss}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-30}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-60}"
F100_CENSORED_WEIGHT="${F100_CENSORED_WEIGHT:-0.01}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-}"

mkdir -p "$OUT_ROOT" "$SUMMARY_OUT" "$LOG_DIR"
if [[ ! -s "$RUN_TIMES" ]]; then
  echo "kind,split_policy,run_name,ablation,split_name,start_iso,end_iso,duration_seconds,exit_code" > "$RUN_TIMES"
fi

COMMON_ARGS=(
  --config "$CONFIG"
  --db "$DB"
  --out-dir "$OUT_ROOT"
  --run-version "v1.2.21"
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
  --source-weighting-method tanimoto_to_finetune
  --source-weighting-alpha 1.0
  --toxicity-binning
  --toxicity-binning-mode aux_classification
  --toxicity-binning-scheme authority_v1
  --toxicity-binning-loss-weight 0.025
  --censored-loss
  --censored-loss-weight "$F100_CENSORED_WEIGHT"
  --censored-loss-margin 0.0
)

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

weight_label() {
  local value="$1"
  echo "${value/./p}"
}

fivefold_split() {
  local fold="$1"
  echo "${FIVEFOLD_SPLIT_PREFIX}${fold}${FIVEFOLD_SPLIT_SUFFIX}"
}

run_exists() {
  local run_name="$1"
  local ablation="$2"
  local split="$3"
  local out_dir="${OUT_ROOT}/v1.2.21_${run_name}/deep/${ablation}/${split}"
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

run_module_ablation() {
  local split_policy="$1"
  local split="$2"
  local seed="$3"
  local fold_label="$4"
  local ablation run_name cw
  cw="$(weight_label "$F100_CENSORED_WEIGHT")"
  for ablation in $MODULE_ABLATIONS; do
    run_name="${RUN_NAME_PREFIX}${split_policy}${fold_label}_ablation_${ablation}_seed${seed}_cebin_lw0025_censored_w${cw}"
    run_train "module" "$split_policy" "$run_name" "$ablation" "$split" "$seed" "${MAINLINE_STRATEGY_ARGS[@]}"
  done
}

run_strategy_ablation() {
  local split_policy="$1"
  local split="$2"
  local seed="$3"
  local fold_label="$4"
  local strategy run_name
  for strategy in $STRATEGY_ABLATIONS; do
    run_name="${RUN_NAME_PREFIX}${split_policy}${fold_label}_strategy_${strategy}_seed${seed}"
    case "$strategy" in
      no_source_weighting)
        run_train "strategy" "$split_policy" "$run_name" "full" "$split" "$seed" \
          --source-weighting-method none \
          --source-weighting-alpha 1.0 \
          --toxicity-binning \
          --toxicity-binning-mode aux_classification \
          --toxicity-binning-scheme authority_v1 \
          --toxicity-binning-loss-weight 0.025 \
          --censored-loss \
          --censored-loss-weight "$F100_CENSORED_WEIGHT" \
          --censored-loss-margin 0.0
        ;;
      no_toxicity_binning)
        run_train "strategy" "$split_policy" "$run_name" "full" "$split" "$seed" \
          --source-weighting-method tanimoto_to_finetune \
          --source-weighting-alpha 1.0 \
          --no-toxicity-binning \
          --censored-loss \
          --censored-loss-weight "$F100_CENSORED_WEIGHT" \
          --censored-loss-margin 0.0
        ;;
      no_censored_loss)
        run_train "strategy" "$split_policy" "$run_name" "full" "$split" "$seed" \
          --source-weighting-method tanimoto_to_finetune \
          --source-weighting-alpha 1.0 \
          --toxicity-binning \
          --toxicity-binning-mode aux_classification \
          --toxicity-binning-scheme authority_v1 \
          --toxicity-binning-loss-weight 0.025 \
          --no-censored-loss
        ;;
      *)
        echo "Unknown strategy ablation: ${strategy}" >&2
        return 2
        ;;
    esac
  done
}

run_one_split_policy() {
  local split_policy="$1"
  local split="$2"
  local seed="$3"
  local fold_label="${4:-}"
  run_module_ablation "$split_policy" "$split" "$seed" "$fold_label"
  run_strategy_ablation "$split_policy" "$split" "$seed" "$fold_label"
}

run_random8() {
  local seed
  for seed in $RANDOM8_SEEDS; do
    run_one_split_policy "random8_2" "$RANDOM8_SPLIT" "$seed" ""
  done
}

run_fivefold() {
  local seed fold
  for seed in $FIVEFOLD_SEEDS; do
    for fold in $FOLDS; do
      run_one_split_policy "random5fold" "$(fivefold_split "$fold")" "$seed" "_fold${fold}"
    done
  done
}

run_smoke() {
  local old_random8_seeds="$RANDOM8_SEEDS"
  local old_fivefold_seeds="$FIVEFOLD_SEEDS"
  local old_folds="$FOLDS"
  local old_modules="$MODULE_ABLATIONS"
  local old_strategies="$STRATEGY_ABLATIONS"
  local old_pretrain="$PRETRAIN_EPOCHS"
  local old_finetune="$FINETUNE_EPOCHS"
  local old_prefix="$RUN_NAME_PREFIX"
  local old_out_root="$OUT_ROOT"
  local old_summary_out="$SUMMARY_OUT"
  local old_run_times="$RUN_TIMES"
  RANDOM8_SEEDS="${SMOKE_RANDOM8_SEEDS:-42}"
  FIVEFOLD_SEEDS="${SMOKE_FIVEFOLD_SEEDS:-42}"
  FOLDS="${SMOKE_FOLDS:-1}"
  MODULE_ABLATIONS="${SMOKE_MODULE_ABLATIONS:-no_context}"
  STRATEGY_ABLATIONS="${SMOKE_STRATEGY_ABLATIONS:-no_source_weighting}"
  PRETRAIN_EPOCHS="${SMOKE_PRETRAIN_EPOCHS:-1}"
  FINETUNE_EPOCHS="${SMOKE_FINETUNE_EPOCHS:-1}"
  RUN_NAME_PREFIX="${SMOKE_RUN_NAME_PREFIX:-smoke_}"
  OUT_ROOT="${old_out_root}_smoke"
  SUMMARY_OUT="${old_summary_out}_smoke"
  RUN_TIMES="${LOG_DIR}/run_v1_2_21_random_split_ablation_smoke_times.csv"
  mkdir -p "$OUT_ROOT" "$SUMMARY_OUT" "$LOG_DIR"
  if [[ ! -s "$RUN_TIMES" ]]; then
    echo "kind,split_policy,run_name,ablation,split_name,start_iso,end_iso,duration_seconds,exit_code" > "$RUN_TIMES"
  fi
  run_random8
  run_fivefold
  RANDOM8_SEEDS="$old_random8_seeds"
  FIVEFOLD_SEEDS="$old_fivefold_seeds"
  FOLDS="$old_folds"
  MODULE_ABLATIONS="$old_modules"
  STRATEGY_ABLATIONS="$old_strategies"
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

echo "[start] $(date -Is) v1.2.21 random split ablation mode=${MODE} random8_seeds=${RANDOM8_SEEDS} fivefold_seeds=${FIVEFOLD_SEEDS} folds=${FOLDS}"
case "$MODE" in
  smoke)
    run_smoke
    ;;
  random8)
    run_random8
    ;;
  fivefold)
    run_fivefold
    ;;
  priority|matrix|all)
    run_random8
    run_fivefold
    ;;
  summarize)
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use smoke, random8, fivefold, priority, matrix, all, or summarize." >&2
    exit 2
    ;;
esac
summarize
echo "[done] $(date -Is) v1.2.21 random split ablation mode=${MODE}"
