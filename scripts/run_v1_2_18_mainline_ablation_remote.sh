#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-matrix}"

PROJECT_DIR="${PROJECT_DIR:-/home/easyai/DL1/ecotox_qsar_transfer}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
CONFIG="${CONFIG:-configs/experiment.remote.easyai.yaml}"
DB="${DB:-outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite}"
SOURCE_TABLE="${SOURCE_TABLE:-aggregated_task_records_aquatic_soil_ptox_qc}"
SPLIT="${SPLIT:-M_v2_aquatic_to_soil_ptox_adapt_C_f100}"
OUT_ROOT="${OUT_ROOT:-outputs/experiments/v1_2_18_mainline_ablation_remote}"
SUMMARY_OUT="${SUMMARY_OUT:-outputs/experiments/v1_2_18_mainline_ablation_remote_summary}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
RUN_TIMES="${RUN_TIMES:-${LOG_DIR}/run_v1_2_18_mainline_ablation_times.csv}"

SEEDS="${SEEDS:-42 1042 2042 3042 4042}"
MODULE_ABLATIONS="${MODULE_ABLATIONS:-no_fingerprint no_descriptors no_species_lifestage no_duration no_context no_medium_adapter no_molecular_residual}"
STRATEGY_ABLATIONS="${STRATEGY_ABLATIONS:-no_source_weighting no_toxicity_binning no_censored_loss}"
INCLUDE_FULL="${INCLUDE_FULL:-0}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-30}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-60}"
F100_CENSORED_WEIGHT="${F100_CENSORED_WEIGHT:-0.01}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-}"

mkdir -p "$OUT_ROOT" "$SUMMARY_OUT" "$LOG_DIR"
if [[ ! -s "$RUN_TIMES" ]]; then
  echo "kind,run_name,ablation,split_name,start_iso,end_iso,duration_seconds,exit_code" > "$RUN_TIMES"
fi

COMMON_ARGS=(
  --config "$CONFIG"
  --db "$DB"
  --out-dir "$OUT_ROOT"
  --run-version "v1.2.18"
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
  local run_name="$2"
  local ablation="$3"
  local split="$4"
  shift 4
  local start_epoch end_epoch duration status start_iso end_iso
  start_iso="$(date -Is)"
  start_epoch="$(date +%s)"
  echo "[run-start] ${start_iso} kind=${kind} run=${run_name} ablation=${ablation} split=${split}"
  set +e
  "$@"
  status=$?
  set -e
  end_epoch="$(date +%s)"
  end_iso="$(date -Is)"
  duration=$((end_epoch - start_epoch))
  echo "${kind},${run_name},${ablation},${split},${start_iso},${end_iso},${duration},${status}" >> "$RUN_TIMES"
  echo "[run-done] ${end_iso} kind=${kind} run=${run_name} ablation=${ablation} split=${split} duration=${duration}s status=${status}"
  return "$status"
}

weight_label() {
  local value="$1"
  echo "${value/./p}"
}

run_exists() {
  local run_name="$1"
  local ablation="$2"
  local out_dir="${OUT_ROOT}/v1.2.18_${run_name}/deep/${ablation}/${SPLIT}"
  [[ -s "${out_dir}/predictions.csv" && -s "${out_dir}/manifest.json" && -s "${out_dir}/history.csv" ]]
}

run_train() {
  local kind="$1"
  local run_name="$2"
  local ablation="$3"
  local seed="$4"
  shift 4
  if run_exists "$run_name" "$ablation"; then
    echo "[skip-existing] ${kind} ${run_name} ${ablation}"
    return 0
  fi
  time_command "$kind" "$run_name" "$ablation" "$SPLIT" \
    "$PYTHON_BIN" -m qsar_tl.training.train \
      "${COMMON_ARGS[@]}" \
      "$@" \
      --run-name-zh "$run_name" \
      --split-name "$SPLIT" \
      --seed "$seed" \
      --epochs "$PRETRAIN_EPOCHS" \
      --finetune-epochs "$FINETUNE_EPOCHS" \
      --learning-rate 0.0005 \
      --dropout 0.10 \
      --weight-decay 0.000009856751793848817 \
      --ablation "$ablation"
}

run_module_matrix() {
  local seed ablation run_name cw
  cw="$(weight_label "$F100_CENSORED_WEIGHT")"
  for seed in $SEEDS; do
    if [[ "$INCLUDE_FULL" == "1" ]]; then
      run_name="ablation_full_seed${seed}_cebin_lw0025_censored_w${cw}"
      run_name="${RUN_NAME_PREFIX}${run_name}"
      run_train "module" "$run_name" "full" "$seed" "${MAINLINE_STRATEGY_ARGS[@]}"
    fi
    for ablation in $MODULE_ABLATIONS; do
      run_name="ablation_${ablation}_seed${seed}_cebin_lw0025_censored_w${cw}"
      run_name="${RUN_NAME_PREFIX}${run_name}"
      run_train "module" "$run_name" "$ablation" "$seed" "${MAINLINE_STRATEGY_ARGS[@]}"
    done
  done
}

run_strategy_matrix() {
  local seed strategy run_name cw
  cw="$(weight_label "$F100_CENSORED_WEIGHT")"
  for seed in $SEEDS; do
    for strategy in $STRATEGY_ABLATIONS; do
      run_name="strategy_${strategy}_seed${seed}"
      run_name="${RUN_NAME_PREFIX}${run_name}"
      case "$strategy" in
        no_source_weighting)
          run_train "strategy" "$run_name" "full" "$seed" \
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
          run_train "strategy" "$run_name" "full" "$seed" \
            --source-weighting-method tanimoto_to_finetune \
            --source-weighting-alpha 1.0 \
            --no-toxicity-binning \
            --censored-loss \
            --censored-loss-weight "$F100_CENSORED_WEIGHT" \
            --censored-loss-margin 0.0
          ;;
        no_censored_loss)
          run_train "strategy" "$run_name" "full" "$seed" \
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
  done
}

run_smoke() {
  local old_seeds="$SEEDS"
  local old_modules="$MODULE_ABLATIONS"
  local old_strategies="$STRATEGY_ABLATIONS"
  local old_pretrain="$PRETRAIN_EPOCHS"
  local old_finetune="$FINETUNE_EPOCHS"
  local old_prefix="$RUN_NAME_PREFIX"
  SEEDS="${SMOKE_SEEDS:-42}"
  MODULE_ABLATIONS="${SMOKE_MODULE_ABLATIONS:-no_fingerprint}"
  STRATEGY_ABLATIONS="${SMOKE_STRATEGY_ABLATIONS:-no_source_weighting}"
  PRETRAIN_EPOCHS="${SMOKE_PRETRAIN_EPOCHS:-1}"
  FINETUNE_EPOCHS="${SMOKE_FINETUNE_EPOCHS:-1}"
  RUN_NAME_PREFIX="${SMOKE_RUN_NAME_PREFIX:-smoke_}"
  run_module_matrix
  run_strategy_matrix
  SEEDS="$old_seeds"
  MODULE_ABLATIONS="$old_modules"
  STRATEGY_ABLATIONS="$old_strategies"
  PRETRAIN_EPOCHS="$old_pretrain"
  FINETUNE_EPOCHS="$old_finetune"
  RUN_NAME_PREFIX="$old_prefix"
}

summarize() {
  "$PYTHON_BIN" scripts/summarize_deep_runs.py \
    --root "$OUT_ROOT" \
    --out-dir "$SUMMARY_OUT" \
    --run-times "$RUN_TIMES"
}

echo "[start] $(date -Is) v1.2.18 mainline ablation mode=${MODE} seeds=${SEEDS}"
case "$MODE" in
  smoke)
    run_smoke
    ;;
  modules)
    run_module_matrix
    ;;
  strategies)
    run_strategy_matrix
    ;;
  matrix|all)
    run_module_matrix
    run_strategy_matrix
    ;;
  summarize)
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use smoke, modules, strategies, matrix, all, or summarize." >&2
    exit 2
    ;;
esac
summarize
echo "[done] $(date -Is) v1.2.18 mainline ablation mode=${MODE}"
