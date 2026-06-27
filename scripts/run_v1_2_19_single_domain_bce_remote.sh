#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-matrix}"

PROJECT_DIR="${PROJECT_DIR:-/home/easyai/DL1/ecotox_qsar_transfer}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
CONFIG="${CONFIG:-configs/experiment.remote.easyai.yaml}"
DB="${DB:-outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite}"
OUT_ROOT="${OUT_ROOT:-outputs/experiments/v1_2_19_single_domain_bce_remote}"
SUMMARY_OUT="${SUMMARY_OUT:-outputs/experiments/v1_2_19_single_domain_bce_remote_summary}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
RUN_TIMES="${RUN_TIMES:-${LOG_DIR}/run_v1_2_19_single_domain_bce_times.csv}"

DOMAINS="${DOMAINS:-aquatic soil}"
SPLIT_POLICIES="${SPLIT_POLICIES:-B C E}"
FOLDS="${FOLDS:-1 2 3 4 5}"
SEEDS="${SEEDS:-42 1042 2042 3042 4042}"
EPOCHS="${EPOCHS:-30}"
F100_CENSORED_WEIGHT="${F100_CENSORED_WEIGHT:-0.01}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-}"

mkdir -p "$OUT_ROOT" "$SUMMARY_OUT" "$LOG_DIR"
if [[ ! -s "$RUN_TIMES" ]]; then
  echo "kind,run_name,domain,split_policy,split_name,start_iso,end_iso,duration_seconds,exit_code" > "$RUN_TIMES"
fi

COMMON_ARGS=(
  --config "$CONFIG"
  --db "$DB"
  --out-dir "$OUT_ROOT"
  --run-version "v1.2.19"
  --ablation full
  --batch-size 512
  --scheduler cosine
  --target-standardization per_task_target
  --device cuda:0
  --metric-min-n 5
  --early-stopping
  --early-stopping-patience 15
  --early-stopping-min-delta 0.0
  --finetune-epochs 0
  --source-weighting-method none
  --source-weighting-alpha 1.0
  --toxicity-binning
  --toxicity-binning-mode aux_classification
  --toxicity-binning-scheme authority_v1
  --toxicity-binning-loss-weight 0.025
  --no-effect-level-weighting
  --censored-loss
  --censored-loss-weight "$F100_CENSORED_WEIGHT"
  --censored-loss-margin 0.0
)

source_table_for_domain() {
  case "$1" in
    aquatic) echo "aggregated_task_records_aquatic_ptox_qc" ;;
    soil) echo "aggregated_task_records_soil_ptox_qc" ;;
    *)
      echo "Unknown domain: $1" >&2
      return 2
      ;;
  esac
}

split_for_domain_policy() {
  local domain="$1"
  local policy="$2"
  local fold="${3:-}"
  local prefix
  case "$domain" in
    aquatic) prefix="AquaticPtoxQC2" ;;
    soil) prefix="SoilPtoxQC2" ;;
    *)
      echo "Unknown domain: $domain" >&2
      return 2
      ;;
  esac
  case "$policy" in
    B) echo "${prefix}_B_random_8_2" ;;
    C) echo "${prefix}_C_chemical_holdout_8_2" ;;
    E) echo "${prefix}_E_random_5fold_fold${fold}" ;;
    *)
      echo "Unknown split policy: $policy" >&2
      return 2
      ;;
  esac
}

time_command() {
  local kind="$1"
  local run_name="$2"
  local domain="$3"
  local policy="$4"
  local split="$5"
  shift 5
  local start_epoch end_epoch duration status start_iso end_iso
  start_iso="$(date -Is)"
  start_epoch="$(date +%s)"
  echo "[run-start] ${start_iso} kind=${kind} run=${run_name} domain=${domain} policy=${policy} split=${split}"
  set +e
  "$@"
  status=$?
  set -e
  end_epoch="$(date +%s)"
  end_iso="$(date -Is)"
  duration=$((end_epoch - start_epoch))
  echo "${kind},${run_name},${domain},${policy},${split},${start_iso},${end_iso},${duration},${status}" >> "$RUN_TIMES"
  echo "[run-done] ${end_iso} kind=${kind} run=${run_name} domain=${domain} policy=${policy} split=${split} duration=${duration}s status=${status}"
  return "$status"
}

weight_label() {
  local value="$1"
  echo "${value/./p}"
}

run_exists() {
  local run_name="$1"
  local split="$2"
  local out_dir="${OUT_ROOT}/v1.2.19_${run_name}/deep/full/${split}"
  [[ -s "${out_dir}/predictions.csv" && -s "${out_dir}/manifest.json" && -s "${out_dir}/history.csv" ]]
}

run_single_domain() {
  local domain="$1"
  local policy="$2"
  local fold="$3"
  local seed="$4"
  local source_table split fold_label run_name cw
  source_table="$(source_table_for_domain "$domain")"
  split="$(split_for_domain_policy "$domain" "$policy" "$fold")"
  fold_label=""
  if [[ "$policy" == "E" ]]; then
    fold_label="_fold${fold}"
  fi
  cw="$(weight_label "$F100_CENSORED_WEIGHT")"
  run_name="${RUN_NAME_PREFIX}single_${domain}_${policy}${fold_label}_seed${seed}_cebin_lw0025_censored_w${cw}"
  if run_exists "$run_name" "$split"; then
    echo "[skip-existing] ${run_name} ${split}"
    return 0
  fi
  time_command "single_domain" "$run_name" "$domain" "$policy${fold_label}" "$split" \
    "$PYTHON_BIN" -m qsar_tl.training.train \
      "${COMMON_ARGS[@]}" \
      --source-table "$source_table" \
      --run-name-zh "$run_name" \
      --split-name "$split" \
      --seed "$seed" \
      --epochs "$EPOCHS" \
      --learning-rate 0.0005 \
      --dropout 0.10 \
      --weight-decay 0.000009856751793848817
}

run_matrix() {
  local domain policy fold seed
  for domain in $DOMAINS; do
    for policy in $SPLIT_POLICIES; do
      for seed in $SEEDS; do
        if [[ "$policy" == "E" ]]; then
          for fold in $FOLDS; do
            run_single_domain "$domain" "$policy" "$fold" "$seed"
          done
        else
          run_single_domain "$domain" "$policy" "" "$seed"
        fi
      done
    done
  done
}

run_smoke() {
  local old_domains="$DOMAINS"
  local old_policies="$SPLIT_POLICIES"
  local old_folds="$FOLDS"
  local old_seeds="$SEEDS"
  local old_epochs="$EPOCHS"
  local old_prefix="$RUN_NAME_PREFIX"
  DOMAINS="${SMOKE_DOMAINS:-soil}"
  SPLIT_POLICIES="${SMOKE_SPLIT_POLICIES:-B}"
  FOLDS="${SMOKE_FOLDS:-1}"
  SEEDS="${SMOKE_SEEDS:-42}"
  EPOCHS="${SMOKE_EPOCHS:-1}"
  RUN_NAME_PREFIX="${SMOKE_RUN_NAME_PREFIX:-smoke_}"
  run_matrix
  DOMAINS="$old_domains"
  SPLIT_POLICIES="$old_policies"
  FOLDS="$old_folds"
  SEEDS="$old_seeds"
  EPOCHS="$old_epochs"
  RUN_NAME_PREFIX="$old_prefix"
}

summarize() {
  "$PYTHON_BIN" scripts/summarize_deep_runs.py \
    --root "$OUT_ROOT" \
    --out-dir "$SUMMARY_OUT" \
    --run-times "$RUN_TIMES"
}

echo "[start] $(date -Is) v1.2.19 single-domain BCE mode=${MODE} domains=${DOMAINS} policies=${SPLIT_POLICIES} seeds=${SEEDS}"
case "$MODE" in
  smoke)
    run_smoke
    ;;
  matrix|all)
    run_matrix
    ;;
  summarize)
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use smoke, matrix, all, or summarize." >&2
    exit 2
    ;;
esac
summarize
echo "[done] $(date -Is) v1.2.19 single-domain BCE mode=${MODE}"
