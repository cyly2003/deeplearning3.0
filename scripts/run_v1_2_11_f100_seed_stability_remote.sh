#!/usr/bin/env bash
set -u -o pipefail

MODE="${1:-matrix}"

cd /home/easyai/DL1/ecotox_qsar_transfer || exit 1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PYTHON_BIN="/opt/anaconda3/bin/python"
CONFIG="configs/experiment.remote.easyai.yaml"
DB="outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite"
TRANSFER_TABLE="aggregated_task_records_aquatic_soil_ptox_qc"
OUT_ROOT="outputs/experiments/v1_2_11_f100_seed_stability_remote"
SUMMARY_OUT="outputs/experiments/v1_2_11_f100_seed_stability_remote_summary"
AUDIT_ROOT="${OUT_ROOT}/audits"
AD_GATE_OUT="${SUMMARY_OUT}/ad_gate_summary.csv"
LOG_DIR="outputs/logs"
RUN_TIMES="${LOG_DIR}/run_v1_2_11_f100_seed_stability_times.csv"

F100_CENSORED_WEIGHT="${F100_CENSORED_WEIGHT:-0.01}"
SEEDS="${SEEDS:-42 1042 2042}"
INCLUDE_CHALLENGER="${INCLUDE_CHALLENGER:-1}"
CHALLENGER_METHOD="${CHALLENGER_METHOD:-tanimoto_proxy_to_finetune}"
CHALLENGER_ALPHA="${CHALLENGER_ALPHA:-0.5}"
CHALLENGER_TAG="${CHALLENGER_TAG:-tanimoto_proxy_a0p5}"

mkdir -p "$OUT_ROOT" "$SUMMARY_OUT" "$AUDIT_ROOT" "$LOG_DIR"
if [[ ! -s "$RUN_TIMES" ]]; then
  echo "kind,run_name,split_name,start_iso,end_iso,duration_seconds,exit_code" > "$RUN_TIMES"
fi

COMMON_ARGS=(
  --config "$CONFIG"
  --db "$DB"
  --out-dir "$OUT_ROOT"
  --run-version "v1.2.11"
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
  --finetune-epochs 60
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
  return 0
}

weight_label() {
  local value="$1"
  echo "${value/./p}"
}

alpha_label() {
  local value="$1"
  echo "${value/./p}"
}

run_transfer() {
  local run_name="$1"
  local seed="$2"
  local source_method="$3"
  local source_alpha="$4"
  local epochs="${5:-30}"
  local finetune_epochs="${6:-60}"
  local split="M_v2_aquatic_to_soil_ptox_adapt_C_f100"
  local out_dir="${OUT_ROOT}/v1.2.11_${run_name}/deep/full/${split}"
  if [[ -s "${out_dir}/predictions.csv" && -s "${out_dir}/manifest.json" && -s "${out_dir}/history.csv" ]]; then
    echo "[skip-existing] transfer ${run_name} ${split}"
    return 0
  fi
  time_command "transfer" "$run_name" "$split" \
    "$PYTHON_BIN" -m qsar_tl.training.train \
      "${COMMON_ARGS[@]}" \
      --seed "$seed" \
      --run-name-zh "$run_name" \
      --split-name "$split" \
      --epochs "$epochs" \
      --finetune-epochs "$finetune_epochs" \
      --learning-rate 0.0005 \
      --dropout 0.10 \
      --weight-decay 0.000009856751793848817 \
      --source-weighting-method "$source_method" \
      --source-weighting-alpha "$source_alpha"
}

prediction_path() {
  local run_name="$1"
  echo "${OUT_ROOT}/v1.2.11_${run_name}/deep/full/M_v2_aquatic_to_soil_ptox_adapt_C_f100/predictions.csv"
}

run_ad_audit() {
  local run_name="$1"
  local split="M_v2_aquatic_to_soil_ptox_adapt_C_f100"
  local pred_path
  pred_path="$(prediction_path "$run_name")"
  local out_dir="${AUDIT_ROOT}/${run_name}"
  if [[ -s "${out_dir}/ad_stratified_metrics.csv" && -s "${out_dir}/ad_audit_manifest.json" ]]; then
    echo "[skip-existing] audit ${run_name}"
    return 0
  fi
  if [[ ! -s "$pred_path" ]]; then
    echo "[skip-missing] audit ${run_name} predictions not found: ${pred_path}" >&2
    return 0
  fi
  time_command "audit" "$run_name" "$split" \
    "$PYTHON_BIN" scripts/audit_prediction_application_domain.py \
      --db "$DB" \
      --source-table "$TRANSFER_TABLE" \
      --split-name "$split" \
      --predictions "$pred_path" \
      --out-dir "$out_dir" \
      --molecular-cache "outputs/features/molecular_features_rdkit_morgan512.jsonl" \
      --fingerprint-size 512 \
      --pca-components 0 \
      --tanimoto-threshold 0.5 \
      --taxon-similarity-threshold 0.8
}

anchor_run_name() {
  local seed="$1"
  local cw
  cw="$(weight_label "$F100_CENSORED_WEIGHT")"
  echo "transfer_f100_anchor_tanimoto_a1_seed${seed}_cebin_lw0025_censored_w${cw}"
}

challenger_run_name() {
  local seed="$1"
  local cw
  cw="$(weight_label "$F100_CENSORED_WEIGHT")"
  echo "transfer_f100_${CHALLENGER_TAG}_seed${seed}_cebin_lw0025_censored_w${cw}"
}

run_seed_matrix() {
  local seed run_name
  for seed in $SEEDS; do
    run_name="$(anchor_run_name "$seed")"
    run_transfer "$run_name" "$seed" "tanimoto_to_finetune" "1.0"
    if [[ "$INCLUDE_CHALLENGER" == "1" ]]; then
      run_name="$(challenger_run_name "$seed")"
      run_transfer "$run_name" "$seed" "$CHALLENGER_METHOD" "$CHALLENGER_ALPHA"
    fi
  done
}

run_seed_audits() {
  local seed run_name
  for seed in $SEEDS; do
    run_name="$(anchor_run_name "$seed")"
    run_ad_audit "$run_name"
    if [[ "$INCLUDE_CHALLENGER" == "1" ]]; then
      run_name="$(challenger_run_name "$seed")"
      run_ad_audit "$run_name"
    fi
  done
}

run_smoke() {
  local seed="${SMOKE_SEED:-42}"
  local run_name
  run_name="smoke_$(anchor_run_name "$seed")"
  run_transfer "$run_name" "$seed" "tanimoto_to_finetune" "1.0" "1" "1"
}

summarize() {
  "$PYTHON_BIN" scripts/summarize_deep_runs.py \
    --root "$OUT_ROOT" \
    --out-dir "$SUMMARY_OUT" \
    --run-times "$RUN_TIMES"
  "$PYTHON_BIN" scripts/summarize_ad_gate.py \
    --audit-root "$AUDIT_ROOT" \
    --out "$AD_GATE_OUT" \
    --split-part test
}

echo "[start] $(date -Is) v1.2.11 f100 seed stability mode=${MODE} seeds=${SEEDS} challenger=${INCLUDE_CHALLENGER}:${CHALLENGER_METHOD}:${CHALLENGER_ALPHA}:${CHALLENGER_TAG}"
case "$MODE" in
  smoke)
    run_smoke
    ;;
  matrix)
    run_seed_matrix
    run_seed_audits
    ;;
  audits)
    run_seed_audits
    ;;
  summarize)
    ;;
  all)
    run_seed_matrix
    run_seed_audits
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use smoke, matrix, audits, summarize, or all." >&2
    exit 2
    ;;
esac
if ! summarize; then
  echo "[summary-failed] $(date -Is) v1.2.11 f100 seed stability mode=${MODE}" >&2
  exit 1
fi
echo "[done] $(date -Is) v1.2.11 f100 seed stability mode=${MODE}"
