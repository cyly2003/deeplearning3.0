#!/usr/bin/env bash
set -u -o pipefail

MODE="${1:-smoke}"

cd /home/easyai/DL1/ecotox_qsar_transfer || exit 1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PYTHON_BIN="/opt/anaconda3/bin/python"
CONFIG="configs/experiment.remote.easyai.yaml"
DB="outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite"
TRANSFER_TABLE="aggregated_task_records_aquatic_soil_ptox_qc"
OUT_ROOT="outputs/experiments/v1_2_8_censored_loss_matrix_remote"
SUMMARY_OUT="outputs/experiments/v1_2_8_censored_loss_matrix_remote_summary"
AD_GATE_OUT="${SUMMARY_OUT}/v1_2_7_ce_best_ad_gate_summary.csv"
V127_AUDIT_ROOT="outputs/experiments/v1_2_7_censored_ordinal_ad_first_batch_remote/audits"
LOG_DIR="outputs/logs"
RUN_TIMES="${LOG_DIR}/run_v1_2_8_censored_loss_matrix_times.csv"

mkdir -p "$OUT_ROOT" "$SUMMARY_OUT" "$LOG_DIR"
if [[ ! -s "$RUN_TIMES" ]]; then
  echo "kind,run_name,split_name,start_iso,end_iso,duration_seconds,exit_code" > "$RUN_TIMES"
fi

COMMON_ARGS=(
  --config "$CONFIG"
  --db "$DB"
  --out-dir "$OUT_ROOT"
  --run-version "v1.2.8"
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
  --source-weighting-method tanimoto_to_finetune
  --source-weighting-alpha 1.0
  --toxicity-binning
  --toxicity-binning-mode aux_classification
  --toxicity-binning-scheme authority_v1
  --no-effect-level-weighting
  --censored-loss
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

run_transfer() {
  local run_name="$1"
  local split="$2"
  local toxicity_loss="$3"
  local censored_loss="$4"
  local epochs="${5:-30}"
  local finetune_epochs="${6:-60}"
  local out_dir="${OUT_ROOT}/v1.2.8_${run_name}/deep/full/${split}"
  if [[ -s "${out_dir}/predictions.csv" && -s "${out_dir}/manifest.json" && -s "${out_dir}/history.csv" ]]; then
    echo "[skip-existing] transfer ${run_name} ${split}"
    return 0
  fi
  time_command "transfer" "$run_name" "$split" \
    "$PYTHON_BIN" -m qsar_tl.training.train \
      "${COMMON_ARGS[@]}" \
      --run-name-zh "$run_name" \
      --split-name "$split" \
      --epochs "$epochs" \
      --finetune-epochs "$finetune_epochs" \
      --learning-rate 0.0005 \
      --dropout 0.10 \
      --weight-decay 0.000009856751793848817 \
      --toxicity-binning-loss-weight "$toxicity_loss" \
      --censored-loss-weight "$censored_loss"
}

run_smoke() {
  run_transfer \
    "smoke_transfer_f20_cebin_lw005_censored_w003" \
    "M_v2_aquatic_to_soil_ptox_adapt_C_f20" \
    "0.05" \
    "0.03" \
    "1" \
    "1"
}

run_matrix() {
  for censored_weight in 0.01 0.03 0.10; do
    label="${censored_weight/./p}"
    run_transfer \
      "transfer_f20_cebin_lw005_censored_w${label}" \
      "M_v2_aquatic_to_soil_ptox_adapt_C_f20" \
      "0.05" \
      "$censored_weight"
    run_transfer \
      "transfer_f100_cebin_lw0025_censored_w${label}" \
      "M_v2_aquatic_to_soil_ptox_adapt_C_f100" \
      "0.025" \
      "$censored_weight"
  done
}

summarize() {
  "$PYTHON_BIN" scripts/summarize_deep_runs.py \
    --root "$OUT_ROOT" \
    --out-dir "$SUMMARY_OUT" \
    --run-times "$RUN_TIMES"
  "$PYTHON_BIN" scripts/summarize_ad_gate.py \
    --audit-root "$V127_AUDIT_ROOT" \
    --out "$AD_GATE_OUT" \
    --split-part test
}

echo "[start] $(date -Is) v1.2.8 censored-loss matrix mode=${MODE}"
case "$MODE" in
  smoke)
    run_smoke
    ;;
  matrix)
    run_matrix
    ;;
  all)
    run_smoke
    run_matrix
    ;;
  summarize)
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use smoke, matrix, all, or summarize." >&2
    exit 2
    ;;
esac
if ! summarize; then
  echo "[summary-failed] $(date -Is) v1.2.8 censored-loss matrix mode=${MODE}" >&2
  exit 1
fi
echo "[done] $(date -Is) v1.2.8 censored-loss matrix mode=${MODE}"
