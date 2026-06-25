#!/usr/bin/env bash
set -u -o pipefail

MODE="${1:-all}"

cd /home/easyai/DL1/ecotox_qsar_transfer || exit 1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PYTHON_BIN="/opt/anaconda3/bin/python"
CONFIG="configs/experiment.remote.easyai.yaml"
DB="outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite"
TRANSFER_TABLE="aggregated_task_records_aquatic_soil_ptox_qc"
SPLIT="M_v2_aquatic_to_soil_ptox_adapt_C_f100"
OUT_ROOT="outputs/experiments/v1_2_13_anchor_validation_policy_remote"
SUMMARY_OUT="outputs/experiments/v1_2_13_anchor_validation_policy_remote_summary"
AUDIT_ROOT="${OUT_ROOT}/audits"
AD_GATE_OUT="${SUMMARY_OUT}/ad_gate_summary.csv"
ENSEMBLE_OUT="${SUMMARY_OUT}/seed_mean_ensemble"
REFERENCE_SUMMARY="outputs/experiments/v1_2_12_f100_5seed_confirmation_remote_summary"
LOG_DIR="outputs/logs"
RUN_TIMES="${LOG_DIR}/run_v1_2_13_anchor_validation_policy_times.csv"

F100_CENSORED_WEIGHT="${F100_CENSORED_WEIGHT:-0.01}"
FINETUNE_VALIDATION_FRACTION="${FINETUNE_VALIDATION_FRACTION:-0.0}"
SEEDS="${SEEDS:-42 1042 2042 3042 4042}"

mkdir -p "$OUT_ROOT" "$SUMMARY_OUT" "$AUDIT_ROOT" "$ENSEMBLE_OUT" "$LOG_DIR"
if [[ ! -s "$RUN_TIMES" ]]; then
  echo "kind,run_name,split_name,start_iso,end_iso,duration_seconds,exit_code" > "$RUN_TIMES"
fi

weight_label() {
  local value="$1"
  echo "${value/./p}"
}

fraction_label() {
  local value="$1"
  if [[ "$value" == "0" || "$value" == "0.0" || "$value" == "0.00" ]]; then
    echo "val0"
  else
    echo "val${value/./p}"
  fi
}

VAL_LABEL="$(fraction_label "$FINETUNE_VALIDATION_FRACTION")"

COMMON_ARGS=(
  --config "$CONFIG"
  --db "$DB"
  --out-dir "$OUT_ROOT"
  --run-version "v1.2.13"
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
  --finetune-validation-fraction "$FINETUNE_VALIDATION_FRACTION"
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

anchor_run_name() {
  local seed="$1"
  local cw
  cw="$(weight_label "$F100_CENSORED_WEIGHT")"
  echo "transfer_f100_anchor_tanimoto_a1_seed${seed}_cebin_lw0025_censored_w${cw}_${VAL_LABEL}"
}

run_transfer() {
  local run_name="$1"
  local seed="$2"
  local epochs="${3:-30}"
  local finetune_epochs="${4:-60}"
  local out_dir="${OUT_ROOT}/v1.2.13_${run_name}/deep/full/${SPLIT}"
  if [[ -s "${out_dir}/predictions.csv" && -s "${out_dir}/manifest.json" && -s "${out_dir}/history.csv" ]]; then
    echo "[skip-existing] transfer ${run_name} ${SPLIT}"
    return 0
  fi
  time_command "transfer" "$run_name" "$SPLIT" \
    "$PYTHON_BIN" -m qsar_tl.training.train \
      "${COMMON_ARGS[@]}" \
      --seed "$seed" \
      --run-name-zh "$run_name" \
      --split-name "$SPLIT" \
      --epochs "$epochs" \
      --finetune-epochs "$finetune_epochs" \
      --learning-rate 0.0005 \
      --dropout 0.10 \
      --weight-decay 0.000009856751793848817 \
      --source-weighting-method tanimoto_to_finetune \
      --source-weighting-alpha 1.0
}

prediction_path() {
  local run_name="$1"
  echo "${OUT_ROOT}/v1.2.13_${run_name}/deep/full/${SPLIT}/predictions.csv"
}

run_ad_audit() {
  local run_name="$1"
  local pred_path
  pred_path="$(prediction_path "$run_name")"
  local out_dir="${AUDIT_ROOT}/${run_name}"
  if [[ -s "${out_dir}/ad_stratified_metrics.csv" && -s "${out_dir}/ad_audit_manifest.json" ]]; then
    echo "[skip-existing] audit ${run_name}"
    return 0
  fi
  if [[ ! -s "$pred_path" ]]; then
    echo "[skip-missing] audit ${run_name} predictions not found: ${pred_path}" >&2
    return 1
  fi
  time_command "audit" "$run_name" "$SPLIT" \
    "$PYTHON_BIN" scripts/audit_prediction_application_domain.py \
      --db "$DB" \
      --source-table "$TRANSFER_TABLE" \
      --split-name "$SPLIT" \
      --predictions "$pred_path" \
      --out-dir "$out_dir" \
      --molecular-cache "outputs/features/molecular_features_rdkit_morgan512.jsonl" \
      --fingerprint-size 512 \
      --pca-components 0 \
      --tanimoto-threshold 0.5 \
      --taxon-similarity-threshold 0.8
}

run_matrix() {
  local seed run_name
  for seed in $SEEDS; do
    run_name="$(anchor_run_name "$seed")"
    run_transfer "$run_name" "$seed"
  done
}

run_audits() {
  local seed run_name
  for seed in $SEEDS; do
    run_name="$(anchor_run_name "$seed")"
    run_ad_audit "$run_name"
  done
}

run_smoke() {
  local seed="${SMOKE_SEED:-42}"
  local run_name
  run_name="smoke_$(anchor_run_name "$seed")"
  run_transfer "$run_name" "$seed" 1 1
}

ensemble_group_members() {
  local members=()
  local seed
  for seed in $SEEDS; do
    members+=("$(anchor_run_name "$seed")")
  done
  local joined=""
  local member
  for member in "${members[@]}"; do
    if [[ -z "$joined" ]]; then
      joined="$member"
    else
      joined="${joined},${member}"
    fi
  done
  echo "$joined"
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
  "$PYTHON_BIN" scripts/summarize_seed_ensembles.py \
    --audit-root "$AUDIT_ROOT" \
    --out-dir "$ENSEMBLE_OUT" \
    --group "anchor_tanimoto_a1_${VAL_LABEL}_5seed_ensemble=$(ensemble_group_members)" \
    --split-part test \
    --write-prediction-rows
  write_validation_policy_comparison
}

summarize_deep_only() {
  "$PYTHON_BIN" scripts/summarize_deep_runs.py \
    --root "$OUT_ROOT" \
    --out-dir "$SUMMARY_OUT" \
    --run-times "$RUN_TIMES"
}

write_validation_policy_comparison() {
  "$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path

summary_out = Path("outputs/experiments/v1_2_13_anchor_validation_policy_remote_summary")
reference = Path("outputs/experiments/v1_2_12_f100_5seed_confirmation_remote_summary/seed_mean_ensemble_focus_summary.csv")
current = summary_out / "seed_mean_ensemble" / "seed_mean_ensemble_focus_summary.csv"
out = summary_out / "validation_policy_comparison.csv"

rows = []
if reference.exists():
    with reference.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("run") == "anchor_tanimoto_a1_5seed_ensemble":
                rows.append({"policy": "val0p2_control_existing_v1_2_12", **row})
if current.exists():
    with current.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows.append({"policy": "val0_new_v1_2_13", **row})

fieldnames = []
for row in rows:
    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames or ["policy"])
    writer.writeheader()
    writer.writerows(rows)
print({"comparison": str(out), "rows": len(rows)})
PY
}

echo "[start] $(date -Is) v1.2.13 anchor validation policy mode=${MODE} seeds=${SEEDS} finetune_validation_fraction=${FINETUNE_VALIDATION_FRACTION}"
case "$MODE" in
  smoke)
    run_smoke
    ;;
  matrix)
    run_matrix
    ;;
  audits)
    run_audits
    ;;
  summarize)
    ;;
  all)
    run_matrix
    run_audits
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use smoke, matrix, audits, summarize, or all." >&2
    exit 2
    ;;
esac
if [[ "$MODE" == "smoke" || "$MODE" == "matrix" ]]; then
  if ! summarize_deep_only; then
    echo "[summary-failed] $(date -Is) v1.2.13 anchor validation policy mode=${MODE}" >&2
    exit 1
  fi
elif ! summarize; then
  echo "[summary-failed] $(date -Is) v1.2.13 anchor validation policy mode=${MODE}" >&2
  exit 1
fi
echo "[done] $(date -Is) v1.2.13 anchor validation policy mode=${MODE}"
