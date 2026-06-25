#!/usr/bin/env bash
set -u -o pipefail

MODE="${1:-first_batch}"

cd /home/easyai/DL1/ecotox_qsar_transfer || exit 1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PYTHON_BIN="/opt/anaconda3/bin/python"
CONFIG="configs/experiment.remote.easyai.yaml"
DB="outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite"
SOIL_TABLE="aggregated_task_records_soil_ptox_qc"
TRANSFER_TABLE="aggregated_task_records_aquatic_soil_ptox_qc"
OUT_ROOT="outputs/experiments/v1_2_7_censored_ordinal_ad_first_batch_remote"
SUMMARY_OUT="outputs/experiments/v1_2_7_censored_ordinal_ad_first_batch_remote_summary"
AUDIT_ROOT="${OUT_ROOT}/audits"
V126_ROOT="outputs/experiments/v1_2_6_authority_binning_matrix_remote"
LOG_DIR="outputs/logs"
RUN_TIMES="${LOG_DIR}/run_v1_2_7_censored_ordinal_ad_first_batch_times.csv"

mkdir -p "$OUT_ROOT" "$SUMMARY_OUT" "$AUDIT_ROOT" "$LOG_DIR"
if [[ ! -s "$RUN_TIMES" ]]; then
  echo "kind,run_name,split_name,start_iso,end_iso,duration_seconds,exit_code" > "$RUN_TIMES"
fi

COMMON_TRAIN_ARGS=(
  --config "$CONFIG"
  --db "$DB"
  --out-dir "$OUT_ROOT"
  --run-version "v1.2.7"
  --ablation full
  --batch-size 512
  --scheduler cosine
  --target-standardization per_task_target
  --device cuda:0
  --metric-min-n 5
  --early-stopping
  --early-stopping-patience 15
  --early-stopping-min-delta 0.0
  --no-effect-level-weighting
  --domain-alignment-method none
  --domain-alignment-weight 0.0
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

run_train() {
  local kind="$1"
  local run_name="$2"
  local source_table="$3"
  local split="$4"
  local epochs="$5"
  local lr="$6"
  local dropout="$7"
  local wd="$8"
  shift 8
  local out_dir="${OUT_ROOT}/v1.2.7_${run_name}/deep/full/${split}"
  if [[ -s "${out_dir}/predictions.csv" && -s "${out_dir}/manifest.json" ]]; then
    echo "[skip-existing] ${kind} ${run_name} ${split}"
    return 0
  fi
  time_command "$kind" "$run_name" "$split" \
    "$PYTHON_BIN" -m qsar_tl.training.train \
      "${COMMON_TRAIN_ARGS[@]}" \
      --run-name-zh "$run_name" \
      --source-table "$source_table" \
      --split-name "$split" \
      --epochs "$epochs" \
      --learning-rate "$lr" \
      --dropout "$dropout" \
      --weight-decay "$wd" \
      "$@"
}

run_soil() {
  run_train "soil" "$1" "$SOIL_TABLE" "$2" "$3" "$4" "$5" "$6" "${@:7}"
}

run_transfer() {
  local run_name="$1"
  local split="$2"
  local epochs="$3"
  local finetune_epochs="$4"
  local lr="$5"
  local dropout="$6"
  local wd="$7"
  local finetune_lr="$8"
  shift 8
  run_train "transfer" "$run_name" "$TRANSFER_TABLE" "$split" "$epochs" "$lr" "$dropout" "$wd" \
    --finetune-epochs "$finetune_epochs" \
    --finetune-learning-rate "$finetune_lr" \
    --finetune-scheduler reduce_on_plateau \
    --finetune-validation-fraction 0.2 \
    --source-weighting-method tanimoto_to_finetune \
    --source-weighting-alpha 1.0 \
    "$@"
}

run_censored_audit() {
  local out_dir="${AUDIT_ROOT}/censored_audit_only"
  if [[ -s "${out_dir}/censored_audit_summary.csv" && -s "${out_dir}/censored_audit_manifest.json" ]]; then
    echo "[skip-existing] audit censored_audit_only"
    return 0
  fi
  time_command "audit" "censored_audit_only" "target_records" \
    "$PYTHON_BIN" scripts/audit_censored_records.py \
      --db "$DB" \
      --table target_records \
      --out-dir "$out_dir"
}

run_proxy_audit() {
  local out_dir="${AUDIT_ROOT}/proxy_audit_logp_koc_tpsa_bins"
  if [[ -s "${out_dir}/proxy_audit_summary.csv" && -s "${out_dir}/proxy_audit_manifest.json" ]]; then
    echo "[skip-existing] audit proxy_audit_logp_koc_tpsa_bins"
    return 0
  fi
  time_command "audit" "proxy_audit_logp_koc_tpsa_bins" "M_v2_aquatic_to_soil_ptox_adapt_C_f100" \
    "$PYTHON_BIN" scripts/audit_proxy_bins.py \
      --db "$DB" \
      --source-table "$TRANSFER_TABLE" \
      --split-name "M_v2_aquatic_to_soil_ptox_adapt_C_f100" \
      --out-dir "$out_dir" \
      --molecular-cache "outputs/features/molecular_features_rdkit_morgan512.jsonl" \
      --fingerprint-size 512
}

run_ad_audit() {
  local run_name="$1"
  local split="$2"
  local pred_path="$3"
  local out_dir="${AUDIT_ROOT}/${run_name}"
  if [[ -s "${out_dir}/ad_stratified_metrics.csv" && -s "${out_dir}/ad_audit_manifest.json" ]]; then
    echo "[skip-existing] audit ${run_name}"
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

summarize() {
  "$PYTHON_BIN" scripts/summarize_deep_runs.py \
    --root "$OUT_ROOT" \
    --out-dir "$SUMMARY_OUT" \
    --run-times "$RUN_TIMES"
}

run_first_batch() {
  run_transfer "transfer_f20_no_bin_rerun" "M_v2_aquatic_to_soil_ptox_adapt_C_f20" 30 60 0.0005 0.10 0.000009856751793848817 0.0003082636455810776 \
    --no-toxicity-binning
  run_transfer "transfer_f20_bin_ordinal_lw005" "M_v2_aquatic_to_soil_ptox_adapt_C_f20" 30 60 0.0005 0.10 0.000009856751793848817 0.0003082636455810776 \
    --toxicity-binning --toxicity-binning-mode ordinal --toxicity-binning-scheme authority_v1 --toxicity-binning-loss-weight 0.05
  run_transfer "transfer_f100_no_bin_rerun" "M_v2_aquatic_to_soil_ptox_adapt_C_f100" 30 60 0.0005 0.10 0.000009856751793848817 0.0003082636455810776 \
    --no-toxicity-binning
  run_transfer "transfer_f100_bin_ordinal_lw0025" "M_v2_aquatic_to_soil_ptox_adapt_C_f100" 30 60 0.0005 0.10 0.000009856751793848817 0.0003082636455810776 \
    --toxicity-binning --toxicity-binning-mode ordinal --toxicity-binning-scheme authority_v1 --toxicity-binning-loss-weight 0.025
  run_soil "soil_fullC_no_bin_anchor" "SoilPtoxQC2_C_chemical_holdout_8_2" 50 0.0003 0.20 0.000009856751793848817 \
    --no-toxicity-binning
  run_soil "soil_fullC_bin_ordinal_lw005" "SoilPtoxQC2_C_chemical_holdout_8_2" 50 0.0003 0.20 0.000009856751793848817 \
    --toxicity-binning --toxicity-binning-mode ordinal --toxicity-binning-scheme authority_v1 --toxicity-binning-loss-weight 0.05

  run_censored_audit
  run_proxy_audit

  run_ad_audit \
    "best_transfer_f100_ad_audit" \
    "M_v2_aquatic_to_soil_ptox_adapt_C_f100" \
    "${V126_ROOT}/v1.2.6_transfer_f100_source_alpha1_authority_bin_aux_lw0025/deep/full/M_v2_aquatic_to_soil_ptox_adapt_C_f100/predictions.csv"
  run_ad_audit \
    "best_transfer_f20_ad_audit" \
    "M_v2_aquatic_to_soil_ptox_adapt_C_f20" \
    "${V126_ROOT}/v1.2.6_transfer_f20_source_alpha1_authority_bin_aux_lw005/deep/full/M_v2_aquatic_to_soil_ptox_adapt_C_f20/predictions.csv"
}

echo "[start] $(date -Is) v1.2.7 censored/ordinal/ad first batch mode=${MODE}"
case "$MODE" in
  first_batch|all)
    run_first_batch
    ;;
  summarize)
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use first_batch, all, or summarize." >&2
    exit 2
    ;;
esac
if ! summarize; then
  echo "[summary-failed] $(date -Is) v1.2.7 first batch mode=${MODE}" >&2
  exit 1
fi
echo "[done] $(date -Is) v1.2.7 censored/ordinal/ad first batch mode=${MODE}"
