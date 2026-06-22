#!/usr/bin/env bash
set -u -o pipefail

MODE="${1:-smoke}"

cd /home/easyai/DL1/ecotox_qsar_transfer || exit 1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PYTHON_BIN="/opt/anaconda3/bin/python"
CONFIG="configs/experiment.remote.easyai.yaml"
DB="outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite"
SOIL_TABLE="aggregated_task_records_soil_ptox_qc"
TRANSFER_TABLE="aggregated_task_records_aquatic_soil_ptox_qc"
OUT_ROOT="outputs/experiments/v1_2_6_authority_binning_matrix_remote"
SUMMARY_OUT="outputs/experiments/v1_2_6_authority_binning_matrix_remote_summary"
LOG_DIR="outputs/logs"
RUN_TIMES="${LOG_DIR}/run_v1_2_6_authority_binning_matrix_times.csv"

mkdir -p "$OUT_ROOT" "$SUMMARY_OUT" "$LOG_DIR"
if [[ ! -s "$RUN_TIMES" ]]; then
  echo "kind,run_name,split_name,start_iso,end_iso,duration_seconds,exit_code" > "$RUN_TIMES"
fi

COMMON_ARGS=(
  --config "$CONFIG"
  --db "$DB"
  --out-dir "$OUT_ROOT"
  --run-version "v1.2.6"
  --ablation full
  --batch-size 512
  --scheduler cosine
  --target-standardization per_task_target
  --device cuda:0
  --metric-min-n 5
  --early-stopping
  --early-stopping-patience 15
  --early-stopping-min-delta 0.0
  --toxicity-binning
  --toxicity-binning-mode aux_classification
  --toxicity-binning-scheme authority_v1
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
  local toxicity_loss="$9"
  shift 9
  local out_dir="${OUT_ROOT}/v1.2.6_${run_name}/deep/full/${split}"
  if [[ -s "${out_dir}/predictions.csv" && -s "${out_dir}/manifest.json" && -s "${out_dir}/toxicity_bin_metrics.csv" ]]; then
    echo "[skip-existing] ${kind} ${run_name} ${split}"
    return 0
  fi
  time_command "$kind" "$run_name" "$split" \
    "$PYTHON_BIN" -m qsar_tl.training.train \
      "${COMMON_ARGS[@]}" \
      --run-name-zh "$run_name" \
      --source-table "$source_table" \
      --split-name "$split" \
      --epochs "$epochs" \
      --learning-rate "$lr" \
      --dropout "$dropout" \
      --weight-decay "$wd" \
      --toxicity-binning-loss-weight "$toxicity_loss" \
      "$@"
}

run_soil() {
  run_train "soil" "$1" "$SOIL_TABLE" "$2" "$3" "$4" "$5" "$6" "$7" "${@:8}"
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
  local toxicity_loss="$9"
  shift 9
  run_train "transfer" "$run_name" "$TRANSFER_TABLE" "$split" "$epochs" "$lr" "$dropout" "$wd" "$toxicity_loss" \
    --finetune-epochs "$finetune_epochs" \
    --finetune-learning-rate "$finetune_lr" \
    --finetune-scheduler reduce_on_plateau \
    --finetune-validation-fraction 0.2 \
    --source-weighting-method tanimoto_to_finetune \
    --source-weighting-alpha 1.0 \
    "$@"
}

summarize() {
  "$PYTHON_BIN" scripts/summarize_deep_runs.py \
    --root "$OUT_ROOT" \
    --out-dir "$SUMMARY_OUT" \
    --run-times "$RUN_TIMES"
}

run_smoke() {
  run_soil "smoke_soil_f100_authority_bin_aux_lw005" "SoilPtoxQC2_C_low_f100" 1 0.0005 0.10 0.000009856751793848817 0.05 \
    --no-effect-level-weighting
  run_transfer "smoke_transfer_f20_source_alpha1_authority_bin_aux_lw005" "M_v2_aquatic_to_soil_ptox_adapt_C_f20" 1 1 0.0005 0.10 0.000009856751793848817 0.0003082636455810776 0.05 \
    --no-effect-level-weighting
}

run_core() {
  # Soil-only: best known split-specific hyperparameter anchors from v1.2.5, with authority-bin auxiliary loss.
  run_soil "soil_f20_authority_bin_aux_lw005_augN001" "SoilPtoxQC2_C_low_f20" 50 0.0005 0.10 0.000009856751793848817 0.05 \
    --no-effect-level-weighting --augment-train-replicates 2 --augment-numeric-noise-std 0.01 --augment-target-noise-std 0.0
  run_soil "soil_f100_authority_bin_aux_lw005_eff050" "SoilPtoxQC2_C_low_f100" 50 0.0005 0.10 0.000009856751793848817 0.05 \
    --effect-level-weighting --effect-level-weighting-beta 0.50
  run_soil "soil_fullC_authority_bin_aux_lw005_core" "SoilPtoxQC2_C_chemical_holdout_8_2" 50 0.0003 0.20 0.000009856751793848817 0.05 \
    --no-effect-level-weighting

  # Transfer: source Tanimoto alpha=1.0 remains the stable baseline; compare low-soil fractions.
  run_transfer "transfer_f20_source_alpha1_authority_bin_aux_lw005" "M_v2_aquatic_to_soil_ptox_adapt_C_f20" 30 60 0.0005 0.10 0.000009856751793848817 0.0003082636455810776 0.05 \
    --no-effect-level-weighting
  run_transfer "transfer_f50_source_alpha1_authority_bin_aux_lw005" "M_v2_aquatic_to_soil_ptox_adapt_C_f50" 30 60 0.0005 0.10 0.000009856751793848817 0.0003082636455810776 0.05 \
    --no-effect-level-weighting
  run_transfer "transfer_f100_source_alpha1_authority_bin_aux_lw005" "M_v2_aquatic_to_soil_ptox_adapt_C_f100" 30 60 0.0005 0.10 0.000009856751793848817 0.0003082636455810776 0.05 \
    --no-effect-level-weighting
  run_transfer "transfer_f20_source_alpha1_effect025_authority_bin_aux_lw005" "M_v2_aquatic_to_soil_ptox_adapt_C_f20" 30 60 0.0005 0.10 0.000009856751793848817 0.0003082636455810776 0.05 \
    --effect-level-weighting --effect-level-weighting-beta 0.25
}

run_hpo() {
  # Authority-bin loss-weight sensitivity around best known anchors.
  for loss_label in lw0025 lw010; do
    case "$loss_label" in
      lw0025) tox_loss="0.025" ;;
      lw010) tox_loss="0.10" ;;
    esac
    run_soil "soil_f100_authority_bin_aux_${loss_label}_eff050" "SoilPtoxQC2_C_low_f100" 50 0.0005 0.10 0.000009856751793848817 "$tox_loss" \
      --effect-level-weighting --effect-level-weighting-beta 0.50
    run_soil "soil_fullC_authority_bin_aux_${loss_label}_core" "SoilPtoxQC2_C_chemical_holdout_8_2" 50 0.0003 0.20 0.000009856751793848817 "$tox_loss" \
      --no-effect-level-weighting
    run_transfer "transfer_f20_source_alpha1_authority_bin_aux_${loss_label}" "M_v2_aquatic_to_soil_ptox_adapt_C_f20" 30 60 0.0005 0.10 0.000009856751793848817 0.0003082636455810776 "$tox_loss" \
      --no-effect-level-weighting
    run_transfer "transfer_f100_source_alpha1_authority_bin_aux_${loss_label}" "M_v2_aquatic_to_soil_ptox_adapt_C_f100" 30 60 0.0005 0.10 0.000009856751793848817 0.0003082636455810776 "$tox_loss" \
      --no-effect-level-weighting
  done

  # Joint best-prior strategy checks.
  run_transfer "transfer_f20_source_alpha1_effect025_authority_bin_aux_lw0025" "M_v2_aquatic_to_soil_ptox_adapt_C_f20" 30 60 0.0005 0.10 0.000009856751793848817 0.0003082636455810776 0.025 \
    --effect-level-weighting --effect-level-weighting-beta 0.25
  run_transfer "transfer_f20_source_alpha1_effect025_authority_bin_aux_lw010" "M_v2_aquatic_to_soil_ptox_adapt_C_f20" 30 60 0.0005 0.10 0.000009856751793848817 0.0003082636455810776 0.10 \
    --effect-level-weighting --effect-level-weighting-beta 0.25
}

echo "[start] $(date -Is) v1.2.6 authority-binning matrix mode=${MODE}"
case "$MODE" in
  smoke)
    run_smoke
    ;;
  core)
    run_core
    ;;
  hpo)
    run_hpo
    ;;
  all)
    run_smoke
    run_core
    run_hpo
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use smoke, core, hpo, or all." >&2
    exit 2
    ;;
esac
if ! summarize; then
  echo "[summary-failed] $(date -Is) v1.2.6 authority-binning matrix mode=${MODE}" >&2
  exit 1
fi
echo "[done] $(date -Is) v1.2.6 authority-binning matrix mode=${MODE}"
