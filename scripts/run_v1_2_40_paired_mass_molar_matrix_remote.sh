#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-smoke}"
case "$MODE" in
  build|smoke|formal) ;;
  *) echo "Usage: $0 [build|smoke|formal]" >&2; exit 2 ;;
esac

PYTHON="${PYTHON:-/opt/anaconda3/envs/qsar-ph3/bin/python}"
CONFIG="${CONFIG:-configs/experiment.remote.easyai.yaml}"
DB="${DB:-outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite}"
SOURCE_TABLE="aggregated_task_records_ptox_soil_mass_molar_qc"
MASS_DIRECT_SPLIT="SoilMgkgMWMatched_B_random_8_2"
MOLAR_DIRECT_SPLIT="SoilMolkgMWMatched_B_random_8_2"
MASS_TRANSFER_SPLIT="M_v1_2_40_ptox_to_soil_mgkg_mw_matched_B_random_8_2"
MOLAR_TRANSFER_SPLIT="M_v1_2_40_ptox_to_soil_molkg_B_random_8_2"
AUDIT_DIR="outputs/audits/v1_2_40_paired_mass_molar"
SUMMARY_DIR="${SUMMARY_DIR:-outputs/experiments/v1_2_40_paired_mass_molar_matrix_summary}"
PARALLEL_JOBS="${PARALLEL_JOBS:-3}"
SEEDS=(42 2042 3407 8417)

mkdir -p "$AUDIT_DIR" "$SUMMARY_DIR" outputs/logs

build_data_and_splits() {
  "$PYTHON" scripts/build_paired_soil_molkg_experiment.py \
    --db "$DB" \
    --output-table "$SOURCE_TABLE" \
    --matched-mgkg-split "$MASS_DIRECT_SPLIT" \
    --molkg-split "$MOLAR_DIRECT_SPLIT" \
    --audit-csv "$AUDIT_DIR/conversion_audit.csv" \
    --summary-json "$AUDIT_DIR/conversion_summary.json"

  "$PYTHON" scripts/build_three_stage_ptox_to_soil_mgkg_split.py \
    --db "$DB" \
    --source-table "$SOURCE_TABLE" \
    --soil-ptox-source-table aggregated_task_records_soil_ptox_qc \
    --soil-ptox-split-name SoilPtoxQC2_B_random_8_2 \
    --soil-mgkg-source-table "$SOURCE_TABLE" \
    --soil-mgkg-split-name "$MASS_DIRECT_SPLIT" \
    --stage3-target-name neg_log10_mg_kg \
    --stage3-target-family solid_neglog_mg_kg \
    --stage3-label soil_mgkg_mw_matched \
    --split-name "$MASS_TRANSFER_SPLIT" \
    --audit-csv "$AUDIT_DIR/${MASS_TRANSFER_SPLIT}_routing_audit.csv" \
    --seed 42

  "$PYTHON" scripts/build_three_stage_ptox_to_soil_mgkg_split.py \
    --db "$DB" \
    --source-table "$SOURCE_TABLE" \
    --soil-ptox-source-table aggregated_task_records_soil_ptox_qc \
    --soil-ptox-split-name SoilPtoxQC2_B_random_8_2 \
    --soil-mgkg-source-table "$SOURCE_TABLE" \
    --soil-mgkg-split-name "$MOLAR_DIRECT_SPLIT" \
    --stage3-target-name neg_log10_mol_kg \
    --stage3-target-family solid_neglog_mol_kg \
    --stage3-label soil_molkg \
    --split-name "$MOLAR_TRANSFER_SPLIT" \
    --audit-csv "$AUDIT_DIR/${MOLAR_TRANSFER_SPLIT}_routing_audit.csv" \
    --seed 42
}

if [[ "$MODE" == "build" ]]; then
  build_data_and_splits
  exit 0
fi

# Serialize with the existing v1.2.39 controller on the single GPU. This call
# waits for that matrix to release its lock, then keeps the GPU reserved for
# this paired matrix.
exec 8>outputs/logs/v1_2_40_paired_mass_molar_matrix.lock
if ! flock -n 8; then
  echo "[blocked] paired mass/molar matrix is already queued or running" >&2
  exit 75
fi
exec 9>outputs/logs/v1_2_39_transfer_optimization_matrix.lock
flock 9

build_data_and_splits

if [[ "$MODE" == "smoke" ]]; then
  EPOCHS=1
  STAGE2_EPOCHS=1
  OUT_ROOT="${OUT_ROOT:-outputs/experiments/v1_2_40_paired_mass_molar_matrix_remote_smoke}"
  RUN_SEEDS=(42)
else
  EPOCHS=30
  STAGE2_EPOCHS=20
  OUT_ROOT="${OUT_ROOT:-outputs/experiments/v1_2_40_paired_mass_molar_matrix_remote}"
  RUN_SEEDS=("${SEEDS[@]}")
fi
mkdir -p "$OUT_ROOT"

run_direct() {
  local cell="$1" seed="$2" split="$3"
  local run_name="${cell}_seed${seed}"
  local run_dir="$OUT_ROOT/v1.2.40_${run_name}/deep/full/$split"
  if [[ -s "$run_dir/predictions.csv" && -s "$run_dir/manifest.json" && -s "$run_dir/best_model.pt" ]]; then
    echo "[skip-existing] cell=$cell seed=$seed run_dir=$run_dir"
    return 0
  fi
  "$PYTHON" -m qsar_tl.training.train \
    --config "$CONFIG" \
    --db "$DB" \
    --out-dir "$OUT_ROOT" \
    --run-version v1.2.40 \
    --run-name-zh "$run_name" \
    --source-table "$SOURCE_TABLE" \
    --split-name "$split" \
    --seed "$seed" \
    --epochs "$EPOCHS" \
    --finetune-epochs 0 \
    --finetune-mgkg-epochs 0 \
    --batch-size 512 \
    --scheduler cosine \
    --learning-rate 0.0005 \
    --weight-decay 0.00001 \
    --dropout 0.10 \
    --target-standardization per_task_target \
    --head-routing task_target \
    --no-medium-adapters \
    --early-stopping \
    --early-stopping-patience 15 \
    --early-stopping-min-delta 0 \
    --monitor-split internal_train_fraction \
    --validation-fraction 0.2 \
    --validation-seed 17073 \
    --source-weighting-method none \
    --no-toxicity-binning \
    --no-censored-loss \
    --no-effect-level-weighting \
    --metric-min-n 5 \
    --device cuda:0 \
    --ablation full
}

run_transfer() {
  local cell="$1" seed="$2" split="$3"
  local run_name="${cell}_seed${seed}"
  local run_dir="$OUT_ROOT/v1.2.40_${run_name}/deep/full/$split"
  if [[ -s "$run_dir/predictions.csv" && -s "$run_dir/manifest.json" && -s "$run_dir/best_model.pt" ]]; then
    echo "[skip-existing] cell=$cell seed=$seed run_dir=$run_dir"
    return 0
  fi
  "$PYTHON" -m qsar_tl.training.train \
    --config "$CONFIG" \
    --db "$DB" \
    --out-dir "$OUT_ROOT" \
    --run-version v1.2.40 \
    --run-name-zh "$run_name" \
    --source-table "$SOURCE_TABLE" \
    --split-name "$split" \
    --seed "$seed" \
    --epochs "$EPOCHS" \
    --finetune-epochs "$STAGE2_EPOCHS" \
    --finetune-mgkg-epochs "$EPOCHS" \
    --batch-size 512 \
    --finetune-batch-size 512 \
    --finetune-mgkg-batch-size 512 \
    --scheduler cosine \
    --finetune-scheduler cosine \
    --finetune-mgkg-scheduler cosine \
    --learning-rate 0.0005 \
    --finetune-learning-rate 0.0001 \
    --finetune-mgkg-learning-rate 0.0005 \
    --finetune-mgkg-trunk-learning-rate 0 \
    --finetune-freeze none \
    --finetune-mgkg-freeze none \
    --finetune-mgkg-head-only-epochs 0 \
    --finetune-mgkg-replay-fraction 0 \
    --finetune-mgkg-toxicity-bin-loss-weight 0 \
    --no-mgkg-residual-adapter \
    --weight-decay 0.00001 \
    --dropout 0.10 \
    --target-standardization per_task_target \
    --head-routing task_target \
    --allow-mixed-target-dimensions \
    --no-medium-adapters \
    --early-stopping \
    --early-stopping-patience 15 \
    --early-stopping-min-delta 0 \
    --monitor-split internal_train_fraction \
    --validation-fraction 0.1 \
    --validation-seed 42 \
    --finetune-validation-fraction 0.2 \
    --finetune-validation-seed 42 \
    --finetune-mgkg-validation-fraction 0.2 \
    --finetune-mgkg-validation-seed 42 \
    --source-weighting-method tanimoto_to_finetune \
    --source-weighting-alpha 1 \
    --source-weight-cache-dir outputs/cache/source_weights \
    --toxicity-binning \
    --toxicity-binning-mode aux_classification \
    --toxicity-binning-scheme authority_v1 \
    --toxicity-binning-loss-weight 0.025 \
    --no-censored-loss \
    --no-effect-level-weighting \
    --metric-min-n 5 \
    --device cuda:0 \
    --ablation full
}

run_cell() {
  local cell="$1" seed="$2" split log
  case "$cell" in
    D_mass)  split="$MASS_DIRECT_SPLIT" ;;
    D_molar) split="$MOLAR_DIRECT_SPLIT" ;;
    X0_mass) split="$MASS_TRANSFER_SPLIT" ;;
    X0_molar) split="$MOLAR_TRANSFER_SPLIT" ;;
    *) echo "Unknown paired matrix cell: $cell" >&2; return 2 ;;
  esac
  log="outputs/logs/paired_mass_molar_${cell}_seed${seed}.log"
  echo "[matrix_start] cell=$cell seed=$seed time=$(date -Is)"
  if [[ "$cell" == D_* ]]; then
    run_direct "$cell" "$seed" "$split" >"$log" 2>&1
  else
    run_transfer "$cell" "$seed" "$split" >"$log" 2>&1
  fi
  echo "[matrix_done] cell=$cell seed=$seed time=$(date -Is)"
}

run_jobs_parallel() {
  local jobs=("$@") active_pids=() failed=0 job cell seed pid
  for job in "${jobs[@]}"; do
    cell="${job%%:*}"
    seed="${job##*:}"
    run_cell "$cell" "$seed" &
    pid=$!
    active_pids+=("$pid")
    if (( ${#active_pids[@]} >= PARALLEL_JOBS )); then
      if ! wait "${active_pids[0]}"; then failed=1; fi
      active_pids=("${active_pids[@]:1}")
    fi
  done
  for pid in "${active_pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if (( failed != 0 )); then
    echo "[matrix_failed] time=$(date -Is)" >&2
    return 1
  fi
}

JOBS=()
for seed in "${RUN_SEEDS[@]}"; do
  for cell in D_mass D_molar X0_mass X0_molar; do
    JOBS+=("${cell}:${seed}")
  done
done
run_jobs_parallel "${JOBS[@]}"

"$PYTHON" scripts/summarize_v1_2_40_paired_mass_molar.py \
  --root "$OUT_ROOT" \
  --db "$DB" \
  --source-table "$SOURCE_TABLE" \
  --output-dir "$SUMMARY_DIR"

echo "[matrix_complete] time=$(date -Is) summary=$SUMMARY_DIR/paired_seed_metrics.csv"
