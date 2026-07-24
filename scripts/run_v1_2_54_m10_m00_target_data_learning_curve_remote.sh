#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-auto}"
case "$MODE" in
  build|smoke|formal|auto) ;;
  *) echo "Usage: $0 [build|smoke|formal|auto]" >&2; exit 2 ;;
esac

PYTHON="${PYTHON:-/opt/anaconda3/envs/qsar-ph3/bin/python}"
CONFIG="${CONFIG:-configs/experiment.remote.easyai.yaml}"
DB="${DB:-outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite}"
SOURCE_TABLE="aggregated_task_records_ptox_soil_mass_molar_qc"
MATRIX_ROOT="outputs/experiments/v1_2_54_m10_m00_target_data_learning_curve"
SMOKE_ROOT="${MATRIX_ROOT}_smoke"
LEGACY_ROOT="outputs/experiments/第二层核心因果实验矩阵_v1_2_44"
AUDIT_DIR="outputs/audits/v1_2_54_m10_m00_target_data_learning_curve"
SPLIT_AUDIT="$AUDIT_DIR/split_audit.json"
SUMMARY_DIR="$MATRIX_ROOT/summary"
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"
FRACTIONS=(10 25 50 75)
SEEDS=(42 2042 3407 8417)
PREDICTION_PARTS=(finetune_mgkg finetune_mgkg_validation test)

mkdir -p "$AUDIT_DIR" "$MATRIX_ROOT" "$SMOKE_ROOT" outputs/logs

exec 9>outputs/logs/v1_2_54_m10_m00_target_data_learning_curve.lock
if ! flock -n 9; then
  echo "[blocked] v1.2.54 learning-curve controller is already running" >&2
  exit 75
fi

build_splits() {
  "$PYTHON" scripts/build_v1_2_54_m10_m00_target_data_learning_curve_splits.py \
    --db "$DB" \
    --source-table "$SOURCE_TABLE" \
    --audit-json "$SPLIT_AUDIT" \
    --write
}

if [[ "$MODE" == "build" ]]; then
  build_splits
  exit 0
fi

build_splits

cell_contract() {
  local route="$1" fraction="$2" seed="$3" smoke="$4"
  CELL_SPLIT="M_v1_2_54_${route}_LC_F${fraction}"
  RUN_NAME="${route}_F${fraction}_种子${seed}"
  if [[ "$smoke" == "1" ]]; then
    RUN_ROOT="$SMOKE_ROOT"
    CELL_STAGE1_EPOCHS=0
    [[ "$route" == "M10" ]] && CELL_STAGE1_EPOCHS=1
    CELL_STAGE3_EPOCHS=1
  else
    RUN_ROOT="$MATRIX_ROOT"
    CELL_STAGE1_EPOCHS=0
    [[ "$route" == "M10" ]] && CELL_STAGE1_EPOCHS=30
    CELL_STAGE3_EPOCHS=30
  fi
  RUN_DIR="$RUN_ROOT/v1.2.54_${RUN_NAME}/deep/full/$CELL_SPLIT"
}

validate_cell() {
  local route="$1" fraction="$2" seed="$3" smoke="$4"
  local -a smoke_arg=()
  cell_contract "$route" "$fraction" "$seed" "$smoke"
  [[ "$smoke" == "1" ]] && smoke_arg=(--smoke)
  "$PYTHON" scripts/validate_v1_2_54_m10_m00_learning_curve_run.py \
    --run-dir "$RUN_DIR" \
    --split-audit "$SPLIT_AUDIT" \
    --route "$route" \
    --fraction "$fraction" \
    --seed "$seed" \
    "${smoke_arg[@]}"
}

run_cell() {
  local route="$1" fraction="$2" seed="$3" smoke="$4" log
  cell_contract "$route" "$fraction" "$seed" "$smoke"
  log="outputs/logs/v1_2_54_${route}_F${fraction}_seed${seed}"
  [[ "$smoke" == "1" ]] && log="${log}_smoke"
  log="${log}.log"

  if [[ -s "$RUN_DIR/predictions.csv" && -s "$RUN_DIR/manifest.json" && -s "$RUN_DIR/best_model.pt" ]]; then
    if validate_cell "$route" "$fraction" "$seed" "$smoke" >>"$log" 2>&1; then
      echo "[skip-valid-existing] route=$route fraction=$fraction seed=$seed smoke=$smoke"
      return 0
    fi
    echo "[invalid-existing-restart] route=$route fraction=$fraction seed=$seed smoke=$smoke" | tee -a "$log"
  elif [[ -d "$RUN_DIR" ]]; then
    echo "[incomplete-existing-restart] route=$route fraction=$fraction seed=$seed smoke=$smoke" | tee -a "$log"
  fi

  echo "[cell_start] route=$route fraction=$fraction seed=$seed smoke=$smoke time=$(date -Is)" | tee -a "$log"
  "$PYTHON" -m qsar_tl.training.train \
    --config "$CONFIG" \
    --db "$DB" \
    --out-dir "$RUN_ROOT" \
    --run-version v1.2.54 \
    --run-name-zh "$RUN_NAME" \
    --source-table "$SOURCE_TABLE" \
    --split-name "$CELL_SPLIT" \
    --task-filter-min-total 0 \
    --task-filter-min-train 0 \
    --task-filter-min-eval 0 \
    --seed "$seed" \
    --epochs "$CELL_STAGE1_EPOCHS" \
    --finetune-epochs 0 \
    --finetune-mgkg-epochs "$CELL_STAGE3_EPOCHS" \
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
    --no-finetune-mgkg-target-bin-sampling \
    --no-finetune-mgkg-hierarchical-head \
    --no-mgkg-residual-adapter \
    --no-swa \
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
    --finetune-mgkg-monitor-split valid \
    --finetune-mgkg-validation-fraction 0 \
    --finetune-mgkg-validation-seed 42 \
    --source-weighting-method none \
    --source-weighting-alpha 1 \
    --source-weight-cache-dir outputs/cache/source_weights \
    --toxicity-binning \
    --toxicity-binning-mode aux_classification \
    --toxicity-binning-scheme authority_v1 \
    --toxicity-binning-loss-weight 0.025 \
    --no-censored-loss \
    --no-effect-level-weighting \
    --metric-min-n 5 \
    --prediction-split-parts "${PREDICTION_PARTS[@]}" \
    --device cuda:0 \
    --ablation full >>"$log" 2>&1

  validate_cell "$route" "$fraction" "$seed" "$smoke" >>"$log" 2>&1
  echo "[cell_done] route=$route fraction=$fraction seed=$seed smoke=$smoke time=$(date -Is)" | tee -a "$log"
}

run_batch() {
  local route="$1" fraction="$2" smoke="$3"
  shift 3
  local seeds=("$@") active=() failed=0 seed pid
  for seed in "${seeds[@]}"; do
    run_cell "$route" "$fraction" "$seed" "$smoke" &
    pid=$!
    active+=("$pid")
    if (( ${#active[@]} >= PARALLEL_JOBS )); then
      if ! wait "${active[0]}"; then failed=1; fi
      active=("${active[@]:1}")
    fi
  done
  for pid in "${active[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if (( failed != 0 )); then
    echo "[batch_failed] route=$route fraction=$fraction smoke=$smoke time=$(date -Is)" >&2
    return 1
  fi
}

run_smoke() {
  run_batch M10 10 1 42
  run_batch M00 10 1 42
  echo "[smoke_complete] time=$(date -Is) root=$SMOKE_ROOT"
}

run_formal() {
  local fraction
  for fraction in "${FRACTIONS[@]}"; do
    run_batch M10 "$fraction" 0 "${SEEDS[@]}"
    run_batch M00 "$fraction" 0 "${SEEDS[@]}"
    echo "[fraction_complete] fraction=$fraction time=$(date -Is)"
  done
  "$PYTHON" scripts/summarize_v1_2_54_m10_m00_target_data_learning_curve.py \
    --matrix-root "$MATRIX_ROOT" \
    --legacy-root "$LEGACY_ROOT" \
    --output-dir "$SUMMARY_DIR" \
    --bootstrap-replicates 20000 \
    --bootstrap-seed 20260724
  echo "[matrix_complete] time=$(date -Is) root=$MATRIX_ROOT summary=$SUMMARY_DIR/summary.json"
}

case "$MODE" in
  smoke) run_smoke ;;
  formal) run_formal ;;
  auto)
    run_smoke
    run_formal
    ;;
esac
