#!/usr/bin/env bash
set -euo pipefail

# v1.2.51: M10-Full robustness validation on five task-stratified row-random
# outer folds. This is interpolation/stability evidence, not scaffold or
# chemical-family extrapolation. One fixed training seed (3407) is used.
#
# Stage 1 is scientifically identical across folds, but the current fail-closed
# checkpoint contract seals the complete split identity. We therefore rerun
# Stage 1 per fold instead of bypassing the compatibility contract.

MODE="${1:-formal}"
case "$MODE" in formal|smoke|validate) ;; *)
  echo "Usage: $0 [formal|smoke|validate]" >&2
  exit 2
  ;;
esac

PYTHON="${PYTHON:-/opt/anaconda3/envs/qsar-ph3/bin/python}"
CONFIG="${CONFIG:-configs/experiment.remote.easyai.yaml}"
DB="${DB:-outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite}"
SOURCE_TABLE="aggregated_task_records_ptox_soil_mass_molar_qc"
SPLIT_PREFIX="M_v1_2_51_M10_随机五折_折"
AUDIT_DIR="outputs/audits/v1_2_51_m10_random_fivefold"
SPLIT_AUDIT="$AUDIT_DIR/split_contract.json"
OUT_ROOT="${OUT_ROOT:-outputs/experiments/v1_2_51_m10_random_fivefold}"
LOG_ROOT="outputs/logs"
CACHE_ROOT="${CACHE_ROOT:-outputs/cache/v1_2_51_m10_random_fivefold_source_weights}"
SEED=3407
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"
FOLDS=(1 2 3 4 5)
PREDICTION_PARTS=(finetune_mgkg finetune_mgkg_validation test)

mkdir -p "$AUDIT_DIR" "$OUT_ROOT" "$LOG_ROOT" "$CACHE_ROOT"
exec 8>"$LOG_ROOT/v1_2_51_m10_random_fivefold.lock"
if ! flock -n 8; then
  echo "[blocked] v1.2.51 M10 random-fivefold controller is already running" >&2
  exit 75
fi

builder_args=(
  --db "$DB"
  --source-table "$SOURCE_TABLE"
  --fold-prefix "$SPLIT_PREFIX"
  --audit-json "$SPLIT_AUDIT"
)
if [[ "$MODE" == "validate" ]]; then
  "$PYTHON" scripts/build_v1_2_51_m10_random_fivefold_splits.py "${builder_args[@]}"
  echo "[validate_complete] deterministic split contract passed without database write"
  exit 0
fi
"$PYTHON" scripts/build_v1_2_51_m10_random_fivefold_splits.py \
  "${builder_args[@]}" --write

if [[ "$MODE" == "smoke" ]]; then
  RUN_ROOT="${OUT_ROOT}_smoke"
  RUN_FOLDS=(1)
  STAGE1_EPOCHS=1
  STAGE3_EPOCHS=1
else
  RUN_ROOT="$OUT_ROOT"
  RUN_FOLDS=("${FOLDS[@]}")
  STAGE1_EPOCHS=30
  STAGE3_EPOCHS=30
fi
mkdir -p "$RUN_ROOT"

set_fold_contract() {
  local fold="$1"
  SPLIT_NAME="${SPLIT_PREFIX}${fold}"
  RUN_NAME="M10_Full_随机五折_折${fold}_种子${SEED}"
  RUN_DIR="$RUN_ROOT/v1.2.51_${RUN_NAME}/deep/full/$SPLIT_NAME"
}

validate_artifacts() {
  [[ -s "$RUN_DIR/predictions.csv" && -s "$RUN_DIR/manifest.json" && -s "$RUN_DIR/best_model.pt" ]]
}

run_fold() {
  local fold="$1" log
  set_fold_contract "$fold"
  log="$LOG_ROOT/v1_2_51_M10_Full_fold${fold}_seed${SEED}.log"
  if validate_artifacts; then
    echo "[skip-valid-existing] fold=$fold seed=$SEED run_dir=$RUN_DIR"
    return 0
  fi
  echo "[fold_start] fold=$fold seed=$SEED split=$SPLIT_NAME weighting=none time=$(date -Is)" | tee -a "$log"
  "$PYTHON" -m qsar_tl.training.train \
    --config "$CONFIG" --db "$DB" --out-dir "$RUN_ROOT" \
    --run-version v1.2.51 --run-name-zh "$RUN_NAME" \
    --source-table "$SOURCE_TABLE" --split-name "$SPLIT_NAME" \
    --task-filter-min-total 0 --task-filter-min-train 0 --task-filter-min-eval 0 \
    --seed "$SEED" --epochs "$STAGE1_EPOCHS" --finetune-epochs 0 \
    --finetune-mgkg-epochs "$STAGE3_EPOCHS" \
    --batch-size 512 --finetune-batch-size 512 --finetune-mgkg-batch-size 512 \
    --scheduler cosine --finetune-scheduler cosine --finetune-mgkg-scheduler cosine \
    --learning-rate 0.0005 --finetune-learning-rate 0.0001 \
    --finetune-mgkg-learning-rate 0.0005 \
    --finetune-mgkg-trunk-learning-rate 0 \
    --finetune-freeze none --finetune-mgkg-freeze none \
    --finetune-mgkg-head-only-epochs 0 --finetune-mgkg-replay-fraction 0 \
    --finetune-mgkg-toxicity-bin-loss-weight 0 \
    --no-finetune-mgkg-target-bin-sampling --no-finetune-mgkg-hierarchical-head \
    --no-mgkg-residual-adapter --no-swa --weight-decay 0.00001 --dropout 0.10 \
    --target-standardization per_task_target --head-routing task_target \
    --allow-mixed-target-dimensions --no-medium-adapters \
    --early-stopping --early-stopping-patience 15 --early-stopping-min-delta 0 \
    --monitor-split internal_train_fraction --validation-fraction 0.1 \
    --validation-seed 42 --finetune-validation-fraction 0.2 \
    --finetune-validation-seed 42 --finetune-mgkg-monitor-split valid \
    --finetune-mgkg-validation-fraction 0 --finetune-mgkg-validation-seed 42 \
    --source-weighting-method none --source-weighting-alpha 1 \
    --source-weight-cache-dir "$CACHE_ROOT" \
    --toxicity-binning --toxicity-binning-mode aux_classification \
    --toxicity-binning-scheme authority_v1 --toxicity-binning-loss-weight 0.025 \
    --no-censored-loss --no-effect-level-weighting --metric-min-n 5 \
    --prediction-split-parts "${PREDICTION_PARTS[@]}" \
    --device cuda:0 --ablation full >>"$log" 2>&1
  validate_artifacts
  echo "[fold_done] fold=$fold seed=$SEED time=$(date -Is)" | tee -a "$log"
}

failed=0
active=()
for fold in "${RUN_FOLDS[@]}"; do
  run_fold "$fold" &
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
if (( failed )); then
  echo "[matrix_failed] mode=$MODE" >&2
  exit 1
fi
echo "[matrix_complete] mode=$MODE folds=${RUN_FOLDS[*]} seed=$SEED time=$(date -Is)"
