#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-formal}"
case "$MODE" in
  build|smoke|formal) ;;
  *) echo "Usage: $0 [build|smoke|formal]" >&2; exit 2 ;;
esac

PYTHON="${PYTHON:-/opt/anaconda3/envs/qsar-ph3/bin/python}"
CONFIG="${CONFIG:-configs/experiment.remote.easyai.yaml}"
DB="${DB:-outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite}"
SOURCE_TABLE="aggregated_task_records_ptox_soil_mass_molar_qc"
PARENT_SPLIT="M_v1_2_44_M11_三阶段_固定评价边界"
M00_SPLIT="M_v1_2_47_SF_M00_从头训练_scaffold_family"
M10_SPLIT="M_v1_2_47_SF_M10_水相预训练_scaffold_family"
M11U_SPLIT="M_v1_2_47_SF_M11U_完整三阶段_scaffold_family"
FORMAL_ROOT="outputs/experiments/v1_2_47_scaffold_family_matrix"
OUT_ROOT="${OUT_ROOT:-$FORMAL_ROOT}"
AUDIT_DIR="${AUDIT_DIR:-outputs/audits/v1_2_47_scaffold_family}"
SPLIT_SUMMARY="$AUDIT_DIR/split_summary.json"
SOURCE_WEIGHT_CACHE="${SOURCE_WEIGHT_CACHE:-outputs/cache/v1_2_47_scaffold_family_source_weights}"
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"
REBUILD_SPLITS="${REBUILD_SPLITS:-1}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-20000}"
SEEDS=(42 2042 3407 8417)
PREDICTION_PARTS=(finetune_mgkg finetune_mgkg_validation test)

mkdir -p "$AUDIT_DIR" "$SOURCE_WEIGHT_CACHE" outputs/logs

build_splits() {
  "$PYTHON" scripts/build_v1_2_47_scaffold_family_splits.py \
    --db "$DB" \
    --source-table "$SOURCE_TABLE" \
    --parent-split "$PARENT_SPLIT" \
    --m00-split "$M00_SPLIT" \
    --m10-split "$M10_SPLIT" \
    --m11u-split "$M11U_SPLIT" \
    --audit-json "$SPLIT_SUMMARY"
}

validate_splits() {
  "$PYTHON" scripts/build_v1_2_47_scaffold_family_splits.py \
    --db "$DB" \
    --source-table "$SOURCE_TABLE" \
    --m00-split "$M00_SPLIT" \
    --m10-split "$M10_SPLIT" \
    --m11u-split "$M11U_SPLIT" \
    --audit-json "$SPLIT_SUMMARY" \
    --validate-only
}

exec 8>outputs/logs/v1_2_47_scaffold_family_matrix.lock
if ! flock -n 8; then
  echo "[blocked] v1.2.47 scaffold-family matrix is already queued or running" >&2
  exit 75
fi
# Serialize the retained controllers sharing the v1.2.40/v1.2.44 source DB or GPU.
exec 9>outputs/logs/v1_2_46_reference_group_matrix.lock
flock 9
exec 7>outputs/logs/v1_2_44_second_layer_matrix.lock
flock 7
exec 3>outputs/logs/v1_2_40_paired_mass_molar_matrix.lock
flock 3

if [[ "$REBUILD_SPLITS" == "1" ]]; then
  build_splits
else
  validate_splits
fi

if [[ "$MODE" == "build" ]]; then
  echo "[build_complete] time=$(date -Is) split_audit=$SPLIT_SUMMARY"
  exit 0
fi

if [[ "$MODE" == "smoke" ]]; then
  STAGE1_EPOCHS=1
  STAGE2_EPOCHS=1
  STAGE3_EPOCHS=1
  RUN_ROOT="${OUT_ROOT}_smoke"
  RUN_SEEDS=(42)
else
  STAGE1_EPOCHS=30
  STAGE2_EPOCHS=20
  STAGE3_EPOCHS=30
  RUN_ROOT="$OUT_ROOT"
  RUN_SEEDS=("${SEEDS[@]}")
fi
mkdir -p "$RUN_ROOT"

cell_contract() {
  local cell="$1" seed="$2"
  case "$cell" in
    M00)
      CELL_SPLIT="$M00_SPLIT"
      RUN_NAME="SF-M00_scaffold_family从头训练_种子${seed}"
      CELL_STAGE1_EPOCHS=0
      CELL_STAGE2_EPOCHS=0
      SOURCE_WEIGHTING="none"
      ;;
    M10)
      CELL_SPLIT="$M10_SPLIT"
      RUN_NAME="SF-M10_scaffold_family仅水相预训练_种子${seed}"
      CELL_STAGE1_EPOCHS="$STAGE1_EPOCHS"
      CELL_STAGE2_EPOCHS=0
      SOURCE_WEIGHTING="none"
      ;;
    M11U)
      CELL_SPLIT="$M11U_SPLIT"
      RUN_NAME="SF-M11U_scaffold_family完整三阶段全参数微调_种子${seed}"
      CELL_STAGE1_EPOCHS="$STAGE1_EPOCHS"
      CELL_STAGE2_EPOCHS="$STAGE2_EPOCHS"
      SOURCE_WEIGHTING="tanimoto_to_finetune"
      ;;
    *) echo "Unknown v1.2.47 cell: $cell" >&2; return 2 ;;
  esac
  RUN_DIR="$RUN_ROOT/v1.2.47_${RUN_NAME}/deep/full/$CELL_SPLIT"
}

validate_cell() (
  local cell="$1" seed="$2"
  cell_contract "$cell" "$seed"
  local -a extra=()
  [[ "$MODE" == "smoke" ]] && extra+=(--smoke)
  "$PYTHON" scripts/validate_v1_2_47_scaffold_family_run.py \
    --run-dir "$RUN_DIR" \
    --cell "$cell" \
    --seed "$seed" \
    --split-summary "$SPLIT_SUMMARY" \
    --source-table "$SOURCE_TABLE" \
    "${extra[@]}"
)

run_cell() {
  local cell="$1" seed="$2" log
  cell_contract "$cell" "$seed"
  log="outputs/logs/v1_2_47_${MODE}_${cell}_seed${seed}.log"
  if [[ -s "$RUN_DIR/predictions.csv" && -s "$RUN_DIR/manifest.json" && -s "$RUN_DIR/best_model.pt" ]]; then
    validate_cell "$cell" "$seed" >>"$log" 2>&1
    echo "[skip-valid-existing] cell=$cell seed=$seed run_dir=$RUN_DIR"
    return 0
  fi
  if [[ -d "$RUN_DIR" ]]; then
    echo "[restart-incomplete-cell] cell=$cell seed=$seed run_dir=$RUN_DIR" | tee -a "$log"
  fi
  echo "[cell_start] mode=$MODE cell=$cell seed=$seed time=$(date -Is)" | tee -a "$log"
  "$PYTHON" -m qsar_tl.training.train \
    --config "$CONFIG" \
    --db "$DB" \
    --out-dir "$RUN_ROOT" \
    --run-version v1.2.47 \
    --run-name-zh "$RUN_NAME" \
    --source-table "$SOURCE_TABLE" \
    --split-name "$CELL_SPLIT" \
    --task-filter-min-total 0 \
    --task-filter-min-train 0 \
    --task-filter-min-eval 0 \
    --seed "$seed" \
    --epochs "$CELL_STAGE1_EPOCHS" \
    --finetune-epochs "$CELL_STAGE2_EPOCHS" \
    --finetune-mgkg-epochs "$STAGE3_EPOCHS" \
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
    --source-weighting-method "$SOURCE_WEIGHTING" \
    --source-weighting-alpha 1 \
    --source-weight-cache-dir "$SOURCE_WEIGHT_CACHE" \
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

  validate_cell "$cell" "$seed" >>"$log" 2>&1
  echo "[cell_done] mode=$MODE cell=$cell seed=$seed time=$(date -Is)" | tee -a "$log"
}

run_batch() {
  local cell="$1"; shift
  local seeds=("$@") active=() failed=0 seed pid
  for seed in "${seeds[@]}"; do
    run_cell "$cell" "$seed" &
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
    echo "[batch_failed] mode=$MODE cell=$cell time=$(date -Is)" >&2
    return 1
  fi
}

for cell in M00 M10 M11U; do
  run_batch "$cell" "${RUN_SEEDS[@]}"
done

if [[ "$MODE" == "smoke" ]]; then
  echo "[smoke_complete] time=$(date -Is) root=$RUN_ROOT split_audit=$SPLIT_SUMMARY"
  exit 0
fi

"$PYTHON" scripts/summarize_v1_2_47_scaffold_family_matrix.py \
  --root "$RUN_ROOT" \
  --split-summary "$SPLIT_SUMMARY" \
  --output-dir "$RUN_ROOT/统一汇总_中文" \
  --seeds "${SEEDS[@]}" \
  --replicates "$BOOTSTRAP_REPLICATES"

echo "[matrix_complete] time=$(date -Is) root=$RUN_ROOT split_audit=$SPLIT_SUMMARY"
