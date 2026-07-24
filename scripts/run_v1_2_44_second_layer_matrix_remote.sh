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
PARENT_SPLIT="M_v1_2_40_ptox_to_soil_molkg_B_random_8_2"
M11_SPLIT="M_v1_2_44_M11_三阶段_固定评价边界"
M00_SPLIT="M_v1_2_44_M00_仅Stage3从头训练_固定评价边界"
M10_SPLIT="M_v1_2_44_M10_仅水相预训练_固定评价边界"
M01_SPLIT="M_v1_2_44_M01_仅土壤pTox_固定评价边界"
FORMAL_ROOT="outputs/experiments/第二层核心因果实验矩阵_v1_2_44"
OUT_ROOT="${OUT_ROOT:-$FORMAL_ROOT}"
AUDIT_DIR="${AUDIT_DIR:-outputs/audits/第二层核心因果实验矩阵_v1_2_44}"
SPLIT_SUMMARY="$AUDIT_DIR/固定拆分与身份审计.json"
CACHE_ROOT="${CACHE_ROOT:-outputs/cache/第二层核心因果实验矩阵_v1_2_44_Stage2}"
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"
REBUILD_SPLITS="${REBUILD_SPLITS:-1}"
SEEDS=(42 2042 3407 8417)
PREDICTION_PARTS=(finetune_mgkg finetune_mgkg_validation test)

mkdir -p "$AUDIT_DIR" "$CACHE_ROOT" outputs/logs

build_splits() {
  "$PYTHON" scripts/build_v1_2_44_second_layer_splits.py \
    --db "$DB" \
    --source-table "$SOURCE_TABLE" \
    --parent-split "$PARENT_SPLIT" \
    --m11-split "$M11_SPLIT" \
    --m00-split "$M00_SPLIT" \
    --m10-split "$M10_SPLIT" \
    --m01-split "$M01_SPLIT" \
    --audit-json "$SPLIT_SUMMARY"
}

validate_existing_splits() {
  "$PYTHON" scripts/validate_v1_2_44_split_state.py \
    --db "$DB" \
    --source-table "$SOURCE_TABLE" \
    --split-summary "$SPLIT_SUMMARY"
}

if [[ "$MODE" == "build" ]]; then
  build_splits
  exit 0
fi

exec 8>outputs/logs/v1_2_44_second_layer_matrix.lock
if ! flock -n 8; then
  echo "[blocked] v1.2.44 second-layer matrix is already queued or running" >&2
  exit 75
fi
# Serialize behind the retained v1.2.40 source controller.
exec 6>outputs/logs/v1_2_40_paired_mass_molar_matrix.lock
flock 6

if [[ "$REBUILD_SPLITS" == "1" ]]; then
  build_splits
else
  validate_existing_splits
fi

if [[ "$MODE" == "smoke" ]]; then
  STAGE1_EPOCHS=1
  STAGE2_EPOCHS=1
  STAGE3_EPOCHS=1
  RUN_ROOT="${OUT_ROOT}_冒烟测试"
  RUN_CACHE_ROOT="${CACHE_ROOT}_冒烟测试"
  RUN_SEEDS=(42)
else
  STAGE1_EPOCHS=30
  STAGE2_EPOCHS=20
  STAGE3_EPOCHS=30
  RUN_ROOT="$OUT_ROOT"
  RUN_CACHE_ROOT="$CACHE_ROOT"
  RUN_SEEDS=("${SEEDS[@]}")
fi
mkdir -p "$RUN_ROOT" "$RUN_CACHE_ROOT"

cell_contract() {
  local cell="$1" seed="$2"
  TASK_MIN_TOTAL=200
  TASK_MIN_TRAIN=100
  case "$cell" in
    M00)
      CELL_SPLIT="$M00_SPLIT"; ABLATION="full"
      RUN_NAME="M00_从头训练_固定评价边界_种子${seed}"
      CELL_STAGE1_EPOCHS=0; CELL_STAGE2_EPOCHS=0
      STAGE1_LR=0.0005; STAGE1_VALID_FRAC=0.1; STAGE1_VALID_SEED=42
      TASK_MIN_EVAL=30; STAGE3_FREEZE="none"; SOURCE_WEIGHTING="none"
      ;;
    M10)
      CELL_SPLIT="$M10_SPLIT"; ABLATION="full"
      RUN_NAME="M10_水相预训练后直接迁移_种子${seed}"
      CELL_STAGE1_EPOCHS="$STAGE1_EPOCHS"; CELL_STAGE2_EPOCHS=0
      STAGE1_LR=0.0005; STAGE1_VALID_FRAC=0.1; STAGE1_VALID_SEED=42
      TASK_MIN_TOTAL=0; TASK_MIN_TRAIN=0
      TASK_MIN_EVAL=0; STAGE3_FREEZE="none"; SOURCE_WEIGHTING="none"
      ;;
    M01)
      CELL_SPLIT="$M01_SPLIT"; ABLATION="full"
      RUN_NAME="M01_土壤pTox后迁移_种子${seed}"
      CELL_STAGE1_EPOCHS="$STAGE2_EPOCHS"; CELL_STAGE2_EPOCHS=0
      STAGE1_LR=0.0001; STAGE1_VALID_FRAC=0.2; STAGE1_VALID_SEED=17073
      TASK_MIN_TOTAL=0; TASK_MIN_TRAIN=0
      TASK_MIN_EVAL=0; STAGE3_FREEZE="none"; SOURCE_WEIGHTING="none"
      ;;
    M11F)
      CELL_SPLIT="$M11_SPLIT"; ABLATION="full"
      RUN_NAME="M11F_完整三阶段冻结主干_种子${seed}"
      CELL_STAGE1_EPOCHS="$STAGE1_EPOCHS"; CELL_STAGE2_EPOCHS="$STAGE2_EPOCHS"
      STAGE1_LR=0.0005; STAGE1_VALID_FRAC=0.1; STAGE1_VALID_SEED=42
      TASK_MIN_EVAL=30; STAGE3_FREEZE="heads_only"; SOURCE_WEIGHTING="tanimoto_to_finetune"
      ;;
    M11U)
      CELL_SPLIT="$M11_SPLIT"; ABLATION="full"
      RUN_NAME="M11U复现_完整三阶段全参数微调_种子${seed}"
      CELL_STAGE1_EPOCHS="$STAGE1_EPOCHS"; CELL_STAGE2_EPOCHS="$STAGE2_EPOCHS"
      STAGE1_LR=0.0005; STAGE1_VALID_FRAC=0.1; STAGE1_VALID_SEED=42
      TASK_MIN_EVAL=30; STAGE3_FREEZE="none"; SOURCE_WEIGHTING="tanimoto_to_finetune"
      ;;
    B2_CONTEXT)
      CELL_SPLIT="$M11_SPLIT"; ABLATION="no_molecular_input"
      RUN_NAME="B2_仅上下文输入_种子${seed}"
      CELL_STAGE1_EPOCHS="$STAGE1_EPOCHS"; CELL_STAGE2_EPOCHS="$STAGE2_EPOCHS"
      STAGE1_LR=0.0005; STAGE1_VALID_FRAC=0.1; STAGE1_VALID_SEED=42
      TASK_MIN_EVAL=30; STAGE3_FREEZE="none"; SOURCE_WEIGHTING="tanimoto_to_finetune"
      ;;
    B2_MOLECULE)
      CELL_SPLIT="$M11_SPLIT"; ABLATION="no_context"
      RUN_NAME="B2_仅分子输入_种子${seed}"
      CELL_STAGE1_EPOCHS="$STAGE1_EPOCHS"; CELL_STAGE2_EPOCHS="$STAGE2_EPOCHS"
      STAGE1_LR=0.0005; STAGE1_VALID_FRAC=0.1; STAGE1_VALID_SEED=42
      TASK_MIN_EVAL=30; STAGE3_FREEZE="none"; SOURCE_WEIGHTING="tanimoto_to_finetune"
      ;;
    *) echo "Unknown v1.2.44 cell: $cell" >&2; return 2 ;;
  esac
  RUN_DIR="$RUN_ROOT/v1.2.44_${RUN_NAME}/deep/$ABLATION/$CELL_SPLIT"
  STAGE2_CACHE="$RUN_CACHE_ROOT/M11_种子${seed}_共同Stage2权重.pt"
  M11F_RUN_DIR="$RUN_ROOT/v1.2.44_M11F_完整三阶段冻结主干_种子${seed}/deep/full/$M11_SPLIT"
}

validate_cell() (
  # Run contract resolution in a subshell so validating the paired M11F run
  # cannot overwrite the caller's M11U globals (freeze mode, run name/path,
  # epochs, or task thresholds).
  local cell="$1" seed="$2"
  local -a extra=()
  cell_contract "$cell" "$seed"
  if [[ "$cell" == "M11F" ]]; then
    extra=(--stage2-cache "$STAGE2_CACHE")
  elif [[ "$cell" == "M11U" ]]; then
    extra=(--stage2-cache "$STAGE2_CACHE" --paired-run-dir "$M11F_RUN_DIR")
  fi
  [[ "$MODE" == "smoke" ]] && extra+=(--smoke)
  "$PYTHON" scripts/validate_v1_2_44_matrix_run.py \
    --run-dir "$RUN_DIR" \
    --cell "$cell" \
    --seed "$seed" \
    --split-summary "$SPLIT_SUMMARY" \
    --source-table "$SOURCE_TABLE" \
    "${extra[@]}"
)

run_cell() {
  local cell="$1" seed="$2" log
  local -a checkpoint_args=()
  cell_contract "$cell" "$seed"
  log="outputs/logs/v1_2_44_${cell}_seed${seed}.log"

  if [[ -s "$RUN_DIR/predictions.csv" && -s "$RUN_DIR/manifest.json" && -s "$RUN_DIR/best_model.pt" ]]; then
    validate_cell "$cell" "$seed" >>"$log" 2>&1
    echo "[skip-valid-existing] cell=$cell seed=$seed run_dir=$RUN_DIR"
    return 0
  fi
  if [[ -d "$RUN_DIR" ]]; then
    echo "[resume-restart-incomplete-cell] cell=$cell seed=$seed run_dir=$RUN_DIR" | tee -a "$log"
  fi
  if [[ "$cell" == "M11F" ]]; then
    checkpoint_args=(--export-finetune-mgkg-init-checkpoint "$STAGE2_CACHE")
  elif [[ "$cell" == "M11U" ]]; then
    if [[ ! -s "$STAGE2_CACHE" ]]; then
      echo "[missing-same-seed-stage2-checkpoint] cell=$cell seed=$seed cache=$STAGE2_CACHE" >&2
      return 1
    fi
    validate_cell M11F "$seed" >>"$log" 2>&1
    checkpoint_args=(--finetune-mgkg-init-checkpoint "$STAGE2_CACHE")
  fi

  echo "[cell_start] cell=$cell seed=$seed time=$(date -Is)" | tee -a "$log"
  "$PYTHON" -m qsar_tl.training.train \
    --config "$CONFIG" \
    --db "$DB" \
    --out-dir "$RUN_ROOT" \
    --run-version v1.2.44 \
    --run-name-zh "$RUN_NAME" \
    --source-table "$SOURCE_TABLE" \
    --split-name "$CELL_SPLIT" \
    --task-filter-min-total "$TASK_MIN_TOTAL" \
    --task-filter-min-train "$TASK_MIN_TRAIN" \
    --task-filter-min-eval "$TASK_MIN_EVAL" \
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
    --learning-rate "$STAGE1_LR" \
    --finetune-learning-rate 0.0001 \
    --finetune-mgkg-learning-rate 0.0005 \
    --finetune-mgkg-trunk-learning-rate 0 \
    --finetune-freeze none \
    --finetune-mgkg-freeze "$STAGE3_FREEZE" \
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
    --validation-fraction "$STAGE1_VALID_FRAC" \
    --validation-seed "$STAGE1_VALID_SEED" \
    --finetune-validation-fraction 0.2 \
    --finetune-validation-seed 42 \
    --finetune-mgkg-monitor-split valid \
    --finetune-mgkg-validation-fraction 0 \
    --finetune-mgkg-validation-seed 42 \
    --source-weighting-method "$SOURCE_WEIGHTING" \
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
    --ablation "$ABLATION" \
    "${checkpoint_args[@]}" >>"$log" 2>&1

  validate_cell "$cell" "$seed" >>"$log" 2>&1
  echo "[cell_done] cell=$cell seed=$seed time=$(date -Is)" | tee -a "$log"
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
    echo "[batch_failed] cell=$cell time=$(date -Is)" >&2
    return 1
  fi
}

# M00 must be trained on the same explicit Stage-3 train/validation/test
# boundary as the transfer routes. The order is causal: M11F creates one fresh
# same-seed Stage-2 state and only then may M11U replay load it. No stale
# baseline checkpoint is accepted.
for cell in M00 M10 M01 M11F M11U B2_CONTEXT B2_MOLECULE; do
  run_batch "$cell" "${RUN_SEEDS[@]}"
done

if [[ "$MODE" == "smoke" ]]; then
  echo "[smoke_complete] time=$(date -Is) root=$RUN_ROOT split_audit=$SPLIT_SUMMARY"
  exit 0
fi

"$PYTHON" scripts/summarize_v1_2_44_second_layer_matrix.py \
  --legacy-root outputs/experiments/v1_2_40_paired_mass_molar_matrix_remote \
  --matrix-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/统一汇总_中文" \
  --seeds "${SEEDS[@]}"

echo "[matrix_complete] time=$(date -Is) root=$RUN_ROOT split_audit=$SPLIT_SUMMARY"
