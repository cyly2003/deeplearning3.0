#!/usr/bin/env bash
set -euo pipefail

# v1.2.50: M10 input ablations on the locked v1.2.44 random 8:2 boundary.
# This deliberately does NOT reuse any v1.2.49 component/scaffold assignment.
# M10-Full is reused from the validated v1.2.44 four-seed reference; this
# runner trains only the four missing, protocol-matched input views.

MODE="${1:-formal}"
shift || true
CELLS=("$@")
case "$MODE" in smoke|formal|validate) ;; *) echo "Usage: $0 [smoke|formal|validate] [cells...]" >&2; exit 2;; esac
if [[ ${#CELLS[@]} -eq 0 ]]; then
  CELLS=(M10_CONTEXT M10_MOLECULE M10_DESCRIPTOR_CONTEXT M10_FINGERPRINT_CONTEXT)
fi

PYTHON="${PYTHON:-/opt/anaconda3/envs/qsar-ph3/bin/python}"
CONFIG="${CONFIG:-configs/experiment.remote.easyai.yaml}"
DB="${DB:-outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite}"
SOURCE_TABLE="aggregated_task_records_ptox_soil_mass_molar_qc"
M10_SPLIT="M_v1_2_44_M10_仅水相预训练_固定评价边界"
SPLIT_AUDIT="outputs/audits/第二层核心因果实验矩阵_v1_2_44/固定拆分与身份审计.json"
OUT_ROOT="${OUT_ROOT:-outputs/experiments/v1_2_50_m10_random_ablation}"
LOG_ROOT="outputs/logs"
CACHE_ROOT="${CACHE_ROOT:-outputs/cache/v1_2_50_m10_random_ablation_source_weights}"
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"
SEEDS=(42 2042 3407 8417)
PREDICTION_PARTS=(finetune_mgkg finetune_mgkg_validation test)

mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$CACHE_ROOT"
exec 8>"$LOG_ROOT/v1_2_50_m10_random_ablation.lock"
if ! flock -n 8; then
  echo "[blocked] v1.2.50 M10 random-ablation controller is already running" >&2
  exit 75
fi

# Fails closed if the random 8:2 M10 assignment no longer has the locked
# v1.2.44 identity contract.  No split builder is invoked here.
"$PYTHON" scripts/validate_v1_2_44_split_state.py \
  --db "$DB" --source-table "$SOURCE_TABLE" --split-summary "$SPLIT_AUDIT"
if [[ "$MODE" == "validate" ]]; then
  echo "[validate_complete] split=$M10_SPLIT"
  exit 0
fi

if [[ "$MODE" == "smoke" ]]; then
  STAGE1_EPOCHS=1; STAGE3_EPOCHS=1; RUN_ROOT="${OUT_ROOT}_smoke"; RUN_SEEDS=(42)
else
  STAGE1_EPOCHS=30; STAGE3_EPOCHS=30; RUN_ROOT="$OUT_ROOT"; RUN_SEEDS=("${SEEDS[@]}")
fi
mkdir -p "$RUN_ROOT"

cell_contract() {
  local cell="$1" seed="$2"
  case "$cell" in
    M10_CONTEXT)
      ABLATION="no_molecular_input"; LABEL="M10_ContextOnly";;
    M10_MOLECULE)
      ABLATION="no_context"; LABEL="M10_MoleculeOnly";;
    M10_DESCRIPTOR_CONTEXT)
      ABLATION="descriptors_with_context"; LABEL="M10_DescriptorContext";;
    M10_FINGERPRINT_CONTEXT)
      ABLATION="fingerprint_with_context"; LABEL="M10_FingerprintContext";;
    *) echo "Unknown v1.2.50 M10 cell: $cell" >&2; return 2;;
  esac
  RUN_NAME="${LABEL}_R8_2_固定评价边界_种子${seed}"
  RUN_DIR="$RUN_ROOT/v1.2.50_${RUN_NAME}/deep/$ABLATION/$M10_SPLIT"
}

validate_artifacts() {
  [[ -s "$RUN_DIR/predictions.csv" && -s "$RUN_DIR/manifest.json" && -s "$RUN_DIR/best_model.pt" ]]
}

run_cell() {
  local cell seed log
  cell="$1"; seed="$2"
  cell_contract "$cell" "$seed"
  log="$LOG_ROOT/v1_2_50_${cell}_seed${seed}.log"
  if validate_artifacts; then
    echo "[skip-valid-existing] cell=$cell seed=$seed run_dir=$RUN_DIR"
    return 0
  fi
  echo "[cell_start] cell=$cell seed=$seed split=$M10_SPLIT weighting=none time=$(date -Is)" | tee -a "$log"
  "$PYTHON" -m qsar_tl.training.train \
    --config "$CONFIG" --db "$DB" --out-dir "$RUN_ROOT" --run-version v1.2.50 --run-name-zh "$RUN_NAME" \
    --source-table "$SOURCE_TABLE" --split-name "$M10_SPLIT" \
    --task-filter-min-total 0 --task-filter-min-train 0 --task-filter-min-eval 0 \
    --seed "$seed" --epochs "$STAGE1_EPOCHS" --finetune-epochs 0 --finetune-mgkg-epochs "$STAGE3_EPOCHS" \
    --batch-size 512 --finetune-batch-size 512 --finetune-mgkg-batch-size 512 \
    --scheduler cosine --finetune-scheduler cosine --finetune-mgkg-scheduler cosine \
    --learning-rate 0.0005 --finetune-learning-rate 0.0001 --finetune-mgkg-learning-rate 0.0005 \
    --finetune-mgkg-trunk-learning-rate 0 --finetune-freeze none --finetune-mgkg-freeze none --finetune-mgkg-head-only-epochs 0 \
    --finetune-mgkg-replay-fraction 0 --finetune-mgkg-toxicity-bin-loss-weight 0 --no-finetune-mgkg-target-bin-sampling \
    --no-finetune-mgkg-hierarchical-head --no-mgkg-residual-adapter --no-swa --weight-decay 0.00001 --dropout 0.10 \
    --target-standardization per_task_target --head-routing task_target --allow-mixed-target-dimensions --no-medium-adapters \
    --early-stopping --early-stopping-patience 15 --early-stopping-min-delta 0 \
    --monitor-split internal_train_fraction --validation-fraction 0.1 --validation-seed 42 \
    --finetune-validation-fraction 0.2 --finetune-validation-seed 42 \
    --finetune-mgkg-monitor-split valid --finetune-mgkg-validation-fraction 0 --finetune-mgkg-validation-seed 42 \
    --source-weighting-method none --source-weighting-alpha 1 --source-weight-cache-dir "$CACHE_ROOT" \
    --toxicity-binning --toxicity-binning-mode aux_classification --toxicity-binning-scheme authority_v1 --toxicity-binning-loss-weight 0.025 \
    --no-censored-loss --no-effect-level-weighting --metric-min-n 5 \
    --prediction-split-parts "${PREDICTION_PARTS[@]}" --device cuda:0 --ablation "$ABLATION" >>"$log" 2>&1
  validate_artifacts
  echo "[cell_done] cell=$cell seed=$seed time=$(date -Is)" | tee -a "$log"
}

run_batch() {
  local cell seed pid failed=0
  local -a active=()
  cell="$1"
  for seed in "${RUN_SEEDS[@]}"; do
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
  if (( failed )); then
    echo "[batch_failed] cell=$cell" >&2
    return 1
  fi
}

for cell in "${CELLS[@]}"; do run_batch "$cell"; done
echo "[matrix_complete] mode=$MODE cells=${CELLS[*]} seeds=${RUN_SEEDS[*]} split=$M10_SPLIT time=$(date -Is)"
