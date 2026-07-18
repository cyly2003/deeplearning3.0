#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-smoke}"
case "$MODE" in
  splits|smoke|formal) ;;
  *) echo "Usage: $0 [splits|smoke|formal]" >&2; exit 2 ;;
esac

PYTHON="${PYTHON:-/opt/anaconda3/envs/qsar-ph3/bin/python}"
CONFIG="${CONFIG:-configs/experiment.remote.easyai.yaml}"
DB="${DB:-outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite}"
SOURCE_TABLE="aggregated_task_records_medium_domain_qc_expanded"
SOIL_PTOX_TABLE="aggregated_task_records_soil_ptox_qc"
SOIL_MGKG_TABLE="aggregated_task_records_soil_mg_kg_qc"
SPLIT_NAME="M_v1_2_39_ptox_to_soil_mgkg_B_random_8_2"
AUDIT_CSV="outputs/audits/v1_2_39_ptox_to_soil_mgkg/${SPLIT_NAME}_routing_audit.csv"
CACHE_DIR="${CACHE_DIR_OVERRIDE:-outputs/cache/source_weights}"
MODEL_SEED="${MODEL_SEED_OVERRIDE:-42}"
FINETUNE_MGKG_FREEZE="${FINETUNE_MGKG_FREEZE_OVERRIDE:-none}"
FINETUNE_MGKG_BATCH_SIZE="${FINETUNE_MGKG_BATCH_SIZE_OVERRIDE:-512}"
FINETUNE_MGKG_LEARNING_RATE="${FINETUNE_MGKG_LEARNING_RATE_OVERRIDE:-0.0005}"
FINETUNE_MGKG_TRUNK_LEARNING_RATE="${FINETUNE_MGKG_TRUNK_LEARNING_RATE_OVERRIDE:-0}"
FINETUNE_MGKG_HEAD_ONLY_EPOCHS="${FINETUNE_MGKG_HEAD_ONLY_EPOCHS_OVERRIDE:-0}"
FINETUNE_MGKG_REPLAY_FRACTION="${FINETUNE_MGKG_REPLAY_FRACTION_OVERRIDE:-0}"
FINETUNE_MGKG_TOXICITY_BIN_LOSS_WEIGHT="${FINETUNE_MGKG_TOXICITY_BIN_LOSS_WEIGHT_OVERRIDE:-0}"
MGKG_RESIDUAL_ADAPTER="${MGKG_RESIDUAL_ADAPTER_OVERRIDE:-0}"
MGKG_RESIDUAL_ADAPTER_BOTTLENECK="${MGKG_RESIDUAL_ADAPTER_BOTTLENECK_OVERRIDE:-64}"
SKIP_SPLIT_BUILD="${SKIP_SPLIT_BUILD_OVERRIDE:-0}"
case "$FINETUNE_MGKG_FREEZE" in
  none|heads_only|last_trunk|heads_embeddings) ;;
  *) echo "FINETUNE_MGKG_FREEZE_OVERRIDE must be none, heads_only, last_trunk, or heads_embeddings" >&2; exit 2 ;;
esac
case "$MGKG_RESIDUAL_ADAPTER" in
  0|1) ;;
  *) echo "MGKG_RESIDUAL_ADAPTER_OVERRIDE must be 0 or 1" >&2; exit 2 ;;
esac

mkdir -p "$(dirname "$AUDIT_CSV")" "$CACHE_DIR" outputs/logs
if [[ "$SKIP_SPLIT_BUILD" != "1" ]]; then
  "$PYTHON" scripts/build_three_stage_ptox_to_soil_mgkg_split.py \
    --db "$DB" \
    --source-table "$SOURCE_TABLE" \
    --soil-ptox-source-table "$SOIL_PTOX_TABLE" \
    --soil-ptox-split-name SoilPtoxQC2_B_random_8_2 \
    --soil-mgkg-source-table "$SOIL_MGKG_TABLE" \
    --soil-mgkg-split-name SoilMgkgQC2_B_random_8_2 \
    --split-name "$SPLIT_NAME" \
    --audit-csv "$AUDIT_CSV" \
    --seed 42
fi

if [[ "$MODE" == "splits" ]]; then
  exit 0
fi

if [[ "$MODE" == "smoke" ]]; then
  STAGE1_EPOCHS=1
  STAGE2_EPOCHS=1
  STAGE3_EPOCHS=1
  OUT_ROOT="${OUT_ROOT_OVERRIDE:-outputs/experiments/v1_2_39_ptox_to_soil_mgkg_3stage_remote_smoke}"
else
  STAGE1_EPOCHS="${PRETRAIN_EPOCHS:-30}"
  STAGE2_EPOCHS="${SOIL_PTOX_EPOCHS:-20}"
  STAGE3_EPOCHS="${SOIL_MGKG_EPOCHS:-20}"
  OUT_ROOT="${OUT_ROOT_OVERRIDE:-outputs/experiments/v1_2_39_ptox_to_soil_mgkg_3stage_remote}"
fi

FREEZE_SUFFIX=""
if [[ "$FINETUNE_MGKG_FREEZE" != "none" ]]; then
  FREEZE_SUFFIX="_mgkg_${FINETUNE_MGKG_FREEZE}"
fi
RUN_NAME="${RUN_NAME_OVERRIDE:-three_stage_protocolfix_routingfix_${MODE}_full_no_adapter${FREEZE_SUFFIX}_random8_2_seed${MODEL_SEED}}"
RUN_DIR="$OUT_ROOT/v1.2.39_${RUN_NAME}/deep/full/$SPLIT_NAME"
if [[ -s "$RUN_DIR/predictions.csv" && -s "$RUN_DIR/manifest.json" && -s "$RUN_DIR/best_model.pt" ]]; then
  echo "[skip-existing] $RUN_DIR"
  exit 0
fi

MGKG_ADAPTER_ARGS=(--no-mgkg-residual-adapter)
if [[ "$MGKG_RESIDUAL_ADAPTER" == "1" ]]; then
  MGKG_ADAPTER_ARGS=(
    --mgkg-residual-adapter
    --mgkg-residual-adapter-bottleneck "$MGKG_RESIDUAL_ADAPTER_BOTTLENECK"
  )
fi

"$PYTHON" -m qsar_tl.training.train \
  --config "$CONFIG" \
  --db "$DB" \
  --out-dir "$OUT_ROOT" \
  --run-version v1.2.39 \
  --run-name-zh "$RUN_NAME" \
  --source-table "$SOURCE_TABLE" \
  --split-name "$SPLIT_NAME" \
  --seed "$MODEL_SEED" \
  --target-standardization per_task_target \
  --head-routing task_target \
  --allow-mixed-target-dimensions \
  --no-medium-adapters \
  --batch-size 512 \
  --scheduler cosine \
  --finetune-scheduler cosine \
  --finetune-mgkg-scheduler cosine \
  --learning-rate 0.0005 \
  --finetune-learning-rate 0.0001 \
  --finetune-mgkg-learning-rate "$FINETUNE_MGKG_LEARNING_RATE" \
  --finetune-mgkg-trunk-learning-rate "$FINETUNE_MGKG_TRUNK_LEARNING_RATE" \
  --finetune-mgkg-head-only-epochs "$FINETUNE_MGKG_HEAD_ONLY_EPOCHS" \
  --finetune-mgkg-replay-fraction "$FINETUNE_MGKG_REPLAY_FRACTION" \
  --finetune-mgkg-toxicity-bin-loss-weight "$FINETUNE_MGKG_TOXICITY_BIN_LOSS_WEIGHT" \
  --finetune-mgkg-batch-size "$FINETUNE_MGKG_BATCH_SIZE" \
  --finetune-freeze none \
  --finetune-mgkg-freeze "$FINETUNE_MGKG_FREEZE" \
  "${MGKG_ADAPTER_ARGS[@]}" \
  --finetune-validation-fraction 0.2 \
  --finetune-mgkg-validation-fraction 0.2 \
  --early-stopping \
  --early-stopping-patience 15 \
  --early-stopping-min-delta 0.0 \
  --monitor-split internal_train_fraction \
  --validation-fraction 0.1 \
  --source-weighting-method tanimoto_to_finetune \
  --source-weighting-alpha 1.0 \
  --source-weight-cache-dir "$CACHE_DIR" \
  --toxicity-binning \
  --toxicity-binning-mode aux_classification \
  --toxicity-binning-scheme authority_v1 \
  --toxicity-binning-loss-weight 0.025 \
  --no-censored-loss \
  --no-effect-level-weighting \
  --weight-decay 0.00001 \
  --dropout 0.10 \
  --metric-min-n 5 \
  --device cuda:0 \
  --epochs "$STAGE1_EPOCHS" \
  --finetune-epochs "$STAGE2_EPOCHS" \
  --finetune-mgkg-epochs "$STAGE3_EPOCHS" \
  --ablation full
