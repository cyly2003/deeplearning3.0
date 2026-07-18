#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-smoke}"
case "$MODE" in
  smoke|formal) ;;
  *) echo "Usage: $0 [smoke|formal]" >&2; exit 2 ;;
esac

PYTHON="${PYTHON:-/opt/anaconda3/envs/qsar-ph3/bin/python}"
CONFIG="${CONFIG:-configs/experiment.remote.easyai.yaml}"
DB="${DB:-outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite}"
SOURCE_TABLE="aggregated_task_records_soil_mg_kg_qc"
SPLIT_NAME="SoilMgkgQC2_B_random_8_2"

if [[ "$MODE" == "smoke" ]]; then
  EPOCHS=1
  OUT_ROOT="${OUT_ROOT_OVERRIDE:-outputs/experiments/v1_2_38_soil_mgkg_remote_smoke}"
else
  EPOCHS="${EPOCHS:-30}"
  OUT_ROOT="${OUT_ROOT_OVERRIDE:-outputs/experiments/v1_2_38_soil_mgkg_remote}"
fi

RUN_NAME="${RUN_NAME_OVERRIDE:-soil_mgkg_protocolfix_no_adapter_no_censored_${MODE}_random8_2_seed42}"
RUN_DIR="$OUT_ROOT/v1.2.38_${RUN_NAME}/deep/full/$SPLIT_NAME"
if [[ -s "$RUN_DIR/predictions.csv" && -s "$RUN_DIR/manifest.json" && -s "$RUN_DIR/best_model.pt" ]]; then
  echo "[skip-existing] $RUN_DIR"
  exit 0
fi

"$PYTHON" -m qsar_tl.training.train \
  --config "$CONFIG" \
  --db "$DB" \
  --out-dir "$OUT_ROOT" \
  --run-version v1.2.38 \
  --run-name-zh "$RUN_NAME" \
  --source-table "$SOURCE_TABLE" \
  --split-name "$SPLIT_NAME" \
  --seed 42 \
  --epochs "$EPOCHS" \
  --finetune-epochs 0 \
  --batch-size 512 \
  --scheduler cosine \
  --target-standardization per_task_target \
  --no-medium-adapters \
  --device cuda:0 \
  --metric-min-n 5 \
  --early-stopping \
  --early-stopping-patience 15 \
  --early-stopping-min-delta 0.0 \
  --monitor-split internal_train_fraction \
  --validation-fraction 0.1 \
  --source-weighting-method none \
  --source-weighting-alpha 1.0 \
  --toxicity-binning \
  --toxicity-binning-mode aux_classification \
  --toxicity-binning-scheme authority_v1 \
  --toxicity-binning-loss-weight 0.025 \
  --no-censored-loss \
  --no-effect-level-weighting \
  --learning-rate 0.0005 \
  --dropout 0.10 \
  --weight-decay 0.000009856751793848817 \
  --ablation full
