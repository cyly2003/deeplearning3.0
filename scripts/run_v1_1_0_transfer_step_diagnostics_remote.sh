#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/easyai/DL1/ecotox_qsar_transfer"
PYTHON_BIN="/opt/anaconda3/bin/python"
CONFIG_PATH="$PROJECT_DIR/configs/experiment.remote.easyai.yaml"
DB_PATH="$PROJECT_DIR/outputs/derived/modeling_dataset_v1_0_0_rebuild.sqlite"
SOURCE_TABLE="aggregated_task_records_aquatic_soil_ptox_qc"
SPLIT_NAME="M_qc_aquatic_to_soil_ptox_adapt_C_f20"
OUT_DIR="$PROJECT_DIR/outputs/experiments/v1_1_0_transfer_step_diagnostics"
LOG_DIR="$PROJECT_DIR/outputs/logs"
MASTER_LOG="$LOG_DIR/v1_1_0_transfer_step_diagnostics_$(date +%Y%m%d_%H%M%S).log"
BASELINE_RUN="$PROJECT_DIR/outputs/experiments/v1_0_0_same_budget/v1.0.0_训练优化重构_同预算水相预训练土壤pTox小样本迁移_f20/deep/full/$SPLIT_NAME"

mkdir -p "$LOG_DIR" "$OUT_DIR"
exec >> "$MASTER_LOG" 2>&1

echo "[start] $(date -Is)"
echo "[project] $PROJECT_DIR"
echo "[db] $DB_PATH"
"$PYTHON_BIN" - <<'PY'
import sqlite3
db = "/home/easyai/DL1/ecotox_qsar_transfer/outputs/derived/modeling_dataset_v1_0_0_rebuild.sqlite"
con = sqlite3.connect(db)
print("[sqlite]", con.execute("pragma integrity_check").fetchone()[0])
print("[rows]", con.execute("select count(1) from aggregated_task_records_aquatic_soil_ptox_qc").fetchone()[0])
print("[split]", con.execute(
    "select split_part,count(1) from split_assignments where split_name='M_qc_aquatic_to_soil_ptox_adapt_C_f20' group by split_part"
).fetchall())
con.close()
PY

echo "[gpu]"
nvidia-smi || true

run_explanations() {
  local run_dir="$1"
  local label="$2"
  local explain_dir="$OUT_DIR/explanations/$label"
  echo "[explain] $label"
  "$PYTHON_BIN" scripts/analyze_endpoint_family_explanations.py \
    --config "$CONFIG_PATH" \
    --run-dir "$run_dir" \
    --db "$DB_PATH" \
    --source-table "$SOURCE_TABLE" \
    --split-part test \
    --families ECx NOEC LOEC \
    --out-dir "$explain_dir" \
    --max-explain-tasks-per-family 1 \
    --max-rows 256 \
    --shap-rows 16 \
    --shap-max-evals 1200 \
    --continue-on-explain-error
}

run_training() {
  local label="$1"
  local run_name_zh="$2"
  local pretrain_epochs="$3"
  local finetune_epochs="$4"
  local scheduler="$5"
  local finetune_scheduler="$6"
  local run_dir="$OUT_DIR/v1.1.0_${run_name_zh}/deep/full/$SPLIT_NAME"

  echo "[run] $label pretrain=$pretrain_epochs finetune=$finetune_epochs scheduler=$scheduler finetune_scheduler=$finetune_scheduler"
  "$PYTHON_BIN" -m qsar_tl.training.train \
    --config "$CONFIG_PATH" \
    --db "$DB_PATH" \
    --source-table "$SOURCE_TABLE" \
    --split-name "$SPLIT_NAME" \
    --out-dir "$OUT_DIR" \
    --run-version v1.1.0 \
    --run-name-zh "$run_name_zh" \
    --epochs "$pretrain_epochs" \
    --batch-size 512 \
    --learning-rate 0.0005 \
    --scheduler "$scheduler" \
    --device cuda:0 \
    --ablation full \
    --monitor-split validation \
    --validation-fraction 0.1 \
    --finetune-epochs "$finetune_epochs" \
    --finetune-learning-rate 0.0003 \
    --finetune-batch-size 512 \
    --finetune-scheduler "$finetune_scheduler" \
    --finetune-freeze none \
    --no-early-stopping

  run_explanations "$run_dir" "$label"
}

if [[ -d "$BASELINE_RUN" ]]; then
  run_explanations "$BASELINE_RUN" "baseline_30p20f_cosine"
else
  echo "[warn] baseline run not found: $BASELINE_RUN"
fi

run_training "finetune40_cosine" "迁移步长诊断_预训练30微调40_cosine" 30 40 cosine cosine
run_training "finetune20_constant" "迁移步长诊断_预训练30微调20_微调恒定学习率" 30 20 cosine none
run_training "pretrain60_cosine" "迁移步长诊断_预训练60微调20_cosine" 60 20 cosine cosine

echo "[done] $(date -Is)"
echo "[outputs] $OUT_DIR"
echo "[log] $MASTER_LOG"

