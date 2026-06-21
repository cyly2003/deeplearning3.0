#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/easyai/DL1/ecotox_qsar_transfer"
PYTHON_BIN="/opt/anaconda3/bin/python"
CONFIG_PATH="$PROJECT_DIR/configs/experiment.remote.easyai.yaml"
DB_PATH="$PROJECT_DIR/outputs/derived/modeling_dataset_v1_0_0_rebuild.sqlite"
OUT_DIR="$PROJECT_DIR/outputs/experiments/v1_0_0_same_budget"
LOG_DIR="$PROJECT_DIR/outputs/logs"
MASTER_LOG="$LOG_DIR/v1_0_0_same_budget_$(date +%Y%m%d_%H%M%S).log"

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
for table in [
    "aggregated_task_records_aquatic_ptox_qc",
    "aggregated_task_records_aquatic_soil_ptox_qc",
]:
    print(f"[rows] {table}", con.execute(f"select count(1) from {table}").fetchone()[0])
con.close()
PY

echo "[gpu]"
nvidia-smi || true

echo "[run] aquatic pTox B split 30 epochs"
"$PYTHON_BIN" -m qsar_tl.training.train \
  --config "$CONFIG_PATH" \
  --db "$DB_PATH" \
  --source-table aggregated_task_records_aquatic_ptox_qc \
  --split-name AquaticPtox_B_random_8_2 \
  --out-dir "$OUT_DIR" \
  --run-version v1.0.0 \
  --run-name-zh "训练优化重构_同预算水相pTox随机划分对照" \
  --epochs 30 \
  --batch-size 512 \
  --learning-rate 0.0005 \
  --device cuda:0 \
  --ablation full \
  --monitor-split validation \
  --validation-fraction 0.1 \
  --no-early-stopping

echo "[run] aquatic pretrain to soil pTox f20, 30 pretrain + 20 finetune"
"$PYTHON_BIN" -m qsar_tl.training.train \
  --config "$CONFIG_PATH" \
  --db "$DB_PATH" \
  --source-table aggregated_task_records_aquatic_soil_ptox_qc \
  --split-name M_qc_aquatic_to_soil_ptox_adapt_C_f20 \
  --out-dir "$OUT_DIR" \
  --run-version v1.0.0 \
  --run-name-zh "训练优化重构_同预算水相预训练土壤pTox小样本迁移_f20" \
  --epochs 30 \
  --batch-size 512 \
  --learning-rate 0.0005 \
  --device cuda:0 \
  --ablation full \
  --monitor-split validation \
  --validation-fraction 0.1 \
  --finetune-epochs 20 \
  --finetune-learning-rate 0.0003 \
  --finetune-batch-size 512 \
  --finetune-freeze none \
  --no-early-stopping

echo "[done] $(date -Is)"
echo "[outputs] $OUT_DIR"
echo "[log] $MASTER_LOG"

