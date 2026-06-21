#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/easyai/DL1/ecotox_qsar_transfer}"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
CONFIG_PATH="$PROJECT_DIR/configs/experiment.remote.easyai.yaml"
DB_PATH="$PROJECT_DIR/outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-512}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-30}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"
FINETUNE_LR="${FINETUNE_LR:-0.0003}"
LEARNING_RATE="${LEARNING_RATE:-0.0005}"
LOG_DIR="$PROJECT_DIR/outputs/logs"
MASTER_LOG="$LOG_DIR/v2_0_0_retrain_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$MASTER_LOG") 2>&1

cd "$PROJECT_DIR"

echo "[start] $(date -Is)"
echo "[project] $PROJECT_DIR"
echo "[db] $DB_PATH"
echo "[log] $MASTER_LOG"

run_train() {
  local out_root="$1"
  local run_version="$2"
  local run_name="$3"
  local source_table="$4"
  local split_name="$5"
  local epochs="$6"
  local batch_size="$7"
  local learning_rate="$8"
  local scheduler="$9"
  shift 9

  echo "[train] run=$run_version/$run_name split=$split_name source=$source_table epochs=$epochs"
  "$PYTHON_BIN" -m qsar_tl.training.train \
    --config "$CONFIG_PATH" \
    --db "$DB_PATH" \
    --source-table "$source_table" \
    --split-name "$split_name" \
    --out-dir "$PROJECT_DIR/$out_root" \
    --run-version "$run_version" \
    --run-name-zh "$run_name" \
    --epochs "$epochs" \
    --batch-size "$batch_size" \
    --learning-rate "$learning_rate" \
    --scheduler "$scheduler" \
    --device "$DEVICE" \
    --ablation full \
    --monitor-split validation \
    --validation-fraction 0.1 \
    --no-early-stopping \
    "$@"
}

run_explain() {
  local run_dir="$1"
  local source_table="$2"
  local out_dir="$3"
  echo "[explain] run_dir=$run_dir"
  "$PYTHON_BIN" scripts/analyze_endpoint_family_explanations.py \
    --config "$CONFIG_PATH" \
    --run-dir "$run_dir" \
    --db "$DB_PATH" \
    --source-table "$source_table" \
    --split-part test \
    --families ECx NOEC LOEC \
    --out-dir "$out_dir" \
    --max-explain-tasks-per-family 1 \
    --max-rows 256 \
    --shap-rows 16 \
    --shap-max-evals 1200 \
    --continue-on-explain-error
}

run_dir() {
  local out_root="$1"
  local run_version="$2"
  local run_name="$3"
  local split_name="$4"
  echo "$PROJECT_DIR/$out_root/${run_version}_${run_name}/deep/full/$split_name"
}

echo "[preflight] validate config"
"$PYTHON_BIN" -m qsar_tl.cli validate-config --config "$CONFIG_PATH"

echo "[preflight] gpu"
nvidia-smi || true

echo "[preflight] sqlite, feature cache, split checks"
"$PYTHON_BIN" - <<'PY'
import json
import sqlite3
from pathlib import Path

db = Path("outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite")
con = sqlite3.connect(db)
cur = con.cursor()
print("[quick_check]", cur.execute("PRAGMA quick_check").fetchone()[0])
for table in [
    "aggregated_task_records_qc",
    "aggregated_task_records_aquatic_ptox_qc",
    "aggregated_task_records_soil_ptox_qc",
    "aggregated_task_records_soil_mg_kg_qc",
    "aggregated_task_records_aquatic_soil_ptox_qc",
]:
    print("[rows]", table, cur.execute(f"select count(*) from {table}").fetchone()[0])

manifest = Path("outputs/features/molecular_features_rdkit_morgan512.jsonl.manifest.json")
print("[feature_manifest]", manifest.read_text(encoding="utf-8"))

split_names = [
    "AquaticPtoxQC2_B_random_8_2",
    "AquaticPtoxQC2_C_chemical_holdout_8_2",
    "SoilPtoxQC2_B_random_8_2",
    "SoilPtoxQC2_C_chemical_holdout_8_2",
    "SoilMgkgQC2_B_random_8_2",
    "SoilMgkgQC2_C_chemical_holdout_8_2",
    "M_v2_aquatic_to_soil_ptox_adapt_C_f10",
    "M_v2_aquatic_to_soil_ptox_adapt_C_f20",
    "M_v2_aquatic_to_soil_ptox_adapt_C_f50",
    "M_v2_aquatic_to_soil_ptox_adapt_C_f100",
    "SoilPtoxQC2_C_low_f10",
    "SoilPtoxQC2_C_low_f20",
    "SoilPtoxQC2_C_low_f50",
    "SoilPtoxQC2_C_low_f100",
]
for split in split_names:
    rows = cur.execute(
        """
        select split_part, count(*)
        from split_assignments
        where split_name = ?
        group by split_part
        order by split_part
        """,
        (split,),
    ).fetchall()
    if not rows:
        raise SystemExit(f"missing split: {split}")
    print("[split]", split, dict(rows))

for split in [
    "AquaticPtoxQC2_C_chemical_holdout_8_2",
    "SoilPtoxQC2_C_chemical_holdout_8_2",
    "SoilMgkgQC2_C_chemical_holdout_8_2",
    "AquaticPtoxQC2_F_chemical_adapt_7_2_1",
    "SoilPtoxQC2_F_chemical_adapt_7_2_1",
    "SoilMgkgQC2_F_chemical_adapt_7_2_1",
]:
    train_groups = {
        row[0]
        for row in cur.execute(
            "select distinct group_key from split_assignments where split_name=? and split_part='train'",
            (split,),
        )
    }
    nontrain_groups = {
        row[0]
        for row in cur.execute(
            "select distinct group_key from split_assignments where split_name=? and split_part in ('finetune','test')",
            (split,),
        )
    }
    overlap = train_groups & nontrain_groups
    print("[chemical_group_overlap]", split, len(overlap))
    if overlap:
        raise SystemExit(f"chemical group leakage in {split}: {list(sorted(overlap))[:5]}")
con.close()
PY

echo "[phase0] smoke training"
run_train "outputs/experiments/v2_0_0_smoke" "v2.0.0-smoke" "reclean_v2_aquatic_ptox_B_1epoch" \
  "aggregated_task_records_aquatic_ptox_qc" "AquaticPtoxQC2_B_random_8_2" \
  1 "$BATCH_SIZE" "$LEARNING_RATE" cosine

run_train "outputs/experiments/v2_0_0_smoke" "v2.0.0-smoke" "reclean_v2_soil_ptox_C_1epoch" \
  "aggregated_task_records_soil_ptox_qc" "SoilPtoxQC2_C_chemical_holdout_8_2" \
  1 "$BATCH_SIZE" "$LEARNING_RATE" cosine

run_train "outputs/experiments/v2_0_0_smoke" "v2.0.0-smoke" "reclean_v2_transfer_f20_1p1epoch" \
  "aggregated_task_records_aquatic_soil_ptox_qc" "M_v2_aquatic_to_soil_ptox_adapt_C_f20" \
  1 "$BATCH_SIZE" "$LEARNING_RATE" cosine \
  --finetune-epochs 1 \
  --finetune-learning-rate "$FINETUNE_LR" \
  --finetune-batch-size "$BATCH_SIZE" \
  --finetune-scheduler cosine \
  --finetune-freeze none

echo "[phase1] main reproduction training"
run_train "outputs/experiments/v2_0_0_main_retrain" "v2.0.0" "reclean_v2_aquatic_ptox_B_random" \
  "aggregated_task_records_aquatic_ptox_qc" "AquaticPtoxQC2_B_random_8_2" \
  "$PRETRAIN_EPOCHS" "$BATCH_SIZE" "$LEARNING_RATE" cosine

run_train "outputs/experiments/v2_0_0_main_retrain" "v2.0.0" "reclean_v2_aquatic_ptox_C_chemical" \
  "aggregated_task_records_aquatic_ptox_qc" "AquaticPtoxQC2_C_chemical_holdout_8_2" \
  "$PRETRAIN_EPOCHS" "$BATCH_SIZE" "$LEARNING_RATE" cosine

run_train "outputs/experiments/v2_0_0_main_retrain" "v2.0.0" "reclean_v2_soil_ptox_B_random" \
  "aggregated_task_records_soil_ptox_qc" "SoilPtoxQC2_B_random_8_2" \
  "$PRETRAIN_EPOCHS" "$BATCH_SIZE" "$LEARNING_RATE" cosine

run_train "outputs/experiments/v2_0_0_main_retrain" "v2.0.0" "reclean_v2_soil_ptox_C_chemical" \
  "aggregated_task_records_soil_ptox_qc" "SoilPtoxQC2_C_chemical_holdout_8_2" \
  "$PRETRAIN_EPOCHS" "$BATCH_SIZE" "$LEARNING_RATE" cosine

run_train "outputs/experiments/v2_0_0_main_retrain" "v2.0.0" "reclean_v2_soil_mgkg_B_random" \
  "aggregated_task_records_soil_mg_kg_qc" "SoilMgkgQC2_B_random_8_2" \
  "$PRETRAIN_EPOCHS" "$BATCH_SIZE" "$LEARNING_RATE" cosine

run_train "outputs/experiments/v2_0_0_main_retrain" "v2.0.0" "reclean_v2_soil_mgkg_C_chemical" \
  "aggregated_task_records_soil_mg_kg_qc" "SoilMgkgQC2_C_chemical_holdout_8_2" \
  "$PRETRAIN_EPOCHS" "$BATCH_SIZE" "$LEARNING_RATE" cosine

echo "[phase2] low-soil transfer and soil-only paired controls"
for frac in f10 f20 f50 f100; do
  run_train "outputs/experiments/v2_0_0_transfer_low_soil" "v2.0.0" "reclean_v2_transfer_${frac}" \
    "aggregated_task_records_aquatic_soil_ptox_qc" "M_v2_aquatic_to_soil_ptox_adapt_C_${frac}" \
    "$PRETRAIN_EPOCHS" "$BATCH_SIZE" "$LEARNING_RATE" cosine \
    --finetune-epochs "$FINETUNE_EPOCHS" \
    --finetune-learning-rate "$FINETUNE_LR" \
    --finetune-batch-size "$BATCH_SIZE" \
    --finetune-scheduler cosine \
    --finetune-freeze none

  run_train "outputs/experiments/v2_0_0_transfer_low_soil" "v2.0.0" "reclean_v2_soilonly_${frac}" \
    "aggregated_task_records_soil_ptox_qc" "SoilPtoxQC2_C_low_${frac}" \
    "$PRETRAIN_EPOCHS" "$BATCH_SIZE" "$LEARNING_RATE" cosine
done

echo "[summarize] phase1/phase2 metrics and select best transfer fraction"
BEST_FRAC="$("$PYTHON_BIN" - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("outputs/experiments/v2_0_0_transfer_low_soil")
rows = []
for metrics_path in root.glob("v2.0.0_reclean_v2_transfer_f*/deep/full/M_v2_aquatic_to_soil_ptox_adapt_C_f*/metrics.csv"):
    frame = pd.read_csv(metrics_path)
    test = frame[
        frame["split_part"].astype(str).eq("test")
        & frame["task_head"].astype(str).str.split("_", n=1).str[0].isin(["ECx", "NOEC", "LOEC"])
    ].copy()
    if test.empty:
        continue
    total_n = float(test["n"].sum())
    weighted_mae = float((test["mae"] * test["n"]).sum() / total_n)
    weighted_rmse = float(((test["rmse"] ** 2 * test["n"]).sum() / total_n) ** 0.5)
    run_label = metrics_path.parts[-5]
    split_name = metrics_path.parts[-2]
    frac = run_label.rsplit("_", 1)[-1]
    rows.append(
        {
            "run_label": run_label,
            "split_name": split_name,
            "fraction": frac,
            "total_n": int(total_n),
            "weighted_mae": weighted_mae,
            "pooled_rmse": weighted_rmse,
            "metrics_path": str(metrics_path),
        }
    )
summary = pd.DataFrame(rows).sort_values("weighted_mae")
out = root / "transfer_fraction_summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(out, index=False, encoding="utf-8-sig")
if summary.empty:
    raise SystemExit("no transfer metrics found")
best = str(summary.iloc[0]["fraction"])
(root / "best_transfer_fraction.txt").write_text(best + "\n", encoding="utf-8")
print(best)
PY
)"
echo "[best_transfer_fraction] $BEST_FRAC"

echo "[ad] soil application-domain reports for paired test sets"
AD_DIR="outputs/ad/v2_0_0"
mkdir -p "$AD_DIR"
"$PYTHON_BIN" -m qsar_tl.cli build-ad-report \
  --config "$CONFIG_PATH" \
  --db "$DB_PATH" \
  --source-table aggregated_task_records_soil_ptox_qc \
  --split-name "SoilPtoxQC2_C_low_${BEST_FRAC}" \
  --out "$AD_DIR/SoilPtoxQC2_C_low_${BEST_FRAC}.csv"

"$PYTHON_BIN" -m qsar_tl.cli build-ad-report \
  --config "$CONFIG_PATH" \
  --db "$DB_PATH" \
  --source-table aggregated_task_records_soil_mg_kg_qc \
  --split-name SoilMgkgQC2_C_chemical_holdout_8_2 \
  --out "$AD_DIR/SoilMgkgQC2_C_chemical_holdout_8_2.csv"

echo "[phase3] transfer diagnostics for best fraction"
run_train "outputs/experiments/v2_0_0_transfer_diagnostics" "v2.0.0" "reclean_v2_transfer_${BEST_FRAC}_finetune40_cosine" \
  "aggregated_task_records_aquatic_soil_ptox_qc" "M_v2_aquatic_to_soil_ptox_adapt_C_${BEST_FRAC}" \
  "$PRETRAIN_EPOCHS" "$BATCH_SIZE" "$LEARNING_RATE" cosine \
  --finetune-epochs 40 \
  --finetune-learning-rate "$FINETUNE_LR" \
  --finetune-batch-size "$BATCH_SIZE" \
  --finetune-scheduler cosine \
  --finetune-freeze none

run_train "outputs/experiments/v2_0_0_transfer_diagnostics" "v2.0.0" "reclean_v2_transfer_${BEST_FRAC}_finetune20_constant" \
  "aggregated_task_records_aquatic_soil_ptox_qc" "M_v2_aquatic_to_soil_ptox_adapt_C_${BEST_FRAC}" \
  "$PRETRAIN_EPOCHS" "$BATCH_SIZE" "$LEARNING_RATE" cosine \
  --finetune-epochs "$FINETUNE_EPOCHS" \
  --finetune-learning-rate "$FINETUNE_LR" \
  --finetune-batch-size "$BATCH_SIZE" \
  --finetune-scheduler none \
  --finetune-freeze none

BASE_TRANSFER_DIR="$(run_dir "outputs/experiments/v2_0_0_transfer_low_soil" "v2.0.0" "reclean_v2_transfer_${BEST_FRAC}" "M_v2_aquatic_to_soil_ptox_adapt_C_${BEST_FRAC}")"
DIAG40_DIR="$(run_dir "outputs/experiments/v2_0_0_transfer_diagnostics" "v2.0.0" "reclean_v2_transfer_${BEST_FRAC}_finetune40_cosine" "M_v2_aquatic_to_soil_ptox_adapt_C_${BEST_FRAC}")"
DIAG20_DIR="$(run_dir "outputs/experiments/v2_0_0_transfer_diagnostics" "v2.0.0" "reclean_v2_transfer_${BEST_FRAC}_finetune20_constant" "M_v2_aquatic_to_soil_ptox_adapt_C_${BEST_FRAC}")"

run_explain "$BASE_TRANSFER_DIR" "aggregated_task_records_aquatic_soil_ptox_qc" \
  "$PROJECT_DIR/outputs/experiments/v2_0_0_transfer_diagnostics/explanations/baseline_${BEST_FRAC}"
run_explain "$DIAG40_DIR" "aggregated_task_records_aquatic_soil_ptox_qc" \
  "$PROJECT_DIR/outputs/experiments/v2_0_0_transfer_diagnostics/explanations/${BEST_FRAC}_finetune40_cosine"
run_explain "$DIAG20_DIR" "aggregated_task_records_aquatic_soil_ptox_qc" \
  "$PROJECT_DIR/outputs/experiments/v2_0_0_transfer_diagnostics/explanations/${BEST_FRAC}_finetune20_constant"

echo "[summarize] AD-stratified test metrics"
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import math
import pandas as pd

def metrics(group: pd.DataFrame) -> dict[str, float]:
    y = pd.to_numeric(group["y_true"], errors="coerce")
    pred = pd.to_numeric(group["y_pred"], errors="coerce")
    mask = y.notna() & pred.notna()
    y = y[mask]
    pred = pred[mask]
    if len(y) == 0:
        return {"n": 0, "r2": math.nan, "rmse": math.nan, "mae": math.nan}
    residual = y - pred
    ss_res = float((residual ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = math.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return {
        "n": int(len(y)),
        "r2": r2,
        "rmse": float((residual ** 2).mean() ** 0.5),
        "mae": float(residual.abs().mean()),
    }

root = Path("outputs/experiments/v2_0_0_transfer_low_soil")
best = (root / "best_transfer_fraction.txt").read_text(encoding="utf-8").strip()
ad_path = Path(f"outputs/ad/v2_0_0/SoilPtoxQC2_C_low_{best}.csv")
ad = pd.read_csv(ad_path, encoding="utf-8-sig")
ad["aggregate_id"] = ad["aggregate_id"].astype(str)
ad_cols = ["aggregate_id", "overall_in_domain", "ad_warning"]

run_specs = [
    (
        f"transfer_{best}",
        Path(f"outputs/experiments/v2_0_0_transfer_low_soil/v2.0.0_reclean_v2_transfer_{best}/deep/full/M_v2_aquatic_to_soil_ptox_adapt_C_{best}/predictions.csv"),
    ),
    (
        f"soilonly_{best}",
        Path(f"outputs/experiments/v2_0_0_transfer_low_soil/v2.0.0_reclean_v2_soilonly_{best}/deep/full/SoilPtoxQC2_C_low_{best}/predictions.csv"),
    ),
]
rows = []
for label, pred_path in run_specs:
    pred = pd.read_csv(pred_path)
    pred = pred[pred["split_part"].astype(str).eq("test")].copy()
    pred["aggregate_id"] = pred["aggregate_id"].astype(str)
    merged = pred.merge(ad[ad_cols], on="aggregate_id", how="left")
    merged["endpoint_family"] = merged["task_head"].astype(str).str.split("_", n=1).str[0]
    for keys, group in merged.groupby(["overall_in_domain", "ad_warning", "endpoint_family"], dropna=False):
        row = {"run_label": label, "overall_in_domain": keys[0], "ad_warning": keys[1], "endpoint_family": keys[2]}
        row.update(metrics(group))
        rows.append(row)
    row = {"run_label": label, "overall_in_domain": "all", "ad_warning": "all", "endpoint_family": "all"}
    row.update(metrics(merged))
    rows.append(row)
out = Path("outputs/experiments/v2_0_0_transfer_diagnostics/ad_stratified_test_metrics.csv")
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
print("[ad_metrics]", out)
PY

echo "[done] $(date -Is)"
echo "[outputs]"
echo "  outputs/experiments/v2_0_0_smoke"
echo "  outputs/experiments/v2_0_0_main_retrain"
echo "  outputs/experiments/v2_0_0_transfer_low_soil"
echo "  outputs/experiments/v2_0_0_transfer_diagnostics"
echo "  outputs/ad/v2_0_0"
echo "[log] $MASTER_LOG"
