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
DIRECT_PREFIX="M_v1_2_42_e_oof_direct"
TRANSFER_PREFIX="M_v1_2_42_e_oof_transfer"
BASELINE_ROOT="outputs/experiments/v1_2_40_paired_mass_molar_matrix_remote"
OUT_ROOT="${OUT_ROOT:-outputs/experiments/v1_2_42_e_series_oof_remote}"
SUMMARY_DIR="${SUMMARY_DIR:-outputs/experiments/v1_2_42_e_series_summary}"
AUDIT_DIR="outputs/audits/v1_2_42_e_series"
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"
FOLDS=(1 2 3 4 5)
SCREEN_SEEDS=(42 3407)
FINAL_SEEDS=(42 2042 3407 8417)
EXPANSION_SEEDS=(2042 8417)

mkdir -p "$OUT_ROOT" "$SUMMARY_DIR" "$AUDIT_DIR" outputs/logs

build_splits() {
  bash scripts/run_v1_2_40_paired_mass_molar_matrix_remote.sh build
  "$PYTHON" scripts/build_v1_2_42_e_series_oof_splits.py \
    --db "$DB" \
    --source-table "$SOURCE_TABLE" \
    --parent-transfer-split "$PARENT_SPLIT" \
    --direct-prefix "$DIRECT_PREFIX" \
    --transfer-prefix "$TRANSFER_PREFIX" \
    --folds 5 \
    --seed 424242 \
    --audit-csv "$AUDIT_DIR/oof_split_audit.csv" \
    --summary-json "$AUDIT_DIR/oof_split_summary.json"
}

if [[ "$MODE" == "build" ]]; then
  build_splits
  exit 0
fi

# The remote host has one GPU. Two concurrent base learners saturated it in the
# preceding matrices without the memory pressure seen at five-way concurrency.
exec 8>outputs/logs/v1_2_42_e_series.lock
if ! flock -n 8; then
  echo "[blocked] v1.2.42 E-series is already queued or running" >&2
  exit 75
fi
exec 9>outputs/logs/v1_2_41_three_stage_optimization_matrix.lock
flock 9
exec 7>outputs/logs/v1_2_40_paired_mass_molar_matrix.lock
flock 7
exec 6>outputs/logs/v1_2_39_transfer_optimization_matrix.lock
flock 6

build_splits

if [[ "$MODE" == "smoke" ]]; then
  STAGE1_EPOCHS=1
  STAGE2_EPOCHS=1
  STAGE3_EPOCHS=1
  OUT_ROOT="${OUT_ROOT}_smoke"
  mkdir -p "$OUT_ROOT"
else
  STAGE1_EPOCHS=30
  STAGE2_EPOCHS=20
  STAGE3_EPOCHS=30
fi

run_direct() {
  local seed="$1" fold="$2" split run_name run_dir
  split="${DIRECT_PREFIX}_fold${fold}"
  run_name="OOF_D_seed${seed}_fold${fold}"
  run_dir="$OUT_ROOT/v1.2.42_${run_name}/deep/full/$split"
  if [[ -s "$run_dir/predictions.csv" && -s "$run_dir/manifest.json" && -s "$run_dir/best_model.pt" ]]; then
    "$PYTHON" scripts/validate_v1_2_42_oof_run.py --run-dir "$run_dir" --architecture D
    echo "[skip-existing] architecture=D seed=$seed fold=$fold run_dir=$run_dir"
    return 0
  fi
  "$PYTHON" -m qsar_tl.training.train \
    --config "$CONFIG" \
    --db "$DB" \
    --out-dir "$OUT_ROOT" \
    --run-version v1.2.42 \
    --run-name-zh "$run_name" \
    --source-table "$SOURCE_TABLE" \
    --split-name "$split" \
    --seed "$seed" \
    --epochs "$STAGE1_EPOCHS" \
    --finetune-epochs 0 \
    --finetune-mgkg-epochs 0 \
    --batch-size 512 \
    --scheduler cosine \
    --learning-rate 0.0005 \
    --weight-decay 0.00001 \
    --dropout 0.10 \
    --target-standardization per_task_target \
    --head-routing task_target \
    --no-medium-adapters \
    --early-stopping \
    --early-stopping-patience 15 \
    --early-stopping-min-delta 0 \
    --monitor-split internal_train_fraction \
    --validation-fraction 0.2 \
    --validation-seed 17073 \
    --source-weighting-method none \
    --no-toxicity-binning \
    --no-censored-loss \
    --no-effect-level-weighting \
    --metric-min-n 5 \
    --prediction-split-parts valid \
    --device cuda:0 \
    --ablation full
  "$PYTHON" scripts/validate_v1_2_42_oof_run.py --run-dir "$run_dir" --architecture D
}

run_transfer() {
  local seed="$1" fold="$2" split run_name run_dir
  split="${TRANSFER_PREFIX}_fold${fold}"
  run_name="OOF_X_seed${seed}_fold${fold}"
  run_dir="$OUT_ROOT/v1.2.42_${run_name}/deep/full/$split"
  if [[ -s "$run_dir/predictions.csv" && -s "$run_dir/manifest.json" && -s "$run_dir/best_model.pt" ]]; then
    "$PYTHON" scripts/validate_v1_2_42_oof_run.py --run-dir "$run_dir" --architecture X
    echo "[skip-existing] architecture=X seed=$seed fold=$fold run_dir=$run_dir"
    return 0
  fi
  "$PYTHON" -m qsar_tl.training.train \
    --config "$CONFIG" \
    --db "$DB" \
    --out-dir "$OUT_ROOT" \
    --run-version v1.2.42 \
    --run-name-zh "$run_name" \
    --source-table "$SOURCE_TABLE" \
    --split-name "$split" \
    --seed "$seed" \
    --epochs "$STAGE1_EPOCHS" \
    --finetune-epochs "$STAGE2_EPOCHS" \
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
    --finetune-mgkg-mse-loss-weight 0 \
    --finetune-mgkg-early-stopping \
    --no-mgkg-residual-adapter \
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
    --finetune-mgkg-monitor-split internal_finetune_fraction \
    --finetune-mgkg-validation-fraction 0.2 \
    --finetune-mgkg-validation-seed 42 \
    --source-weighting-method tanimoto_to_finetune \
    --source-weighting-alpha 1 \
    --source-weight-cache-dir outputs/cache/source_weights \
    --toxicity-binning \
    --toxicity-binning-mode aux_classification \
    --toxicity-binning-scheme authority_v1 \
    --toxicity-binning-loss-weight 0.025 \
    --no-censored-loss \
    --no-effect-level-weighting \
    --metric-min-n 5 \
    --prediction-split-parts valid \
    --device cuda:0 \
    --ablation full
  "$PYTHON" scripts/validate_v1_2_42_oof_run.py --run-dir "$run_dir" --architecture X
}

run_cell() {
  local architecture="$1" seed="$2" fold="$3" log
  log="outputs/logs/v1_2_42_OOF_${architecture}_seed${seed}_fold${fold}.log"
  echo "[oof_start] architecture=$architecture seed=$seed fold=$fold time=$(date -Is)"
  if [[ "$architecture" == "D" ]]; then
    run_direct "$seed" "$fold" >"$log" 2>&1
  else
    run_transfer "$seed" "$fold" >"$log" 2>&1
  fi
  echo "[oof_done] architecture=$architecture seed=$seed fold=$fold time=$(date -Is)"
}

run_jobs_parallel() {
  local jobs=("$@") active=() failed=0 job architecture seed fold pid
  for job in "${jobs[@]}"; do
    IFS=: read -r architecture seed fold <<<"$job"
    run_cell "$architecture" "$seed" "$fold" &
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
    echo "[matrix_failed] time=$(date -Is)" >&2
    return 1
  fi
}

if [[ "$MODE" == "smoke" ]]; then
  run_jobs_parallel "D:42:1" "X:42:1"
  echo "[smoke_complete] time=$(date -Is)"
  exit 0
fi

SCREEN_JOBS=()
for seed in "${SCREEN_SEEDS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    SCREEN_JOBS+=("D:${seed}:${fold}" "X:${seed}:${fold}")
  done
done
run_jobs_parallel "${SCREEN_JOBS[@]}"

"$PYTHON" scripts/summarize_v1_2_42_e_series.py \
  --oof-root "$OUT_ROOT" \
  --baseline-root "$BASELINE_ROOT" \
  --output-dir "$SUMMARY_DIR" \
  --screen-seeds "${SCREEN_SEEDS[@]}" \
  --final-seeds "${FINAL_SEEDS[@]}" \
  --folds 5 \
  --selection-only

WINNER="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("selected_candidate") or "")' "$SUMMARY_DIR/selected_candidate.json")"
if [[ -z "$WINNER" ]]; then
  echo "[matrix_complete_no_winner] time=$(date -Is) summary=$SUMMARY_DIR/selected_candidate.json"
  exit 0
fi

echo "[winner_locked] candidate=$WINNER source=oof_validation_only time=$(date -Is)"
EXPANSION_JOBS=()
for seed in "${EXPANSION_SEEDS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    EXPANSION_JOBS+=("D:${seed}:${fold}" "X:${seed}:${fold}")
  done
done
run_jobs_parallel "${EXPANSION_JOBS[@]}"

"$PYTHON" scripts/summarize_v1_2_42_e_series.py \
  --oof-root "$OUT_ROOT" \
  --baseline-root "$BASELINE_ROOT" \
  --output-dir "$SUMMARY_DIR" \
  --screen-seeds "${SCREEN_SEEDS[@]}" \
  --final-seeds "${FINAL_SEEDS[@]}" \
  --folds 5

echo "[matrix_complete] time=$(date -Is) winner=$WINNER summary=$SUMMARY_DIR/ensemble_metrics.csv"
