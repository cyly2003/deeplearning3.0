#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-formal}"
case "$MODE" in
  smoke|formal) ;;
  *) echo "Usage: $0 [smoke|formal]" >&2; exit 2 ;;
esac

PYTHON="${PYTHON:-/opt/anaconda3/envs/qsar-ph3/bin/python}"
CONFIG="${CONFIG:-configs/experiment.remote.easyai.yaml}"
DB="${DB:-outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite}"
SOURCE_TABLE="aggregated_task_records_ptox_soil_mass_molar_qc"
SPLIT="M_v1_2_40_ptox_to_soil_molkg_B_random_8_2"
BASELINE_ROOT="outputs/experiments/v1_2_40_paired_mass_molar_matrix_remote"
OUT_ROOT="${OUT_ROOT:-outputs/experiments/v1_2_41_three_stage_optimization_matrix_remote}"
SUMMARY_DIR="${SUMMARY_DIR:-outputs/experiments/v1_2_41_three_stage_optimization_matrix_summary}"
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"
SCREEN_SEEDS=(3407 42)
FINAL_SEEDS=(42 2042 3407 8417)
EXPANSION_SEEDS=(2042 8417)
CANDIDATES=(S1 S2 S3)

mkdir -p "$OUT_ROOT" "$SUMMARY_DIR" outputs/logs

# Keep all long transfer controllers mutually exclusive on the single GPU.
exec 8>outputs/logs/v1_2_41_three_stage_optimization_matrix.lock
if ! flock -n 8; then
  echo "[blocked] v1.2.41 matrix is already queued or running" >&2
  exit 75
fi
exec 9>outputs/logs/v1_2_40_paired_mass_molar_matrix.lock
flock 9
exec 7>outputs/logs/v1_2_39_transfer_optimization_matrix.lock
flock 7

if [[ "$MODE" == "smoke" ]]; then
  STAGE1_EPOCHS=1
  STAGE2_EPOCHS=1
  STAGE3_EPOCHS=2
  SCREEN_SEEDS=(42)
  CANDIDATES=(S1)
  OUT_ROOT="${OUT_ROOT}_smoke"
  mkdir -p "$OUT_ROOT"
else
  STAGE1_EPOCHS=30
  STAGE2_EPOCHS=20
  STAGE3_EPOCHS=40
fi

run_candidate() {
  local candidate="$1" seed="$2" run_name run_dir log
  local -a candidate_args
  run_name="${candidate}_seed${seed}"
  run_dir="$OUT_ROOT/v1.2.41_${run_name}/deep/full/$SPLIT"
  log="outputs/logs/v1_2_41_${candidate}_seed${seed}.log"
  if [[ -s "$run_dir/predictions.csv" && -s "$run_dir/manifest.json" && -s "$run_dir/best_model.pt" ]]; then
    if [[ "$MODE" == "formal" ]]; then
      "$PYTHON" scripts/validate_v1_2_41_three_stage_run.py \
        --run-dir "$run_dir" \
        --candidate "$candidate"
    fi
    echo "[skip-existing] candidate=$candidate seed=$seed run_dir=$run_dir"
    return 0
  fi

  case "$candidate" in
    S1)
      candidate_args=(--no-swa --finetune-mgkg-mse-loss-weight 0)
      ;;
    S2)
      candidate_args=(
        --swa
        --swa-phase finetune_mgkg
        --swa-start-epoch 31
        --finetune-mgkg-mse-loss-weight 0
      )
      ;;
    S3)
      candidate_args=(
        --swa
        --swa-phase finetune_mgkg
        --swa-start-epoch 31
        --finetune-mgkg-mse-loss-weight 0.3
      )
      ;;
    *) echo "Unknown candidate: $candidate" >&2; return 2 ;;
  esac

  echo "[candidate_start] candidate=$candidate seed=$seed time=$(date -Is)"
  "$PYTHON" -m qsar_tl.training.train \
    --config "$CONFIG" \
    --db "$DB" \
    --out-dir "$OUT_ROOT" \
    --run-version v1.2.41 \
    --run-name-zh "$run_name" \
    --source-table "$SOURCE_TABLE" \
    --split-name "$SPLIT" \
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
    --finetune-mgkg-trunk-learning-rate 0.0001 \
    --finetune-freeze none \
    --finetune-mgkg-freeze none \
    --finetune-mgkg-head-only-epochs 0 \
    --finetune-mgkg-replay-fraction 0 \
    --finetune-mgkg-toxicity-bin-loss-weight 0 \
    --no-finetune-mgkg-early-stopping \
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
    --device cuda:0 \
    --ablation full \
    "${candidate_args[@]}" >"$log" 2>&1

  if [[ "$MODE" == "formal" ]]; then
    "$PYTHON" scripts/validate_v1_2_41_three_stage_run.py \
      --run-dir "$run_dir" \
      --candidate "$candidate"
  fi
  echo "[candidate_done] candidate=$candidate seed=$seed time=$(date -Is)"
}

run_batch() {
  local candidate="$1"
  shift
  local seeds=("$@") active=() failed=0 seed pid
  for seed in "${seeds[@]}"; do
    run_candidate "$candidate" "$seed" &
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
    echo "[batch_failed] candidate=$candidate time=$(date -Is)" >&2
    return 1
  fi
}

for candidate in "${CANDIDATES[@]}"; do
  run_batch "$candidate" "${SCREEN_SEEDS[@]}"
done

if [[ "$MODE" == "smoke" ]]; then
  echo "[smoke_complete] time=$(date -Is)"
  exit 0
fi

"$PYTHON" scripts/summarize_v1_2_41_three_stage_optimization.py \
  --root "$OUT_ROOT" \
  --baseline-root "$BASELINE_ROOT" \
  --output-dir "$SUMMARY_DIR" \
  --screen-seeds "${SCREEN_SEEDS[@]}" \
  --final-seeds "${FINAL_SEEDS[@]}" \
  --selection-only

WINNER="$($PYTHON -c 'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")).get("selected_candidate"); print(value or "")' "$SUMMARY_DIR/selected_candidate.json")"
if [[ -z "$WINNER" ]]; then
  echo "[matrix_complete_no_winner] time=$(date -Is) summary=$SUMMARY_DIR/selected_candidate.json"
  exit 0
fi

echo "[winner_locked] candidate=$WINNER source=validation_only time=$(date -Is)"
run_batch "$WINNER" "${EXPANSION_SEEDS[@]}"

"$PYTHON" scripts/summarize_v1_2_41_three_stage_optimization.py \
  --root "$OUT_ROOT" \
  --baseline-root "$BASELINE_ROOT" \
  --output-dir "$SUMMARY_DIR" \
  --screen-seeds "${SCREEN_SEEDS[@]}" \
  --final-seeds "${FINAL_SEEDS[@]}"

echo "[matrix_complete] time=$(date -Is) winner=$WINNER summary=$SUMMARY_DIR/ensemble_metrics.csv"
