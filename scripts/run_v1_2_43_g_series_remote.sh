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
SCREEN_SPLIT="M_v1_2_43_g_screen"
FINAL_SPLIT="M_v1_2_43_g_final"
BASELINE_ROOT="outputs/experiments/v1_2_40_paired_mass_molar_matrix_remote"
OUT_ROOT="${OUT_ROOT:-outputs/experiments/v1_2_43_g_series_remote}"
SUMMARY_DIR="${SUMMARY_DIR:-outputs/experiments/v1_2_43_g_series_summary}"
AUDIT_DIR="outputs/audits/v1_2_43_g_series"
CACHE_ROOT="${CACHE_ROOT:-outputs/cache/v1_2_43_g_stage2_init}"
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"
# Keep the original 100/30 train/evaluation safeguards, while admitting task
# routes with 150--199 visible records. This keeps the three screen-only soil
# routes (158, 190, and 191 records after final-test withholding) trainable
# without broadly admitting sparse heads.
TASK_FILTER_MIN_TOTAL="${TASK_FILTER_MIN_TOTAL:-150}"
TASK_FILTER_MIN_TRAIN="${TASK_FILTER_MIN_TRAIN:-100}"
TASK_FILTER_MIN_EVAL="${TASK_FILTER_MIN_EVAL:-30}"
SCREEN_SEEDS=(42 3407)
FINAL_SEEDS=(42 2042 3407 8417)
CANDIDATES=(G0 G1 G2 G3)

mkdir -p "$OUT_ROOT" "$SUMMARY_DIR" "$AUDIT_DIR" "$CACHE_ROOT" outputs/logs

build_split() {
  local phase="$1"
  local -a lock_args=()
  if [[ "$phase" == "final" ]]; then
    lock_args=(--winner-lock "$SUMMARY_DIR/winner_lock.json")
  fi
  "$PYTHON" scripts/build_v1_2_43_g_splits.py \
    --db "$DB" \
    --source-table "$SOURCE_TABLE" \
    --parent-split "$PARENT_SPLIT" \
    --old-baseline-root "$BASELINE_ROOT" \
    --screen-split "$SCREEN_SPLIT" \
    --final-split "$FINAL_SPLIT" \
    --phase "$phase" \
    "${lock_args[@]}" \
    --validation-seed 17073 \
    --quantile-bins 10 \
    --audit-csv "$AUDIT_DIR/${phase}_split_audit.csv" \
    --summary-json "$AUDIT_DIR/${phase}_split_summary.json"
}

if [[ "$MODE" == "build" ]]; then
  # Build only the no-test screening split. The final split is deliberately
  # created after validation-only winner locking in formal mode.
  build_split screen
  exit 0
fi

# One matrix controller owns the single GPU at a time. Two cells may share the
# GPU within this controller; this was the stable throughput point in v1.2.40-42.
exec 8>outputs/logs/v1_2_43_g_series.lock
if ! flock -n 8; then
  echo "[blocked] v1.2.43 G-series is already queued or running" >&2
  exit 75
fi
exec 9>outputs/logs/v1_2_42_e_series.lock
flock 9
exec 7>outputs/logs/v1_2_41_three_stage_optimization_matrix.lock
flock 7
exec 6>outputs/logs/v1_2_40_paired_mass_molar_matrix.lock
flock 6
exec 5>outputs/logs/v1_2_39_transfer_optimization_matrix.lock
flock 5

build_split screen

if [[ "$MODE" == "smoke" ]]; then
  STAGE1_EPOCHS=1
  STAGE2_EPOCHS=1
  STAGE3_EPOCHS=1
  RUN_ROOT="${OUT_ROOT}_smoke"
  RUN_CACHE_ROOT="${CACHE_ROOT}_smoke"
  ACTIVE_SCREEN_SEEDS=(42)
  mkdir -p "$RUN_ROOT" "$RUN_CACHE_ROOT"
else
  STAGE1_EPOCHS=30
  STAGE2_EPOCHS=20
  STAGE3_EPOCHS=40
  RUN_ROOT="$OUT_ROOT"
  RUN_CACHE_ROOT="$CACHE_ROOT"
  ACTIVE_SCREEN_SEEDS=("${SCREEN_SEEDS[@]}")
fi

run_cell() {
  local phase="$1" candidate="$2" seed="$3"
  local split run_name run_dir log cache_path prediction_parts
  local split_summary
  local -a candidate_args checkpoint_args
  if [[ "$phase" == "screen" ]]; then
    split="$SCREEN_SPLIT"
    split_summary="$AUDIT_DIR/screen_split_summary.json"
    prediction_parts="finetune_mgkg_validation"
  else
    split="$FINAL_SPLIT"
    split_summary="$AUDIT_DIR/final_split_summary.json"
    prediction_parts="finetune_mgkg_validation test"
  fi
  run_name="${candidate}_${phase}_seed${seed}"
  run_dir="$RUN_ROOT/v1.2.43_${run_name}/deep/full/$split"
  log="outputs/logs/v1_2_43_${candidate}_${phase}_seed${seed}.log"
  cache_path="$RUN_CACHE_ROOT/${phase}_seed${seed}_stage2_init.pt"

  case "$candidate" in
    G0)
      candidate_args=(
        --no-finetune-mgkg-target-bin-sampling
        --finetune-mgkg-mse-loss-weight 0
        --no-finetune-mgkg-hierarchical-head
      )
      if [[ -s "$cache_path" ]]; then
        checkpoint_args=(--finetune-mgkg-init-checkpoint "$cache_path")
      else
        checkpoint_args=(--export-finetune-mgkg-init-checkpoint "$cache_path")
      fi
      ;;
    G1)
      candidate_args=(
        --finetune-mgkg-target-bin-sampling
        --finetune-mgkg-target-bins 10
        --finetune-mgkg-sampling-min-weight 0.5
        --finetune-mgkg-sampling-max-weight 2.0
        --finetune-mgkg-mse-loss-weight 0
        --no-finetune-mgkg-hierarchical-head
      )
      checkpoint_args=(--finetune-mgkg-init-checkpoint "$cache_path")
      ;;
    G2)
      candidate_args=(
        --finetune-mgkg-target-bin-sampling
        --finetune-mgkg-target-bins 10
        --finetune-mgkg-sampling-min-weight 0.5
        --finetune-mgkg-sampling-max-weight 2.0
        --finetune-mgkg-mse-loss-weight 0.15
        --no-finetune-mgkg-hierarchical-head
      )
      checkpoint_args=(--finetune-mgkg-init-checkpoint "$cache_path")
      ;;
    G3)
      candidate_args=(
        --no-finetune-mgkg-target-bin-sampling
        --finetune-mgkg-mse-loss-weight 0
        --finetune-mgkg-hierarchical-head
        --finetune-mgkg-hierarchical-family-tau 128
        --finetune-mgkg-hierarchical-task-tau 64
      )
      checkpoint_args=(--finetune-mgkg-init-checkpoint "$cache_path")
      ;;
    *) echo "Unknown G candidate: $candidate" >&2; return 2 ;;
  esac

  if [[ "$candidate" != "G0" && ! -s "$cache_path" ]]; then
    echo "[missing_stage2_cache] phase=$phase candidate=$candidate seed=$seed cache=$cache_path" >&2
    return 1
  fi
  if [[ -s "$run_dir/predictions.csv" && -s "$run_dir/manifest.json" && -s "$run_dir/best_model.pt" && -s "$cache_path" ]]; then
    local -a smoke_arg=()
    [[ "$MODE" == "smoke" ]] && smoke_arg=(--smoke)
    "$PYTHON" scripts/validate_v1_2_43_g_run.py \
      --run-dir "$run_dir" \
      --candidate "$candidate" \
      --phase "$phase" \
      --seed "$seed" \
      --split-name "$split" \
      --source-table "$SOURCE_TABLE" \
      --db "$DB" \
      --split-summary "$split_summary" \
      --stage2-cache "$cache_path" \
      "${smoke_arg[@]}"
    echo "[skip-existing] phase=$phase candidate=$candidate seed=$seed run_dir=$run_dir"
    return 0
  fi

  echo "[g_cell_start] phase=$phase candidate=$candidate seed=$seed time=$(date -Is)"
  # shellcheck disable=SC2086
  "$PYTHON" -m qsar_tl.training.train \
    --config "$CONFIG" \
    --db "$DB" \
    --out-dir "$RUN_ROOT" \
    --run-version v1.2.43 \
    --run-name-zh "$run_name" \
    --source-table "$SOURCE_TABLE" \
    --split-name "$split" \
    --task-filter-min-total "$TASK_FILTER_MIN_TOTAL" \
    --task-filter-min-train "$TASK_FILTER_MIN_TRAIN" \
    --task-filter-min-eval "$TASK_FILTER_MIN_EVAL" \
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
    --finetune-mgkg-freeze heads_only \
    --finetune-mgkg-head-only-epochs 0 \
    --finetune-mgkg-replay-fraction 0 \
    --finetune-mgkg-toxicity-bin-loss-weight 0 \
    --no-finetune-mgkg-early-stopping \
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
    --finetune-mgkg-validation-seed 17073 \
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
    --prediction-split-parts $prediction_parts \
    --device cuda:0 \
    --ablation full \
    "${candidate_args[@]}" \
    "${checkpoint_args[@]}" >"$log" 2>&1

  local -a smoke_arg=()
  [[ "$MODE" == "smoke" ]] && smoke_arg=(--smoke)
  "$PYTHON" scripts/validate_v1_2_43_g_run.py \
    --run-dir "$run_dir" \
    --candidate "$candidate" \
    --phase "$phase" \
    --seed "$seed" \
    --split-name "$split" \
    --source-table "$SOURCE_TABLE" \
    --db "$DB" \
    --split-summary "$split_summary" \
    --stage2-cache "$cache_path" \
    "${smoke_arg[@]}"
  echo "[g_cell_done] phase=$phase candidate=$candidate seed=$seed time=$(date -Is)"
}

run_batch() {
  local phase="$1" candidate="$2"
  shift 2
  local seeds=("$@") active=() failed=0 seed pid
  for seed in "${seeds[@]}"; do
    run_cell "$phase" "$candidate" "$seed" &
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
    echo "[g_batch_failed] phase=$phase candidate=$candidate time=$(date -Is)" >&2
    return 1
  fi
}

# G0 creates the same-seed stage-2 cache first. Every challenger then starts
# from that exact cache, isolating only its stage-3 intervention.
run_batch screen G0 "${ACTIVE_SCREEN_SEEDS[@]}"
for candidate in G1 G2 G3; do
  run_batch screen "$candidate" "${ACTIVE_SCREEN_SEEDS[@]}"
done

if [[ "$MODE" == "smoke" ]]; then
  echo "[smoke_complete] time=$(date -Is)"
  exit 0
fi

"$PYTHON" scripts/summarize_v1_2_43_g_series.py \
  --root "$OUT_ROOT" \
  --output-dir "$SUMMARY_DIR" \
  --screen-split-summary "$AUDIT_DIR/screen_split_summary.json" \
  --screen-seeds "${SCREEN_SEEDS[@]}" \
  --final-seeds "${FINAL_SEEDS[@]}" \
  --selection-only

WINNER="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("selected_candidate") or "")' "$SUMMARY_DIR/winner_lock.json")"
if [[ -z "$WINNER" ]]; then
  echo "[matrix_complete_no_winner] time=$(date -Is) summary=$SUMMARY_DIR/winner_lock.json"
  exit 0
fi

echo "[winner_locked] candidate=$WINNER source=fresh_validation_only time=$(date -Is)"
# Only now materialize a split containing the untouched outer test.
build_split final

# The final split has a separate fail-closed checkpoint contract. G0 therefore
# creates one final-phase stage-2 cache per seed before the winner reuses it.
run_batch final G0 "${FINAL_SEEDS[@]}"
run_batch final "$WINNER" "${FINAL_SEEDS[@]}"

"$PYTHON" scripts/summarize_v1_2_43_g_series.py \
  --root "$OUT_ROOT" \
  --output-dir "$SUMMARY_DIR" \
  --screen-split-summary "$AUDIT_DIR/screen_split_summary.json" \
  --screen-seeds "${SCREEN_SEEDS[@]}" \
  --final-seeds "${FINAL_SEEDS[@]}"

echo "[matrix_complete] time=$(date -Is) winner=$WINNER summary=$SUMMARY_DIR/ensemble_metrics.csv"
