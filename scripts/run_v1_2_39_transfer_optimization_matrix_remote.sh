#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-/opt/anaconda3/envs/qsar-ph3/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/experiments/v1_2_39_transfer_optimization_matrix_remote}"
SUMMARY_DIR="${SUMMARY_DIR:-outputs/experiments/v1_2_39_transfer_optimization_matrix_summary}"
RUN_MULTI_SEED="${RUN_MULTI_SEED:-1}"
SCREEN_SEED="${SCREEN_SEED:-42}"
REFIT_SEEDS=(2042 3407 8417)

mkdir -p "$OUT_ROOT" "$SUMMARY_DIR" outputs/logs
exec 8>outputs/logs/v1_2_39_transfer_optimization_matrix.lock
if ! flock -n 8; then
  echo "[blocked] transfer optimization matrix is already queued or running" >&2
  exit 75
fi

# Do not overlap the legacy queued head-only control on the single GPU.
exec 9>outputs/logs/headsonly_control_queue.lock
flock 9

run_cell() {
  local cell="$1"
  local seed="$2"
  local epochs freeze batch head_only trunk_lr replay adapter bottleneck
  case "$cell" in
    T0) epochs=20; freeze=none;        batch=512; head_only=0; trunk_lr=0;       replay=0;    adapter=0; bottleneck=64 ;;
    T1) epochs=20; freeze=heads_only;  batch=512; head_only=0; trunk_lr=0;       replay=0;    adapter=0; bottleneck=64 ;;
    T2) epochs=30; freeze=last_trunk;  batch=512; head_only=5; trunk_lr=0.0005;  replay=0;    adapter=0; bottleneck=64 ;;
    T3) epochs=30; freeze=last_trunk;  batch=256; head_only=5; trunk_lr=0.00003; replay=0;    adapter=0; bottleneck=64 ;;
    T4) epochs=30; freeze=last_trunk;  batch=256; head_only=5; trunk_lr=0.00003; replay=0.25; adapter=0; bottleneck=64 ;;
    T5) epochs=30; freeze=last_trunk;  batch=256; head_only=5; trunk_lr=0.00003; replay=0;    adapter=1; bottleneck=64 ;;
    *) echo "Unknown matrix cell: $cell" >&2; return 2 ;;
  esac
  echo "[matrix_start] cell=$cell seed=$seed time=$(date -Is)"
  env \
    MODEL_SEED_OVERRIDE="$seed" \
    OUT_ROOT_OVERRIDE="$OUT_ROOT" \
    RUN_NAME_OVERRIDE="transfer_matrix_${cell}_seed${seed}" \
    SOIL_MGKG_EPOCHS="$epochs" \
    FINETUNE_MGKG_FREEZE_OVERRIDE="$freeze" \
    FINETUNE_MGKG_BATCH_SIZE_OVERRIDE="$batch" \
    FINETUNE_MGKG_LEARNING_RATE_OVERRIDE=0.0005 \
    FINETUNE_MGKG_TRUNK_LEARNING_RATE_OVERRIDE="$trunk_lr" \
    FINETUNE_MGKG_HEAD_ONLY_EPOCHS_OVERRIDE="$head_only" \
    FINETUNE_MGKG_REPLAY_FRACTION_OVERRIDE="$replay" \
    FINETUNE_MGKG_TOXICITY_BIN_LOSS_WEIGHT_OVERRIDE=0 \
    MGKG_RESIDUAL_ADAPTER_OVERRIDE="$adapter" \
    MGKG_RESIDUAL_ADAPTER_BOTTLENECK_OVERRIDE="$bottleneck" \
    bash scripts/run_v1_2_39_ptox_to_soil_mgkg_3stage_remote.sh formal
  echo "[matrix_done] cell=$cell seed=$seed time=$(date -Is)"
}

for cell in T0 T1 T2 T3 T4 T5; do
  run_cell "$cell" "$SCREEN_SEED"
done

SCREEN_SUMMARY="$SUMMARY_DIR/seed${SCREEN_SEED}_validation_ranking.csv"
"$PYTHON" scripts/summarize_v1_2_39_transfer_optimization_matrix.py \
  --root "$OUT_ROOT" \
  --output "$SCREEN_SUMMARY" \
  --seed "$SCREEN_SEED"

if [[ "$RUN_MULTI_SEED" == "1" ]]; then
  mapfile -t TOP_CELLS < <(
    "$PYTHON" -c 'import csv,sys; rows=list(csv.DictReader(open(sys.argv[1],encoding="utf-8-sig"))); print("\n".join(row["matrix_cell"] for row in rows[:2]))' "$SCREEN_SUMMARY"
  )
  for cell in "${TOP_CELLS[@]}"; do
    for seed in "${REFIT_SEEDS[@]}"; do
      run_cell "$cell" "$seed"
    done
  done
fi

echo "[matrix_complete] time=$(date -Is) summary=$SCREEN_SUMMARY"
