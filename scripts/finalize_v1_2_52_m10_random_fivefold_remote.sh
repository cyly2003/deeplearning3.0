#!/usr/bin/env bash
set -euo pipefail

# Wait for the v1.2.52 training lock to be released, then summarize only if the
# formal controller recorded a clean matrix_complete node.

PYTHON="${PYTHON:-/opt/anaconda3/envs/qsar-ph3/bin/python}"
LOCK="outputs/logs/v1_2_52_m10_random_fivefold_remaining_seeds.lock"
FORMAL_LOG="outputs/logs/v1_2_52_m10_random_fivefold_remaining_seeds_formal.log"
SUMMARY_LOG="outputs/logs/v1_2_52_m10_random_fivefold_four_seed_summary.log"

mkdir -p outputs/logs
exec 9>"$LOCK"
flock 9

if ! grep -q '^\[matrix_complete\] mode=formal' "$FORMAL_LOG"; then
  echo "[summary_blocked] v1.2.52 formal matrix did not finish cleanly" | tee "$SUMMARY_LOG"
  exit 1
fi

echo "[summary_start] time=$(date -Is)" | tee "$SUMMARY_LOG"
"$PYTHON" scripts/summarize_v1_2_52_m10_random_fivefold_four_seed.py \
  >>"$SUMMARY_LOG" 2>&1
echo "[summary_complete] time=$(date -Is)" | tee -a "$SUMMARY_LOG"
