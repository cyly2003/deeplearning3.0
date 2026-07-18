#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/easyai/DL1/ecotox_qsar_transfer}"
WAIT_PATTERN="${WAIT_PATTERN:-run_v1_2_33_molecular_signal_ablation_remote.sh}"
POLL_SECONDS="${POLL_SECONDS:-300}"

cd "$PROJECT_DIR"
mkdir -p outputs/logs

echo "[queue-start] $(date -Is) waiting_for=${WAIT_PATTERN} poll_seconds=${POLL_SECONDS}"
while pgrep -f "$WAIT_PATTERN" >/dev/null 2>&1; do
  echo "[queue-wait] $(date -Is) active_pattern=${WAIT_PATTERN}"
  sleep "$POLL_SECONDS"
done

echo "[queue-run] $(date -Is) starting graph-only priority"
bash scripts/run_v1_2_36_graph_only_remote.sh priority
echo "[queue-done] $(date -Is) graph-only priority completed"
