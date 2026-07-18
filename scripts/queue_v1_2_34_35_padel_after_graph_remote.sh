#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/easyai/DL1/ecotox_qsar_transfer}"
WAIT_PATTERN="${WAIT_PATTERN:-queue_v1_2_36_graph_only_after_v1_2_33_remote.sh}"
POLL_SECONDS="${POLL_SECONDS:-300}"

cd "$PROJECT_DIR"
mkdir -p outputs/logs

echo "[queue-start] $(date -Is) waiting_for=${WAIT_PATTERN} poll_seconds=${POLL_SECONDS}"
while pgrep -f "$WAIT_PATTERN" >/dev/null 2>&1; do
  echo "[queue-wait] $(date -Is) active_pattern=${WAIT_PATTERN}"
  sleep "$POLL_SECONDS"
done

if [[ ! -s outputs/features/molecular_features_padel_morgan512.jsonl ]]; then
  echo "[queue-error] missing PaDEL cache: outputs/features/molecular_features_padel_morgan512.jsonl" >&2
  exit 1
fi

echo "[queue-run] $(date -Is) starting v1.2.34 PaDEL raw"
bash scripts/run_v1_2_34_padel_molecular_signal_remote.sh priority

echo "[queue-run] $(date -Is) starting v1.2.35 PaDEL prior clustered"
CONFIG="configs/experiment.remote.easyai.padel_prior_clustered.yaml" \
RUN_VERSION="v1.2.35" \
EXPERIMENT_LABEL="padel_prior_clustered" \
OUT_ROOT="outputs/experiments/v1_2_35_padel_prior_clustered_remote" \
SUMMARY_OUT="outputs/experiments/v1_2_35_padel_prior_clustered_remote_summary" \
RUN_TIMES="outputs/logs/run_v1_2_35_padel_prior_clustered_times.csv" \
bash scripts/run_v1_2_34_padel_molecular_signal_remote.sh priority

echo "[queue-done] $(date -Is) PaDEL raw and prior clustered completed"
