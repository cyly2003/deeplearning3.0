#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

V52_CONTROLLER_PID="${V52_CONTROLLER_PID:-1390696}"
PRIORITY_LOG="outputs/logs/v1_2_54_priority_then_resume_v1_2_52.log"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
resumed=0

mkdir -p outputs/logs

resume_fivefold() {
  if (( resumed == 1 )); then
    return 0
  fi
  if kill -0 "$V52_CONTROLLER_PID" 2>/dev/null; then
    controller_cmd="$(ps -p "$V52_CONTROLLER_PID" -o cmd= || true)"
    if [[ "$controller_cmd" == *"run_v1_2_52_m10_random_fivefold_remaining_seeds_remote.sh"* ]]; then
      kill -CONT "$V52_CONTROLLER_PID"
      echo "[v1_2_52_resumed] pid=$V52_CONTROLLER_PID time=$(date -Is)" | tee -a "$PRIORITY_LOG"
      resumed=1
      return 0
    fi
    echo "[resume_refused_pid_command_mismatch] pid=$V52_CONTROLLER_PID cmd=$controller_cmd time=$(date -Is)" | tee -a "$PRIORITY_LOG"
    return 1
  fi
  echo "[v1_2_52_controller_absent] pid=$V52_CONTROLLER_PID time=$(date -Is)" | tee -a "$PRIORITY_LOG"
  resumed=1
}

trap resume_fivefold EXIT

if ! kill -0 "$V52_CONTROLLER_PID" 2>/dev/null; then
  echo "[missing_paused_v1_2_52_controller] pid=$V52_CONTROLLER_PID time=$(date -Is)" | tee -a "$PRIORITY_LOG"
  exit 1
fi
controller_state="$(ps -p "$V52_CONTROLLER_PID" -o state= | tr -d ' ')"
if [[ "$controller_state" != T* ]]; then
  echo "[v1_2_52_not_stopped] pid=$V52_CONTROLLER_PID state=$controller_state time=$(date -Is)" | tee -a "$PRIORITY_LOG"
  exit 1
fi

while pgrep -f "qsar_tl.training.train.*--run-version v1.2.52" >/dev/null 2>&1; do
  echo "[wait_v1_2_52_active_children] time=$(date -Is)" | tee -a "$PRIORITY_LOG"
  sleep 30
done

echo "[priority_start] task=v1.2.54 attempts=$MAX_ATTEMPTS time=$(date -Is)" | tee -a "$PRIORITY_LOG"
success=0
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "[priority_attempt_start] attempt=$attempt time=$(date -Is)" | tee -a "$PRIORITY_LOG"
  if bash scripts/run_v1_2_54_m10_m00_target_data_learning_curve_remote.sh auto >>"$PRIORITY_LOG" 2>&1; then
    echo "[priority_attempt_complete] attempt=$attempt time=$(date -Is)" | tee -a "$PRIORITY_LOG"
    success=1
    break
  fi
  echo "[priority_attempt_failed] attempt=$attempt time=$(date -Is)" | tee -a "$PRIORITY_LOG"
  sleep 15
done

if (( success != 1 )); then
  echo "[priority_failed_after_retries] task=v1.2.54 time=$(date -Is)" | tee -a "$PRIORITY_LOG"
  exit 1
fi

summary="outputs/experiments/v1_2_54_m10_m00_target_data_learning_curve/summary/summary.json"
if [[ ! -s "$summary" ]]; then
  echo "[priority_summary_missing] path=$summary time=$(date -Is)" | tee -a "$PRIORITY_LOG"
  exit 1
fi

echo "[priority_complete] task=v1.2.54 summary=$summary time=$(date -Is)" | tee -a "$PRIORITY_LOG"
resume_fivefold
trap - EXIT
