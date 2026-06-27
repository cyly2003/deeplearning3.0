#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-check}"

PROJECT_DIR="${PROJECT_DIR:-/home/easyai/DL1/ecotox_qsar_transfer}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
RANDOM_SCRIPT="${RANDOM_SCRIPT:-scripts/run_v1_2_15_random_split_policy_remote.sh}"
ABLATION_SCRIPT="${ABLATION_SCRIPT:-scripts/run_v1_2_18_mainline_ablation_remote.sh}"
SINGLE_DOMAIN_SCRIPT="${SINGLE_DOMAIN_SCRIPT:-scripts/run_v1_2_19_single_domain_bce_remote.sh}"

RANDOM_ROOT="${RANDOM_ROOT:-outputs/experiments/v1_2_15_random_split_policy_formal_remote}"
RANDOM_SUMMARY="${RANDOM_SUMMARY:-outputs/experiments/v1_2_15_random_split_policy_formal_remote_summary}"
REQUIRED_RANDOM_SEEDS="${REQUIRED_RANDOM_SEEDS:-3042 4042}"
ENSEMBLE_SEEDS="${ENSEMBLE_SEEDS:-42 1042 2042 3042 4042}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-600}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-0}"
ABLATION_MODE="${ABLATION_MODE:-matrix}"
SINGLE_DOMAIN_MODE="${SINGLE_DOMAIN_MODE:-matrix}"
RUN_ABLATION="${RUN_ABLATION:-1}"
RUN_SINGLE_DOMAIN="${RUN_SINGLE_DOMAIN:-1}"

random8_prediction_path() {
  local seed="$1"
  echo "${RANDOM_ROOT}/v1.2.15_transfer_f100_anchor_random8_2_seed${seed}_cebin_lw0025_censored_w0p01/deep/full/M_v2_aquatic_to_soil_ptox_adapt_B_random_8_2_f100/predictions.csv"
}

random5fold_prediction_path() {
  local seed="$1"
  local fold="$2"
  echo "${RANDOM_ROOT}/v1.2.15_transfer_f100_anchor_random5fold_fold${fold}_seed${seed}_cebin_lw0025_censored_w0p01/deep/full/M_v2_aquatic_to_soil_ptox_adapt_E_random_5fold_fold${fold}_f100/predictions.csv"
}

count_random_outputs() {
  local seed fold path complete=0 missing=0 expected=0
  for seed in $REQUIRED_RANDOM_SEEDS; do
    path="$(random8_prediction_path "$seed")"
    expected=$((expected + 1))
    if [[ -s "$path" ]]; then
      complete=$((complete + 1))
    else
      missing=$((missing + 1))
      echo "[missing-random] seed=${seed} policy=random8_2"
    fi
    for fold in 1 2 3 4 5; do
      path="$(random5fold_prediction_path "$seed" "$fold")"
      expected=$((expected + 1))
      if [[ -s "$path" ]]; then
        complete=$((complete + 1))
      else
        missing=$((missing + 1))
        echo "[missing-random] seed=${seed} policy=random5fold fold=${fold}"
      fi
    done
  done
  echo "[random-output-count] complete=${complete} missing=${missing} expected=${expected}"
  [[ "$complete" -eq "$expected" && "$missing" -eq 0 ]]
}

summary_has_required_seeds() {
  local combined="${RANDOM_SUMMARY}/split_policy_ensemble_combined_summary.csv"
  local required
  required="$(echo "$ENSEMBLE_SEEDS" | tr ' ' ';')"
  [[ -s "$combined" ]] && grep -q "$required" "$combined"
}

ensure_random_summary() {
  if summary_has_required_seeds; then
    echo "[random-summary] five_seed_ready"
    return 0
  fi
  echo "[random-summary] stale_or_missing; rebuilding ensemble summary"
  ENSEMBLE_SEEDS="$ENSEMBLE_SEEDS" bash "$RANDOM_SCRIPT" ensemble
  summary_has_required_seeds
}

random_ready() {
  count_random_outputs && ensure_random_summary
}

print_status() {
  echo "[time] $(date -Is)"
  count_random_outputs || true
  if summary_has_required_seeds; then
    echo "[summary-seeds] five_seed_ready"
  else
    echo "[summary-seeds] stale_or_incomplete"
  fi
  pgrep -af "run_v1_2_15_random_split_policy_remote|qsar_tl.training.train" \
    | grep -v "pgrep -af" \
    | awk '{ if (length($0) > 220) print substr($0, 1, 220) "..."; else print }' \
    | head -8 || true
}

wait_for_random() {
  local start now elapsed
  start="$(date +%s)"
  while true; do
    if random_ready; then
      echo "[wait-random] complete"
      return 0
    fi
    now="$(date +%s)"
    elapsed=$((now - start))
    if [[ "$MAX_WAIT_SECONDS" -gt 0 && "$elapsed" -ge "$MAX_WAIT_SECONDS" ]]; then
      echo "[wait-random] timeout elapsed=${elapsed}s max=${MAX_WAIT_SECONDS}s" >&2
      return 124
    fi
    echo "[wait-random] incomplete elapsed=${elapsed}s; sleeping ${CHECK_INTERVAL_SECONDS}s"
    sleep "$CHECK_INTERVAL_SECONDS"
  done
}

run_followup() {
  if [[ "$RUN_ABLATION" == "1" ]]; then
    echo "[followup-start] v1.2.18 ablation mode=${ABLATION_MODE}"
    bash "$ABLATION_SCRIPT" "$ABLATION_MODE"
    echo "[followup-done] v1.2.18 ablation"
  fi
  if [[ "$RUN_SINGLE_DOMAIN" == "1" ]]; then
    echo "[followup-start] v1.2.19 single-domain mode=${SINGLE_DOMAIN_MODE}"
    bash "$SINGLE_DOMAIN_SCRIPT" "$SINGLE_DOMAIN_MODE"
    echo "[followup-done] v1.2.19 single-domain"
  fi
}

case "$MODE" in
  check)
    print_status
    ;;
  wait)
    wait_for_random
    ;;
  followup)
    random_ready
    run_followup
    ;;
  wait_then_followup)
    wait_for_random
    run_followup
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use check, wait, followup, or wait_then_followup." >&2
    exit 2
    ;;
esac
