#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"

cd /home/easyai/DL1/ecotox_qsar_transfer
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PYTHON_BIN="/opt/anaconda3/bin/python"
BASE_AUDIT_ROOT="outputs/experiments/v1_2_11_f100_seed_stability_remote/audits"
SUMMARY_OUT="outputs/experiments/v1_2_12_f100_5seed_confirmation_remote_summary"
LOG_DIR="outputs/logs"

EXTENSION_SEEDS="${EXTENSION_SEEDS:-3042 4042}"
ALL_SEEDS="${ALL_SEEDS:-42 1042 2042 3042 4042}"
F100_CENSORED_WEIGHT="${F100_CENSORED_WEIGHT:-0.01}"

mkdir -p "$SUMMARY_OUT" "$LOG_DIR"

weight_label() {
  local value="$1"
  echo "${value/./p}"
}

run_extension_matrix() {
  echo "[v1.2.12] $(date -Is) extending tanimoto_proxy seeds=${EXTENSION_SEEDS}"
  SEEDS="$EXTENSION_SEEDS" \
    CHALLENGER_METHOD="tanimoto_proxy_to_finetune" \
    CHALLENGER_ALPHA="0.5" \
    CHALLENGER_TAG="tanimoto_proxy_a0p5" \
    bash scripts/run_v1_2_11_f100_seed_stability_remote.sh matrix

  echo "[v1.2.12] $(date -Is) extending proxy_distance seeds=${EXTENSION_SEEDS}"
  SEEDS="$EXTENSION_SEEDS" \
    CHALLENGER_METHOD="proxy_distance_to_finetune" \
    CHALLENGER_ALPHA="0.5" \
    CHALLENGER_TAG="proxydist_a0p5" \
    bash scripts/run_v1_2_11_f100_seed_stability_remote.sh matrix
}

run_names_for_group() {
  local group="$1"
  local cw seed run_names=()
  cw="$(weight_label "$F100_CENSORED_WEIGHT")"
  for seed in $ALL_SEEDS; do
    case "$group" in
      anchor)
        run_names+=("transfer_f100_anchor_tanimoto_a1_seed${seed}_cebin_lw0025_censored_w${cw}")
        ;;
      tanimoto_proxy)
        run_names+=("transfer_f100_tanimoto_proxy_a0p5_seed${seed}_cebin_lw0025_censored_w${cw}")
        ;;
      proxydist)
        run_names+=("transfer_f100_proxydist_a0p5_seed${seed}_cebin_lw0025_censored_w${cw}")
        ;;
      *)
        echo "Unknown group: ${group}" >&2
        return 2
        ;;
    esac
  done
  local joined="${run_names[0]}"
  for run_name in "${run_names[@]:1}"; do
    joined="${joined},${run_name}"
  done
  echo "$joined"
}

run_ensembles() {
  "$PYTHON_BIN" scripts/summarize_seed_ensembles.py \
    --audit-root "$BASE_AUDIT_ROOT" \
    --out-dir "$SUMMARY_OUT" \
    --write-prediction-rows \
    --group "anchor_tanimoto_a1_5seed_ensemble=$(run_names_for_group anchor)" \
    --group "tanimoto_proxy_a0p5_5seed_ensemble=$(run_names_for_group tanimoto_proxy)" \
    --group "proxydist_a0p5_5seed_ensemble=$(run_names_for_group proxydist)"
}

echo "[start] $(date -Is) v1.2.12 f100 5-seed confirmation mode=${MODE} extension_seeds=${EXTENSION_SEEDS} all_seeds=${ALL_SEEDS}"
case "$MODE" in
  matrix)
    run_extension_matrix
    ;;
  ensemble)
    run_ensembles
    ;;
  all)
    run_extension_matrix
    run_ensembles
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use matrix, ensemble, or all." >&2
    exit 2
    ;;
esac
echo "[done] $(date -Is) v1.2.12 f100 5-seed confirmation mode=${MODE}"
