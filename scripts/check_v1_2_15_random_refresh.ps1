param(
    [string]$Config = "configs/experiment.remote.easyai.yaml",
    [string]$LocalPython = "python",
    [string]$RemoteHost = "",
    [int]$Port = 0
)

. (Join-Path $PSScriptRoot "remote_common.ps1")

$settings = Read-RemoteConfig -Config $Config -LocalPython $LocalPython
if ($RemoteHost) { $settings.host = $RemoteHost }
if ($Port -gt 0) { $settings.port = $Port }

$remote = Get-Remote -Settings $settings
$sshArgs = Get-SshArgs -Settings $settings
$projectDir = Quote-RemoteArg -Value $settings.project_dir

$bashBody = @'
set -euo pipefail
cd PROJECT_DIR_PLACEHOLDER

echo "[time] $(date -Is)"
echo "[project] $PWD"

log="$(ls -1t outputs/logs/run_v1_2_15_random_split_policy_5seed_refresh_*.log 2>/dev/null | head -1 || true)"
echo "[log] ${log:-MISSING}"
if [[ -n "$log" ]]; then
  echo "[latest-log-lines]"
  grep -n "\[run-start\|\[run-done\|\[epoch\|\[finetune epoch\|\[done\]" "$log" | tail -30 || true
fi

echo "[processes]"
pgrep -af "run_v1_2_15_random_split_policy_remote|qsar_tl.training.train" \
  | grep -v "check_run" \
  | grep -v "pgrep -af" || true

root="outputs/experiments/v1_2_15_random_split_policy_formal_remote"
summary="outputs/experiments/v1_2_15_random_split_policy_formal_remote_summary"
expected=0
complete=0
missing=0

check_run() {
  local run_name="$1"
  local split_name="$2"
  local dir="${root}/v1.2.15_${run_name}/deep/full/${split_name}"
  expected=$((expected + 1))
  if [[ -s "${dir}/predictions.csv" && -s "${dir}/manifest.json" && -s "${dir}/history.csv" ]]; then
    complete=$((complete + 1))
  else
    missing=$((missing + 1))
    echo "[missing-run] ${run_name} ${split_name}"
  fi
}

for seed in 3042 4042; do
  check_run \
    "transfer_f100_anchor_random8_2_seed${seed}_cebin_lw0025_censored_w0p01" \
    "M_v2_aquatic_to_soil_ptox_adapt_B_random_8_2_f100"
  for fold in 1 2 3 4 5; do
    check_run \
      "transfer_f100_anchor_random5fold_fold${fold}_seed${seed}_cebin_lw0025_censored_w0p01" \
      "M_v2_aquatic_to_soil_ptox_adapt_E_random_5fold_fold${fold}_f100"
  done
done

echo "[refresh-completion] complete=${complete} missing=${missing} expected=${expected}"

echo "[summary-files]"
combined="${summary}/split_policy_ensemble_combined_summary.csv"
for file in \
  "$combined" \
  "${summary}/split_policy_ensemble_family_summary.csv" \
  "${summary}/split_policy_ensemble_fold_summary.csv"; do
  if [[ -s "$file" ]]; then
    echo "present $(stat -c%s "$file") $file"
  else
    echo "missing $file"
  fi
done

if [[ -s "$combined" ]]; then
  if grep -q "42;1042;2042;3042;4042" "$combined"; then
    echo "[summary-seeds] five_seed_ready"
  else
    echo "[summary-seeds] stale_or_incomplete"
  fi
  echo "[combined-summary]"
  cat "$combined"
fi
'@

$bashBody = $bashBody.Replace("PROJECT_DIR_PLACEHOLDER", $settings.project_dir)
& ssh @sshArgs $remote "bash -lc $(Quote-RemoteArg -Value $bashBody)"
if ($LASTEXITCODE -ne 0) {
    throw "Remote random refresh check failed on $remote"
}
