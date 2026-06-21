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
$python = Quote-RemoteArg -Value $settings.python
$conda = if ($settings.conda) { Quote-RemoteArg -Value $settings.conda } else { "conda" }

$bashBody = @(
    "set -euo pipefail",
    "hostname",
    "pwd",
    "lsb_release -a",
    "df -h /",
    "nvidia-smi",
    "$python --version",
    "$python -c 'import sys; print(sys.executable)'",
    "$conda --version || true",
    "mkdir -p $projectDir",
    "cd $projectDir",
    "pwd"
) -join "; "

& ssh @sshArgs $remote "bash -lc $(Quote-RemoteArg -Value $bashBody)"
if ($LASTEXITCODE -ne 0) {
    throw "Remote GPU check failed on $remote"
}
