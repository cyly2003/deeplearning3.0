param(
    [string]$Config = "configs/experiment.remote.easyai.yaml",
    [string]$LocalPython = "E:\TOOLS\anaconda\python.exe",
    [switch]$Sync,
    [switch]$Preflight,
    [switch]$Launch,
    [switch]$DryRun
)

. (Join-Path $PSScriptRoot "remote_common.ps1")

$settings = Read-RemoteConfig -Config $Config -LocalPython $LocalPython
$remote = Get-Remote -Settings $settings
$sshArgs = Get-SshArgs -Settings $settings
$scpArgs = Get-ScpArgs -Settings $settings
$projectRoot = $settings.project_root
$remoteProject = $settings.project_dir.TrimEnd("/")
$remoteProjectQuoted = Quote-RemoteArg -Value $remoteProject
$pythonQuoted = Quote-RemoteArg -Value $settings.python

$allowlist = @(
    "scripts/build_v1_2_46_reference_group_splits.py",
    "scripts/summarize_v1_2_45_reference_cluster_bootstrap.py",
    "scripts/validate_v1_2_46_reference_group_run.py",
    "scripts/summarize_v1_2_46_reference_group_matrix.py",
    "scripts/run_v1_2_46_reference_group_matrix_remote.sh",
    "tests/test_v1_2_46_reference_group_matrix.py"
)

function Invoke-RemoteBash {
    param([Parameter(Mandatory = $true)][string]$Body)
    $command = "bash -lc $(Quote-RemoteArg -Value $Body)"
    if ($DryRun) {
        $display = @("ssh") + $sshArgs + @($remote, $command)
        Write-Host (Format-DisplayCommand -Parts $display)
        return
    }
    & ssh @sshArgs $remote $command
    if ($LASTEXITCODE -ne 0) {
        throw "Remote command failed on $remote"
    }
}
if ($Sync) {
    foreach ($relative in $allowlist) {
        $localPath = Join-Path $projectRoot $relative
        if (-not (Test-Path -LiteralPath $localPath)) {
            throw "Allowlisted launch file is missing: $relative"
        }
        $remoteRelative = $relative -replace "\\", "/"
        $remoteDir = Split-Path $remoteRelative -Parent
        $remoteDir = $remoteDir -replace "\\", "/"
        Invoke-RemoteBash -Body "set -euo pipefail; mkdir -p $remoteProjectQuoted/$(Quote-RemoteArg -Value $remoteDir)"
        $destination = "${remote}:$remoteProject/$remoteDir/"
        if ($DryRun) {
            $display = @("scp") + $scpArgs + @($localPath, $destination)
            Write-Host (Format-DisplayCommand -Parts $display)
        }
        else {
            & scp @scpArgs $localPath $destination
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to sync allowlisted file: $relative"
            }
        }
    }
    Write-Host "Synced v1.2.45/v1.2.46 allowlist only."
}

if ($Preflight) {
    $compileFiles = ($allowlist | Where-Object { $_ -like "*.py" }) -join " "
    $body = @(
        "set -euo pipefail",
        "cd $remoteProjectQuoted",
        "$pythonQuoted -m py_compile $compileFiles",
        "bash -n scripts/run_v1_2_46_reference_group_matrix_remote.sh",
        "$pythonQuoted -m pytest tests/test_v1_2_46_reference_group_matrix.py tests/test_v1_2_44_matrix.py tests/test_summarize_v1_2_44_matrix.py -q",
        "$pythonQuoted scripts/summarize_v1_2_45_reference_cluster_bootstrap.py",
        "REBUILD_SPLITS=1 PARALLEL_JOBS=2 bash scripts/run_v1_2_46_reference_group_matrix_remote.sh build"
    ) -join "; "
    Invoke-RemoteBash -Body $body
    Write-Host "Remote v1.2.45 statistics and v1.2.46 split preflight completed."
}

if ($Launch) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $controllerLog = "outputs/logs/v1_2_46_reference_group_gate_$stamp.log"
    $pidFile = "outputs/logs/v1_2_46_reference_group_gate_$stamp.pid"
    $gate = (
        "set -euo pipefail; " +
        "REBUILD_SPLITS=0 PARALLEL_JOBS=2 bash scripts/run_v1_2_46_reference_group_matrix_remote.sh smoke; " +
        "REBUILD_SPLITS=0 PARALLEL_JOBS=2 bash scripts/run_v1_2_46_reference_group_matrix_remote.sh formal"
    )
    $body = (
        "set -euo pipefail; cd $remoteProjectQuoted; mkdir -p outputs/logs; " +
        "setsid bash -lc $(Quote-RemoteArg -Value $gate) " +
        "> $(Quote-RemoteArg -Value $controllerLog) 2>&1 < /dev/null & " +
        "pid=`$!; echo `$pid > $(Quote-RemoteArg -Value $pidFile); " +
        "printf 'pid=%s\nlog=%s\npidfile=%s\n' `$pid $(Quote-RemoteArg -Value $controllerLog) $(Quote-RemoteArg -Value $pidFile)"
    )
    Invoke-RemoteBash -Body $body
}
