param(
    [string]$Config = "configs/experiment.remote.easyai.yaml",
    [string]$LocalPython = "python",
    [string]$RemoteHost = "",
    [int]$Port = 0,
    [switch]$RebuildModelingTables,
    [switch]$BuildMediumTables,
    [switch]$GenerateSplits,
    [switch]$RunBaselineMatrix,
    [switch]$DryRun,
    [string]$SourceTable = "",
    [string]$SplitNamePrefix = "",
    [string[]]$Models = @(),
    [string]$ModelsCsv = "",
    [string[]]$SplitNames = @(),
    [string]$SplitNamesCsv = "",
    [string]$BaselineOutDir = "outputs/experiments/baseline_matrix",
    [int]$Limit = 0
)

. (Join-Path $PSScriptRoot "remote_common.ps1")

$settings = Read-RemoteConfig -Config $Config -LocalPython $LocalPython
if ($RemoteHost) { $settings.host = $RemoteHost }
if ($Port -gt 0) { $settings.port = $Port }

$remote = Get-Remote -Settings $settings
$sshArgs = Get-SshArgs -Settings $settings
$configName = Split-Path $settings.config_path -Leaf
$remoteProject = $settings.project_dir.TrimEnd("/")
$remoteConfig = Join-RemotePath -Base $remoteProject -Child "configs/$configName"
$remoteDb = Join-RemotePath -Base $remoteProject -Child $settings.modeling_tables_db

$projectDirQuoted = Quote-RemoteArg -Value $remoteProject
$pythonQuoted = Quote-RemoteArg -Value $settings.python
$configQuoted = Quote-RemoteArg -Value $remoteConfig
$dbQuoted = Quote-RemoteArg -Value $remoteDb
$outDirQuoted = Quote-RemoteArg -Value (Join-RemotePath -Base $remoteProject -Child $BaselineOutDir)

$commands = @(
    "$pythonQuoted -m qsar_tl.cli validate-config --config $configQuoted"
)

if ($RebuildModelingTables) {
    $commands += "$pythonQuoted -m qsar_tl.cli build-modeling-tables --config $configQuoted"
    $commands += "$pythonQuoted -m qsar_tl.cli build-task-tables --config $configQuoted --db $dbQuoted"
}

if ($BuildMediumTables) {
    $commands += "$pythonQuoted scripts/build_medium_domain_tables.py --db $dbQuoted"
}

if ($GenerateSplits) {
    $sourceTableArg = if ($SourceTable) { " --source-table $(Quote-RemoteArg -Value $SourceTable)" } else { "" }
    $prefixArg = if ($SplitNamePrefix) { " --split-name-prefix $(Quote-RemoteArg -Value $SplitNamePrefix)" } else { "" }
    $commands += "$pythonQuoted -m qsar_tl.cli generate-experiment-splits --config $configQuoted --db $dbQuoted$sourceTableArg$prefixArg"
}

if ($RunBaselineMatrix) {
    $limitArg = if ($Limit -gt 0) { " --limit $Limit" } else { "" }
    if ($ModelsCsv) {
        $Models = $ModelsCsv.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    }
    if ($SplitNamesCsv) {
        $SplitNames = $SplitNamesCsv.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    }
    $modelArgs = ""
    foreach ($model in $Models) {
        $modelArgs += " --model $(Quote-RemoteArg -Value $model)"
    }
    $splitArgs = ""
    foreach ($splitName in $SplitNames) {
        $splitArgs += " --split-name $(Quote-RemoteArg -Value $splitName)"
    }
    $sourceTableArg = if ($SourceTable) { " --source-table $(Quote-RemoteArg -Value $SourceTable)" } else { "" }
    $commands += "$pythonQuoted -m qsar_tl.cli run-baseline-matrix --config $configQuoted --db $dbQuoted$sourceTableArg --out-dir $outDirQuoted --continue-on-error$limitArg$modelArgs$splitArgs"
}

$joinedCommands = $commands -join " && "
$bashBody = "set -euo pipefail; cd $projectDirQuoted && $joinedCommands"
$remoteCommand = "bash -lc $(Quote-RemoteArg -Value $bashBody)"

if ($DryRun) {
    $displayParts = @("ssh") + $sshArgs + @($remote, $remoteCommand)
    Write-Host (Format-DisplayCommand -Parts $displayParts)
    return
}

& ssh @sshArgs $remote $remoteCommand
if ($LASTEXITCODE -ne 0) {
    throw "Remote sequence failed on $remote"
}
