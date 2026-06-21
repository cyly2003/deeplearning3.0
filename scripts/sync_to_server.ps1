param(
    [string]$Config = "configs/experiment.remote.easyai.yaml",
    [string]$LocalPython = "python",
    [string]$RemoteHost = "",
    [int]$Port = 0,
    [switch]$IncludeSourceDb,
    [switch]$IncludeDerivedDb
)

. (Join-Path $PSScriptRoot "remote_common.ps1")

$settings = Read-RemoteConfig -Config $Config -LocalPython $LocalPython
if ($RemoteHost) { $settings.host = $RemoteHost }
if ($Port -gt 0) { $settings.port = $Port }
$projectRoot = $settings.project_root
$remote = Get-Remote -Settings $settings
$sshArgs = Get-SshArgs -Settings $settings
$scpArgs = Get-ScpArgs -Settings $settings
$remoteProject = $settings.project_dir.TrimEnd("/")
$remoteProjectQuoted = Quote-RemoteArg -Value $remoteProject
$tempDataFiles = @()

function New-StableSqliteCopy {
    param(
        [string]$SourcePath,
        [string]$LocalPython
    )

    $extension = [System.IO.Path]::GetExtension($SourcePath).ToLowerInvariant()
    if ($extension -notin @(".sqlite", ".sqlite3", ".db")) {
        return $SourcePath
    }

    $tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("qsar-sync-" + [System.Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $tempDir | Out-Null
    $copyPath = Join-Path $tempDir ([System.IO.Path]::GetFileName($SourcePath))
    $backupScript = @'
import sqlite3
import sys

source_path, copy_path = sys.argv[1], sys.argv[2]
source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
try:
    target = sqlite3.connect(copy_path)
    try:
        source.backup(target)
    finally:
        target.close()
finally:
    source.close()
'@
    $backupScript | & $LocalPython - $SourcePath $copyPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create stable SQLite copy: $SourcePath"
    }
    $script:tempDataFiles += $copyPath
    return $copyPath
}

& ssh @sshArgs $remote "mkdir -p $remoteProjectQuoted"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create remote project directory: $remoteProject"
}

$corePaths = @(
    "pyproject.toml",
    "README.md",
    "PROJECT_STATUS.md",
    "style_journal_clean_v1.yaml",
    "configs",
    "docs",
    "qsar_tl",
    "scripts",
    "tests"
) | Where-Object { Test-Path (Join-Path $projectRoot $_) }

Push-Location $projectRoot
try {
    & scp @scpArgs -r @corePaths "${remote}:$remoteProject/"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to sync project files to $remoteProject"
    }

    $configName = Split-Path $settings.config_path -Leaf
    & scp @scpArgs $Config "${remote}:$remoteProject/configs/$configName"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to sync active config: $Config"
    }

    $featureDir = Join-Path $projectRoot "outputs/features"
    if (Test-Path $featureDir) {
        & ssh @sshArgs $remote "mkdir -p $(Quote-RemoteArg -Value (Join-RemotePath -Base $remoteProject -Child 'outputs'))"
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create remote outputs directory."
        }
        & scp @scpArgs -r "outputs/features" "${remote}:$remoteProject/outputs/"
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to sync outputs/features."
        }
    }

    $dataFiles = @()
    if ($IncludeSourceDb -and $settings.sqlite_db) {
        $dataFiles += [string]$settings.sqlite_db
    }
    if ($IncludeDerivedDb -and $settings.modeling_tables_db) {
        $dataFiles += [string]$settings.modeling_tables_db
    }

    foreach ($path in $dataFiles) {
        if (-not (Test-Path $path)) {
            throw "Requested data file does not exist: $path"
        }
        $uploadPath = New-StableSqliteCopy -SourcePath $path -LocalPython $LocalPython
        $remotePath = Join-RemotePath -Base $remoteProject -Child $path
        $remoteDir = Split-Path ($remotePath -replace "/", [System.IO.Path]::DirectorySeparatorChar) -Parent
        $remoteDir = $remoteDir -replace "\\", "/"
        $remoteDirQuoted = Quote-RemoteArg -Value $remoteDir
        & ssh @sshArgs $remote "mkdir -p $remoteDirQuoted"
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create remote data directory: $remoteDir"
        }
        & scp @scpArgs $uploadPath "${remote}:$remoteDir/"
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to sync data file: $path"
        }
        $remoteFile = Join-RemotePath -Base $remoteDir -Child ([System.IO.Path]::GetFileName($uploadPath))
        $expectedRemoteFile = Join-RemotePath -Base $remoteDir -Child ([System.IO.Path]::GetFileName($path))
        if ($remoteFile -ne $expectedRemoteFile) {
            & ssh @sshArgs $remote "mv $(Quote-RemoteArg -Value $remoteFile) $(Quote-RemoteArg -Value $expectedRemoteFile)"
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to move synced data file into place: $expectedRemoteFile"
            }
        }
    }
}
finally {
    foreach ($tempFile in $tempDataFiles) {
        $tempDir = Split-Path $tempFile -Parent
        if ($tempDir -and (Test-Path $tempDir)) {
            $resolvedTempDir = (Resolve-Path -LiteralPath $tempDir).Path
            $resolvedRoot = (Resolve-Path -LiteralPath ([System.IO.Path]::GetTempPath())).Path.TrimEnd("\", "/")
            $tempLeaf = Split-Path $resolvedTempDir -Leaf
            if ($resolvedTempDir.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase) -and $tempLeaf.StartsWith("qsar-sync-")) {
                Remove-Item -LiteralPath $resolvedTempDir -Recurse -Force
            }
        }
    }
    Pop-Location
}

Write-Host "Synced project to ${remote}:$remoteProject"
