param(
    [string]$Config = "configs/experiment.remote.easyai.yaml",
    [string]$LocalPython = "python",
    [string]$RemoteHost = "",
    [int]$Port = 0,
    [string[]]$RemoteSubdirs = @("outputs/experiments", "outputs/logs", "outputs/derived"),
    [string]$RemoteSubdirsCsv = ""
)

. (Join-Path $PSScriptRoot "remote_common.ps1")

$settings = Read-RemoteConfig -Config $Config -LocalPython $LocalPython
if ($RemoteHost) { $settings.host = $RemoteHost }
if ($Port -gt 0) { $settings.port = $Port }
if ($RemoteSubdirsCsv) {
    $RemoteSubdirs = $RemoteSubdirsCsv.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
}
$projectRoot = $settings.project_root
$remote = Get-Remote -Settings $settings
$sshArgs = Get-SshArgs -Settings $settings
$scpArgs = Get-ScpArgs -Settings $settings
$remoteProject = $settings.project_dir.TrimEnd("/")

foreach ($subdir in $RemoteSubdirs) {
    $remotePath = Join-RemotePath -Base $remoteProject -Child $subdir
    $remotePathQuoted = Quote-RemoteArg -Value $remotePath
    & ssh @sshArgs $remote "test -e $remotePathQuoted"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Skip missing remote path: $remotePath"
        continue
    }

    $localParent = Join-Path $projectRoot (Split-Path $subdir -Parent)
    New-Item -ItemType Directory -Force -Path $localParent | Out-Null
    & scp @scpArgs -r "${remote}:$remotePath" $localParent
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to pull remote path: $remotePath"
    }
}

Write-Host "Pulled remote outputs from ${remote}:$remoteProject"
