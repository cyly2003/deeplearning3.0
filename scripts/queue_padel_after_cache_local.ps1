param(
    [string]$CachePath = "outputs/features/molecular_features_padel_morgan512.jsonl",
    [string]$ManifestPath = "outputs/features/molecular_features_padel_morgan512.jsonl.manifest.json",
    [int]$ExpectedRows = 6353,
    [int]$PollSeconds = 300,
    [string]$LocalPython = "E:\TOOLS\anaconda\python.exe",
    [string]$RemoteHost = "i.easy-ai.cloud",
    [int]$Port = 32136,
    [string]$RemoteUser = "easyai",
    [string]$RemoteProject = "/home/easyai/DL1/ecotox_qsar_transfer"
)

Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

Write-Host "[local-queue-start] $(Get-Date -Format o) cache=$CachePath expected_rows=$ExpectedRows"
while ($true) {
    if ((Test-Path $CachePath) -and (Test-Path $ManifestPath)) {
        try {
            $manifest = Get-Content -Raw $ManifestPath | ConvertFrom-Json
            $rowsWritten = [int]$manifest.rows_written
            $descriptorCount = [int]$manifest.descriptor_count
            if ($rowsWritten -ge $ExpectedRows -and $descriptorCount -gt 0) {
                Write-Host "[local-queue-cache-ready] $(Get-Date -Format o) rows_written=$rowsWritten descriptor_count=$descriptorCount"
                break
            }
            Write-Host "[local-queue-wait] $(Get-Date -Format o) rows_written=$rowsWritten descriptor_count=$descriptorCount"
        }
        catch {
            Write-Host "[local-queue-wait] $(Get-Date -Format o) manifest not ready: $($_.Exception.Message)"
        }
    }
    else {
        Write-Host "[local-queue-wait] $(Get-Date -Format o) cache or manifest missing"
    }
    Start-Sleep -Seconds $PollSeconds
}

Write-Host "[local-queue-sync] $(Get-Date -Format o) syncing project and features"
& pwsh .\scripts\sync_to_server.ps1 -LocalPython $LocalPython
if ($LASTEXITCODE -ne 0) {
    throw "sync_to_server.ps1 failed with exit code $LASTEXITCODE"
}

$remote = "${RemoteUser}@${RemoteHost}"
$remoteCommand = "cd $RemoteProject && mkdir -p outputs/logs && nohup bash scripts/queue_v1_2_34_35_padel_after_graph_remote.sh > outputs/logs/queue_v1_2_34_35_padel_after_graph.nohup.log 2>&1 < /dev/null & echo `$!"
Write-Host "[local-queue-launch] $(Get-Date -Format o) launching remote PaDEL queue"
& ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=5 -p $Port $remote "bash -lc '$remoteCommand'"
Write-Host "[local-queue-done] $(Get-Date -Format o)"
