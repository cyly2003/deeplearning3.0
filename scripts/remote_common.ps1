Set-StrictMode -Version Latest

function Get-ProjectRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Resolve-ProjectPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ProjectRoot
    )
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $ProjectRoot $Path)
}

function Read-RemoteConfig {
    param(
        [string]$Config = "configs/experiment.remote.easyai.yaml",
        [string]$LocalPython = "python"
    )

    $projectRoot = Get-ProjectRoot
    $configPath = Resolve-ProjectPath -Path $Config -ProjectRoot $projectRoot
    $extractScript = @'
import json
import sys

from qsar_tl.config import load_config

config = load_config(sys.argv[1])
execution = config.get("execution", {})
remote = execution.get("remote", {}) if isinstance(execution.get("remote", {}), dict) else {}
paths = config.get("paths", {})
data = config.get("data", {})

payload = {
    "host": remote.get("host"),
    "port": remote.get("port"),
    "user": remote.get("user"),
    "project_dir": remote.get("project_dir"),
    "python": remote.get("python"),
    "conda": remote.get("conda"),
    "sqlite_db": paths.get("sqlite_db"),
    "output_dir": paths.get("output_dir"),
    "logs_dir": paths.get("logs_dir"),
    "modeling_tables_db": data.get("modeling_tables_db"),
}
print(json.dumps(payload, ensure_ascii=False))
'@

    Push-Location $projectRoot
    try {
        $json = & $LocalPython -c $extractScript $configPath
    }
    finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to read remote settings from config: $Config"
    }

    $settings = $json | ConvertFrom-Json
    foreach ($name in @("host", "user", "project_dir", "python")) {
        if (-not $settings.$name) {
            throw "Missing execution.remote.$name in config: $Config"
        }
    }
    $settings | Add-Member -NotePropertyName config_path -NotePropertyValue $configPath -Force
    $settings | Add-Member -NotePropertyName project_root -NotePropertyValue $projectRoot -Force
    return $settings
}

function Get-SshArgs {
    param([Parameter(Mandatory = $true)]$Settings)
    $args = @("-o", "ServerAliveInterval=60", "-o", "ServerAliveCountMax=5")
    if ($Settings.port) {
        $args += @("-p", [string]$Settings.port)
    }
    return $args
}

function Get-ScpArgs {
    param([Parameter(Mandatory = $true)]$Settings)
    $args = @("-o", "ServerAliveInterval=60", "-o", "ServerAliveCountMax=5")
    if ($Settings.port) {
        $args += @("-P", [string]$Settings.port)
    }
    return $args
}

function Get-Remote {
    param([Parameter(Mandatory = $true)]$Settings)
    return "$($Settings.user)@$($Settings.host)"
}

function Join-RemotePath {
    param(
        [Parameter(Mandatory = $true)][string]$Base,
        [Parameter(Mandatory = $true)][string]$Child
    )
    $cleanBase = $Base.TrimEnd("/")
    $cleanChild = ($Child -replace "\\", "/").TrimStart("/")
    return "$cleanBase/$cleanChild"
}

function Quote-RemoteArg {
    param([Parameter(Mandatory = $true)][string]$Value)
    $escaped = $Value -replace "'", "'\''"
    return "'$escaped'"
}

function Format-DisplayCommand {
    param([Parameter(Mandatory = $true)][string[]]$Parts)
    $formatted = foreach ($part in $Parts) {
        if ($part -match "[\s&|()]") {
            $escaped = $part.Replace([string][char]96, [string][char]96 + [string][char]96)
            $escaped = $escaped.Replace([string][char]34, [string][char]96 + [string][char]34)
            $escaped = $escaped.Replace('$', [string][char]96 + '$')
            [string][char]34 + $escaped + [string][char]34
        }
        else {
            $part
        }
    }
    return ($formatted -join " ")
}
