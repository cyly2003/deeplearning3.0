param(
    [string]$Config = "configs/experiment.remote.easyai.yaml",
    [string]$LocalPython = "python",
    [string]$RemoteHost = "",
    [int]$Port = 0,
    [string]$SplitNamesCsv = "B_random_8_2,C_chemical_holdout_8_2,F_chemical_adapt_7_2_1",
    [string]$SourceTable = "",
    [string]$AblationsCsv = "full,no_fingerprint,no_duration,no_species_lifestage,no_context",
    [int]$Limit = 10000,
    [int]$Epochs = 30,
    [int]$BatchSize = 256,
    [double]$LearningRate = 0,
    [string]$Device = "cuda:0",
    [string]$OutDir = "outputs/experiments/deep_ablation_pilot_remote",
    [switch]$EarlyStopping,
    [int]$EarlyStoppingPatience = 0,
    [double]$EarlyStoppingMinDelta = -1,
    [double]$ValidationFraction = -1,
    [string]$MonitorSplit = "",
    [int]$FinetuneEpochs = 0,
    [double]$FinetuneLearningRate = 0,
    [int]$FinetuneBatchSize = 0,
    [string]$FinetuneFreeze = "",
    [switch]$ContinueOnError,
    [switch]$DryRun
)

$script = Join-Path $PSScriptRoot "train_remote.ps1"
$splitNames = $SplitNamesCsv.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
$ablations = $AblationsCsv.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }

if (-not $splitNames) {
    throw "No split names supplied."
}
if (-not $ablations) {
    throw "No ablations supplied."
}

foreach ($splitName in $splitNames) {
    foreach ($ablation in $ablations) {
        Write-Host "[remote-batch] split=$splitName ablation=$ablation"
        $trainScriptParams = @{
            Config = $Config
            LocalPython = $LocalPython
            SplitName = $splitName
            Ablation = $ablation
            Limit = $Limit
            Epochs = $Epochs
            BatchSize = $BatchSize
            Device = $Device
            OutDir = $OutDir
        }
        if ($RemoteHost) { $trainScriptParams.RemoteHost = $RemoteHost }
        if ($Port -gt 0) { $trainScriptParams.Port = $Port }
        if ($SourceTable) { $trainScriptParams.SourceTable = $SourceTable }
        if ($LearningRate -gt 0) { $trainScriptParams.LearningRate = $LearningRate }
        if ($EarlyStopping) { $trainScriptParams.EarlyStopping = $true }
        if ($EarlyStoppingPatience -gt 0) { $trainScriptParams.EarlyStoppingPatience = $EarlyStoppingPatience }
        if ($EarlyStoppingMinDelta -ge 0) { $trainScriptParams.EarlyStoppingMinDelta = $EarlyStoppingMinDelta }
        if ($ValidationFraction -ge 0) { $trainScriptParams.ValidationFraction = $ValidationFraction }
        if ($MonitorSplit) { $trainScriptParams.MonitorSplit = $MonitorSplit }
        if ($FinetuneEpochs -gt 0) { $trainScriptParams.FinetuneEpochs = $FinetuneEpochs }
        if ($FinetuneLearningRate -gt 0) { $trainScriptParams.FinetuneLearningRate = $FinetuneLearningRate }
        if ($FinetuneBatchSize -gt 0) { $trainScriptParams.FinetuneBatchSize = $FinetuneBatchSize }
        if ($FinetuneFreeze) { $trainScriptParams.FinetuneFreeze = $FinetuneFreeze }
        if ($DryRun) { $trainScriptParams.DryRun = $true }

        try {
            & $script @trainScriptParams
        }
        catch {
            if (-not $ContinueOnError) {
                throw
            }
            Write-Warning "[remote-batch-skip] split=$splitName ablation=$ablation error=$($_.Exception.Message)"
        }
    }
}
