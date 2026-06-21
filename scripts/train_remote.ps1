param(
    [string]$Config = "configs/experiment.remote.easyai.yaml",
    [string]$LocalPython = "python",
    [string]$RemoteHost = "",
    [int]$Port = 0,
    [string]$Db = "",
    [string]$SplitName = "",
    [string]$SourceTable = "",
    [int]$Limit = 0,
    [int]$Epochs = 0,
    [int]$BatchSize = 0,
    [double]$LearningRate = 0,
    [double]$WeightDecay = -1,
    [string]$Scheduler = "",
    [double]$Dropout = -1,
    [string]$TargetStandardization = "",
    [string]$Device = "",
    [string]$OutDir = "",
    [string]$RunVersion = "",
    [string]$RunNameZh = "",
    [string]$Ablation = "full",
    [switch]$EarlyStopping,
    [int]$EarlyStoppingPatience = 0,
    [double]$EarlyStoppingMinDelta = -1,
    [double]$ValidationFraction = -1,
    [string]$MonitorSplit = "",
    [int]$FinetuneEpochs = 0,
    [double]$FinetuneLearningRate = 0,
    [int]$FinetuneBatchSize = 0,
    [string]$FinetuneScheduler = "",
    [string]$FinetuneFreeze = "",
    [double]$FinetuneValidationFraction = -1,
    [int]$AugmentTrainReplicates = 0,
    [int]$AugmentFinetuneReplicates = 0,
    [double]$AugmentNumericNoiseStd = -1,
    [double]$AugmentTargetNoiseStd = -1,
    [int]$TestNoiseReplicates = 0,
    [double]$TestNoiseNumericStd = -1,
    [switch]$FeatureZScoreCorrection,
    [switch]$NoFeatureZScoreCorrection,
    [double]$FeatureZScoreThreshold = -1,
    [int]$MetricMinN = 0,
    [string]$SourceWeightingMethod = "",
    [double]$SourceWeightingAlpha = -1,
    [switch]$EffectLevelWeighting,
    [switch]$NoEffectLevelWeighting,
    [double]$EffectLevelWeightingBeta = -1,
    [string]$DomainAlignmentMethod = "",
    [double]$DomainAlignmentWeight = -1,
    [switch]$Swa,
    [switch]$NoSwa,
    [int]$SwaStartEpoch = 0,
    [switch]$SmokeTest,
    [switch]$DryRun
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
$logStamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$logSplit = if ($SplitName) { ($SplitName -replace "[^A-Za-z0-9_.-]", "_") } else { "default" }
$logAblation = if ($Ablation) { ($Ablation -replace "[^A-Za-z0-9_.-]", "_") } else { "full" }
$remoteLog = Join-RemotePath -Base $remoteProject -Child "outputs/logs/remote_train_${logSplit}_${logAblation}_$logStamp.log"

$projectDirQuoted = Quote-RemoteArg -Value $remoteProject
$pythonQuoted = Quote-RemoteArg -Value $settings.python
$configQuoted = Quote-RemoteArg -Value $remoteConfig
$logQuoted = Quote-RemoteArg -Value $remoteLog

$trainArgs = "--config $configQuoted"
if ($Db) {
    $remoteDb = if ($Db.StartsWith("/")) { $Db } else { Join-RemotePath -Base $remoteProject -Child $Db }
    $trainArgs += " --db $(Quote-RemoteArg -Value $remoteDb)"
}
if ($SplitName) { $trainArgs += " --split-name $(Quote-RemoteArg -Value $SplitName)" }
if ($SourceTable) { $trainArgs += " --source-table $(Quote-RemoteArg -Value $SourceTable)" }
if ($Limit -gt 0) { $trainArgs += " --limit $Limit" }
if ($Epochs -gt 0) { $trainArgs += " --epochs $Epochs" }
if ($BatchSize -gt 0) { $trainArgs += " --batch-size $BatchSize" }
if ($LearningRate -gt 0) { $trainArgs += " --learning-rate $LearningRate" }
if ($WeightDecay -ge 0) { $trainArgs += " --weight-decay $WeightDecay" }
if ($Scheduler) { $trainArgs += " --scheduler $(Quote-RemoteArg -Value $Scheduler)" }
if ($Dropout -ge 0) { $trainArgs += " --dropout $Dropout" }
if ($TargetStandardization) { $trainArgs += " --target-standardization $(Quote-RemoteArg -Value $TargetStandardization)" }
if ($Device) { $trainArgs += " --device $(Quote-RemoteArg -Value $Device)" }
if ($RunVersion) { $trainArgs += " --run-version $(Quote-RemoteArg -Value $RunVersion)" }
if ($RunNameZh) { $trainArgs += " --run-name-zh $(Quote-RemoteArg -Value $RunNameZh)" }
if ($Ablation) { $trainArgs += " --ablation $(Quote-RemoteArg -Value $Ablation)" }
if ($EarlyStopping) { $trainArgs += " --early-stopping" }
if ($EarlyStoppingPatience -gt 0) { $trainArgs += " --early-stopping-patience $EarlyStoppingPatience" }
if ($EarlyStoppingMinDelta -ge 0) { $trainArgs += " --early-stopping-min-delta $EarlyStoppingMinDelta" }
if ($ValidationFraction -ge 0) { $trainArgs += " --validation-fraction $ValidationFraction" }
if ($MonitorSplit) { $trainArgs += " --monitor-split $(Quote-RemoteArg -Value $MonitorSplit)" }
if ($FinetuneEpochs -gt 0) { $trainArgs += " --finetune-epochs $FinetuneEpochs" }
if ($FinetuneLearningRate -gt 0) { $trainArgs += " --finetune-learning-rate $FinetuneLearningRate" }
if ($FinetuneBatchSize -gt 0) { $trainArgs += " --finetune-batch-size $FinetuneBatchSize" }
if ($FinetuneScheduler) { $trainArgs += " --finetune-scheduler $(Quote-RemoteArg -Value $FinetuneScheduler)" }
if ($FinetuneFreeze) { $trainArgs += " --finetune-freeze $(Quote-RemoteArg -Value $FinetuneFreeze)" }
if ($FinetuneValidationFraction -ge 0) { $trainArgs += " --finetune-validation-fraction $FinetuneValidationFraction" }
if ($AugmentTrainReplicates -gt 0) { $trainArgs += " --augment-train-replicates $AugmentTrainReplicates" }
if ($AugmentFinetuneReplicates -gt 0) { $trainArgs += " --augment-finetune-replicates $AugmentFinetuneReplicates" }
if ($AugmentNumericNoiseStd -ge 0) { $trainArgs += " --augment-numeric-noise-std $AugmentNumericNoiseStd" }
if ($AugmentTargetNoiseStd -ge 0) { $trainArgs += " --augment-target-noise-std $AugmentTargetNoiseStd" }
if ($TestNoiseReplicates -gt 0) { $trainArgs += " --test-noise-replicates $TestNoiseReplicates" }
if ($TestNoiseNumericStd -ge 0) { $trainArgs += " --test-noise-numeric-std $TestNoiseNumericStd" }
if ($FeatureZScoreCorrection) { $trainArgs += " --feature-zscore-correction" }
if ($NoFeatureZScoreCorrection) { $trainArgs += " --no-feature-zscore-correction" }
if ($FeatureZScoreThreshold -ge 0) { $trainArgs += " --feature-zscore-threshold $FeatureZScoreThreshold" }
if ($MetricMinN -gt 0) { $trainArgs += " --metric-min-n $MetricMinN" }
if ($SourceWeightingMethod) { $trainArgs += " --source-weighting-method $(Quote-RemoteArg -Value $SourceWeightingMethod)" }
if ($SourceWeightingAlpha -ge 0) { $trainArgs += " --source-weighting-alpha $SourceWeightingAlpha" }
if ($EffectLevelWeighting) { $trainArgs += " --effect-level-weighting" }
if ($NoEffectLevelWeighting) { $trainArgs += " --no-effect-level-weighting" }
if ($EffectLevelWeightingBeta -ge 0) { $trainArgs += " --effect-level-weighting-beta $EffectLevelWeightingBeta" }
if ($DomainAlignmentMethod) { $trainArgs += " --domain-alignment-method $(Quote-RemoteArg -Value $DomainAlignmentMethod)" }
if ($DomainAlignmentWeight -ge 0) { $trainArgs += " --domain-alignment-weight $DomainAlignmentWeight" }
if ($Swa) { $trainArgs += " --swa" }
if ($NoSwa) { $trainArgs += " --no-swa" }
if ($SwaStartEpoch -gt 0) { $trainArgs += " --swa-start-epoch $SwaStartEpoch" }
if ($OutDir) {
    $remoteOutDir = Join-RemotePath -Base $remoteProject -Child $OutDir
    $trainArgs += " --out-dir $(Quote-RemoteArg -Value $remoteOutDir)"
}

$trainCommand = "$pythonQuoted -m qsar_tl.training.train $trainArgs"
if ($SmokeTest) {
    $trainCommand = "$pythonQuoted -m qsar_tl.cli validate-config --config $configQuoted && $trainCommand"
}

$bashBody = "set -o pipefail; cd $projectDirQuoted && mkdir -p outputs/logs && ($trainCommand) 2>&1 | tee $logQuoted"
$remoteCommand = "bash -lc $(Quote-RemoteArg -Value $bashBody)"

if ($DryRun) {
    $displayParts = @("ssh") + $sshArgs + @($remote, $remoteCommand)
    Write-Host (Format-DisplayCommand -Parts $displayParts)
    return
}

& ssh @sshArgs $remote $remoteCommand
if ($LASTEXITCODE -ne 0) {
    throw "Remote training command failed on $remote"
}

Write-Host "Remote log: ${remote}:$remoteLog"
