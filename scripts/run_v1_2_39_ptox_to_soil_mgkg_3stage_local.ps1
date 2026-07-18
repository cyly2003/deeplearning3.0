[CmdletBinding()]
param(
    [ValidateSet("all", "splits", "main", "ablations", "explain", "summarize", "smoke")]
    [string]$Mode = "all",
    [int[]]$Seeds = @(42),
    [int[]]$Folds = @(1, 2, 3, 4, 5),
    [int]$PretrainEpochs = 30,
    [int]$SoilPtoxEpochs = 20,
    [int]$SoilMgkgEpochs = 20,
    [int]$BatchSize = 512,
    [int]$SmokeEpochs = 1,
    [string]$Python = ".venv-cuda\Scripts\python.exe",
    [string]$Device = "cuda:0",
    [bool]$RunFull5Fold = $true,
    [bool]$RunAblations = $true,
    [switch]$SkipNetworkCheck
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$config = "configs\experiment.local.cuda3050ti.yaml"
$db = "outputs\derived\modeling_dataset_v2_0_0_rebuild.sqlite"
$sourceTable = "aggregated_task_records_medium_domain_qc_expanded"
$soilPtoxTable = "aggregated_task_records_soil_ptox_qc"
$soilMgkgTable = "aggregated_task_records_soil_mg_kg_qc"
$outRoot = "outputs\experiments\v1_2_39_ptox_to_soil_mgkg_3stage_local"
$summaryOut = "outputs\experiments\v1_2_39_ptox_to_soil_mgkg_3stage_local_summary"
$auditDir = "outputs\audits\v1_2_39_ptox_to_soil_mgkg"
$logDir = "outputs\logs"
$runTimes = Join-Path $logDir "run_v1_2_39_ptox_to_soil_mgkg_3stage_local_times.csv"
$runVersion = "v1.2.39"
$protocolTag = "protocolfix_routingfix"
$splitPrefix = "M_v1_2_39_ptox_to_soil_mgkg_"
$random8Split = "${splitPrefix}B_random_8_2"

# All variants explicitly keep adapters disabled. pTox water-to-soil evidence
# already showed a negative adapter contribution, so it is not an ablation axis.
$baseTrainArgs = @(
    "-m", "qsar_tl.training.train",
    "--config", $config,
    "--db", $db,
    "--out-dir", $outRoot,
    "--run-version", $runVersion,
    "--source-table", $sourceTable,
    "--target-standardization", "per_task_target",
    "--head-routing", "task_target",
    "--allow-mixed-target-dimensions",
    "--no-medium-adapters",
    "--batch-size", "$BatchSize",
    "--scheduler", "cosine",
    "--finetune-scheduler", "cosine",
    "--finetune-mgkg-scheduler", "cosine",
    "--learning-rate", "0.0005",
    "--finetune-learning-rate", "0.0001",
    "--finetune-mgkg-learning-rate", "0.0005",
    "--finetune-freeze", "none",
    # Soil mg/kg has a distinct response scale and a newly initialized head.
    # Fine-tune the full network so the shared pTox representation can adapt.
    "--finetune-mgkg-freeze", "none",
    "--finetune-validation-fraction", "0.2",
    "--finetune-mgkg-validation-fraction", "0.2",
    "--early-stopping",
    "--early-stopping-patience", "15",
    "--early-stopping-min-delta", "0.0",
    # Stage 1 is selected only from held-out aquatic pTox training rows.
    # Stage 2/3 continue to use their own internal validation fractions above.
    "--monitor-split", "internal_train_fraction",
    "--validation-fraction", "0.1",
    "--source-weighting-method", "tanimoto_to_finetune",
    "--source-weighting-alpha", "1.0",
    "--toxicity-binning",
    "--toxicity-binning-mode", "aux_classification",
    "--toxicity-binning-scheme", "authority_v1",
    "--toxicity-binning-loss-weight", "0.025",
    # The current censored-row router does not cover finetune_mgkg, so this
    # protocol must not make a stage-3 censored-loss claim.
    "--no-censored-loss",
    "--no-effect-level-weighting",
    "--weight-decay", "0.00001",
    "--dropout", "0.10",
    "--metric-min-n", "5",
    "--device", $Device
)

$ablationMatrix = @(
    @{ Label = "M1_no_context"; Ablation = "no_context"; Extra = @() },
    @{ Label = "M2_no_species_lifestage"; Ablation = "no_species_lifestage"; Extra = @() },
    @{ Label = "M3_no_toxicity_binning"; Ablation = "full"; Extra = @("--no-toxicity-binning") },
    @{ Label = "M4_no_source_weighting"; Ablation = "full"; Extra = @("--source-weighting-method", "none", "--source-weighting-alpha", "1.0") },
    @{ Label = "M6_no_molecular_residual"; Ablation = "no_molecular_residual"; Extra = @() },
    @{ Label = "MS1_descriptors_with_context"; Ablation = "descriptors_with_context"; Extra = @() },
    @{ Label = "MS2_fingerprint_with_context"; Ablation = "fingerprint_with_context"; Extra = @() },
    @{ Label = "MS4_no_molecular_input"; Ablation = "no_molecular_input"; Extra = @() }
)
# MS3 is exactly the M1 input state: descriptors plus fingerprint, without any
# numeric, duration, taxonomic, or other categorical context. It is therefore
# reported as a named reuse of M1 rather than training a duplicate model.
$reusedAblationResults = @(
    @{ Label = "MS3_molecular_only_no_context"; Reuses = "M1_no_context"; Ablation = "no_context" }
)

function Assert-Python {
    if (-not (Test-Path $Python)) {
        throw "Python interpreter not found: $Python"
    }
}

function Initialize-Outputs {
    New-Item -ItemType Directory -Force -Path $outRoot, $summaryOut, $auditDir, $logDir | Out-Null
    if (-not (Test-Path $runTimes)) {
        "kind,run_name,ablation,split_name,start_iso,end_iso,duration_seconds,exit_code" | Set-Content -Encoding UTF8 $runTimes
    }
}

function Invoke-NetworkPreflight {
    if ($SkipNetworkCheck) {
        Write-Host "[network] preflight skipped by request"
        return
    }
    $networkCheckScript = "scripts\test_clash_overnight_connectivity.ps1"
    if (-not (Test-Path $networkCheckScript)) {
        Write-Host "[network] optional Clash preflight is not part of the portable training package; skipping"
        return
    }
    & pwsh -NoProfile -File $networkCheckScript -Attempts 4 -RetryDelaySeconds 15
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Clash preflight did not pass. Local training can continue because data and caches are local, but SHAP dependency installation or remote operations should not be started until the proxy is restored."
    }
}

function Invoke-Step {
    param(
        [string]$Kind,
        [string]$RunName,
        [string]$Ablation,
        [string]$SplitName,
        [scriptblock]$Command
    )
    $start = Get-Date
    Write-Host "[run-start] $($start.ToString('o')) kind=$Kind run=$RunName ablation=$Ablation split=$SplitName"
    & $Command
    $status = $LASTEXITCODE
    $end = Get-Date
    $duration = [int](New-TimeSpan -Start $start -End $end).TotalSeconds
    "$Kind,$RunName,$Ablation,$SplitName,$($start.ToString('o')),$($end.ToString('o')),$duration,$status" | Add-Content -Encoding UTF8 $runTimes
    Write-Host "[run-done] $($end.ToString('o')) kind=$Kind run=$RunName ablation=$Ablation split=$SplitName duration=${duration}s status=$status"
    if ($status -ne 0) {
        throw "Command failed: kind=$Kind run=$RunName ablation=$Ablation split=$SplitName exit_code=$status"
    }
}

function Get-FoldSplit {
    param([int]$Fold)
    return "${splitPrefix}E_random_5fold_fold$Fold"
}

function Get-SoilPtoxFoldSplit {
    param([int]$Fold)
    return "SoilPtoxQC2_E_random_5fold_fold$Fold"
}

function Get-SoilMgkgFoldSplit {
    param([int]$Fold)
    return "SoilMgkgQC2_E_random_5fold_fold$Fold"
}

function Get-RunDirectory {
    param(
        [string]$RunName,
        [string]$Ablation,
        [string]$SplitName
    )
    return (Join-Path $outRoot "$runVersion`_$RunName\deep\$Ablation\$SplitName")
}

function Test-RunExists {
    param(
        [string]$RunName,
        [string]$Ablation,
        [string]$SplitName
    )
    $runDir = Get-RunDirectory -RunName $RunName -Ablation $Ablation -SplitName $SplitName
    return (
        (Test-Path (Join-Path $runDir "predictions.csv")) -and
        (Test-Path (Join-Path $runDir "manifest.json")) -and
        (Test-Path (Join-Path $runDir "best_model.pt"))
    )
}

function Invoke-SplitBuild {
    param(
        [string]$TransferSplit,
        [string]$SoilPtoxSplit,
        [string]$SoilMgkgSplit
    )
    $routingAudit = Join-Path $auditDir "${TransferSplit}_routing_audit.csv"
    & $Python scripts\build_three_stage_ptox_to_soil_mgkg_split.py `
        --db $db `
        --source-table $sourceTable `
        --soil-ptox-source-table $soilPtoxTable `
        --soil-ptox-split-name $SoilPtoxSplit `
        --soil-mgkg-source-table $soilMgkgTable `
        --soil-mgkg-split-name $SoilMgkgSplit `
        --split-name $TransferSplit `
        --audit-csv $routingAudit `
        --seed 42
    if ($LASTEXITCODE -ne 0) {
        throw "Three-stage split creation failed: $TransferSplit"
    }
    Assert-RoutingAudit -TransferSplit $TransferSplit
}

function Assert-RoutingAudit {
    param([string]$TransferSplit)
    $routingAudit = Join-Path $auditDir "${TransferSplit}_routing_audit.csv"
    if (-not (Test-Path $routingAudit)) {
        throw "Routing audit is missing for split: $TransferSplit"
    }
    $rows = @(Import-Csv $routingAudit)
    if ($rows.Count -ne 1) {
        throw "Routing audit must contain exactly one data row: $routingAudit"
    }
    $audit = $rows[0]
    if ($audit.routing_rule -ne "exact_aggregate_or_raw_result_v1") {
        throw "Routing audit rule mismatch for split: $TransferSplit"
    }
    if (
        [int]$audit.residual_aggregate_id_overlap -ne 0 -or
        [int]$audit.residual_result_id_overlap -ne 0
    ) {
        throw "Routing audit contains residual stage-1/soil-pTox overlap: $routingAudit"
    }
    if (
        [int]$audit.aquatic_ptox_stage1_after + [int]$audit.aquatic_ptox_excluded_total -ne
        [int]$audit.aquatic_ptox_candidates_before
    ) {
        throw "Routing audit candidate accounting mismatch: $routingAudit"
    }
}

function Invoke-Random8Split {
    Invoke-SplitBuild -TransferSplit $random8Split -SoilPtoxSplit "SoilPtoxQC2_B_random_8_2" -SoilMgkgSplit "SoilMgkgQC2_B_random_8_2"
}

function Invoke-Splits {
    Invoke-Random8Split
    if ($RunFull5Fold) {
        foreach ($fold in $Folds) {
            Invoke-SplitBuild -TransferSplit (Get-FoldSplit -Fold $fold) -SoilPtoxSplit (Get-SoilPtoxFoldSplit -Fold $fold) -SoilMgkgSplit (Get-SoilMgkgFoldSplit -Fold $fold)
        }
    }
}

function Invoke-TrainOne {
    param(
        [string]$Kind,
        [string]$Label,
        [string]$Ablation,
        [string]$SplitName,
        [int]$Seed,
        [int]$Stage1Epochs,
        [int]$Stage2Epochs,
        [int]$Stage3Epochs,
        [string[]]$ExtraArgs = @()
    )
    # Keep repaired runs separate from earlier local outputs whose stage-1
    # model selection and split contracts were invalid.
    Assert-RoutingAudit -TransferSplit $SplitName
    $runName = "three_stage_${protocolTag}_${Label}_seed$Seed"
    if (Test-RunExists -RunName $runName -Ablation $Ablation -SplitName $SplitName) {
        Write-Host "[skip-existing] $runName $Ablation $SplitName"
        return
    }
    $trainArgs = @(
        $baseTrainArgs + @(
            "--run-name-zh", $runName,
            "--split-name", $SplitName,
            "--seed", "$Seed",
            "--epochs", "$Stage1Epochs",
            "--finetune-epochs", "$Stage2Epochs",
            "--finetune-mgkg-epochs", "$Stage3Epochs",
            "--ablation", $Ablation
        ) + $ExtraArgs
    )
    Invoke-Step -Kind $Kind -RunName $runName -Ablation $Ablation -SplitName $SplitName -Command {
        & $Python @trainArgs
    }
}

function Invoke-MainRuns {
    Invoke-Random8FullRun
    Invoke-Full5FoldRuns
}

function Invoke-Random8FullRun {
    foreach ($seed in $Seeds) {
        Invoke-TrainOne -Kind "full" -Label "full_no_adapter_random8_2" -Ablation "full" -SplitName $random8Split -Seed $seed -Stage1Epochs $PretrainEpochs -Stage2Epochs $SoilPtoxEpochs -Stage3Epochs $SoilMgkgEpochs
    }
}

function Invoke-Full5FoldRuns {
    if (-not $RunFull5Fold) {
        Write-Host "[fivefold] skipped by RunFull5Fold=false"
        return
    }
    foreach ($seed in $Seeds) {
        foreach ($fold in $Folds) {
            Invoke-TrainOne -Kind "full" -Label "full_no_adapter_random5fold_fold$fold" -Ablation "full" -SplitName (Get-FoldSplit -Fold $fold) -Seed $seed -Stage1Epochs $PretrainEpochs -Stage2Epochs $SoilPtoxEpochs -Stage3Epochs $SoilMgkgEpochs
        }
    }
}

function Invoke-AblationRuns {
    if (-not $RunAblations) {
        Write-Host "[ablations] skipped by RunAblations=false"
        return
    }
    foreach ($variant in $ablationMatrix) {
        Invoke-TrainOne `
            -Kind "ablation" `
            -Label $variant.Label `
            -Ablation $variant.Ablation `
            -SplitName $random8Split `
            -Seed $Seeds[0] `
            -Stage1Epochs $PretrainEpochs `
            -Stage2Epochs $SoilPtoxEpochs `
            -Stage3Epochs $SoilMgkgEpochs `
            -ExtraArgs $variant.Extra
    }
}

function Assert-ShapAvailable {
    & $Python -c "import shap; print(shap.__version__)"
    if ($LASTEXITCODE -ne 0) {
        throw "The active local environment lacks shap. Install the project ML extras before launching an overnight SHAP/PDP run."
    }
}

function Invoke-Explanation {
    Assert-RoutingAudit -TransferSplit $random8Split
    Assert-ShapAvailable
    $runName = "three_stage_${protocolTag}_full_no_adapter_random8_2_seed$($Seeds[0])"
    $modelDir = Get-RunDirectory -RunName $runName -Ablation "full" -SplitName $random8Split
    if (-not (Test-Path (Join-Path $modelDir "best_model.pt"))) {
        throw "Full random 8:2 model is missing: $modelDir"
    }
    $explainDir = Join-Path $modelDir "grouped_shap_pdp"
    if (Test-Path (Join-Path $explainDir "attribution_manifest.json")) {
        Write-Host "[skip-existing] grouped SHAP/PDP $explainDir"
        return
    }
    Invoke-Step -Kind "explain" -RunName $runName -Ablation "full" -SplitName $random8Split -Command {
        & $Python scripts\explain_grouped_shap.py `
            --model-dir $modelDir `
            --db $db `
            --source-table $sourceTable `
            --split-name $random8Split `
            --target-family "solid_neglog_mg_kg" `
            --out-dir $explainDir `
            --seed $Seeds[0] `
            --background-rows 48 `
            --explain-rows 32 `
            --shap-max-evals 1200 `
            --global-top-k 20 `
            --group-top-k 12 `
            --descriptor-pdp-top-k 5 `
            --fingerprint-contrast-top-k 12 `
            --device cuda
    }
}

function Invoke-Summary {
    Assert-RoutingAudit -TransferSplit $random8Split
    & $Python scripts\summarize_deep_runs.py `
        --root $outRoot `
        --out-dir $summaryOut `
        --run-times $runTimes
    if ($LASTEXITCODE -ne 0) {
        throw "summarize_deep_runs.py failed with exit code $LASTEXITCODE"
    }
    & $Python scripts\summarize_v1_2_39_three_stage_ablation.py `
        --root $outRoot `
        --out-dir $summaryOut
    if ($LASTEXITCODE -ne 0) {
        throw "summarize_v1_2_39_three_stage_ablation.py failed with exit code $LASTEXITCODE"
    }
}

Assert-Python
Initialize-Outputs
Invoke-NetworkPreflight
Write-Host "[start] $((Get-Date).ToString('o')) v1.2.39 strict three-stage pTox -> soil pTox -> soil mg/kg mode=$Mode"
Write-Host "[contract] adapters disabled; pTox head shared across aquatic/soil; mg/kg receives a separate final-stage head."

switch ($Mode) {
    "splits" { Invoke-Splits }
    "main" {
        Invoke-Splits
        Invoke-MainRuns
    }
    "ablations" {
        Invoke-Random8Split
        Invoke-AblationRuns
    }
    "explain" { Invoke-Explanation }
    "summarize" { Invoke-Summary }
    "smoke" {
        Invoke-Random8Split
        Invoke-TrainOne -Kind "smoke" -Label "smoke_full_no_adapter" -Ablation "full" -SplitName $random8Split -Seed $Seeds[0] -Stage1Epochs $SmokeEpochs -Stage2Epochs $SmokeEpochs -Stage3Epochs $SmokeEpochs
    }
    "all" {
        Invoke-Splits
        Invoke-Random8FullRun
        Invoke-Explanation
        Invoke-Full5FoldRuns
        Invoke-AblationRuns
        Invoke-Summary
    }
}

Write-Host "[done] $((Get-Date).ToString('o')) v1.2.39 strict three-stage mode=$Mode"
