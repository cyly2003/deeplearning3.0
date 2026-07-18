param(
    [ValidateSet("all", "splits", "formal", "smoke", "summarize")]
    [string]$Mode = "all",
    [int[]]$Seeds = @(42),
    [int[]]$Folds = @(1, 2, 3, 4, 5),
    [int]$Epochs = 30,
    [int]$SmokeEpochs = 1,
    [int]$BatchSize = 512,
    [string]$Python = ".venv-cuda\Scripts\python.exe",
    [string]$Device = "cuda:0",
    [bool]$RunRandom8 = $true,
    [bool]$Run5Fold = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$config = "configs\experiment.local.cuda3050ti.yaml"
$db = "outputs\derived\modeling_dataset_v2_0_0_rebuild.sqlite"
$sourceTable = "aggregated_task_records_soil_mg_kg_qc"
$outRoot = "outputs\experiments\v1_2_38_soil_mgkg_random_local"
$summaryOut = "outputs\experiments\v1_2_38_soil_mgkg_random_local_summary"
$logDir = "outputs\logs"
$runTimes = Join-Path $logDir "run_v1_2_38_soil_mgkg_random_local_times.csv"
$runVersion = "v1.2.38"
$splitPrefix = "SoilMgkgQC2_"
$random8Split = "${splitPrefix}B_random_8_2"

function Initialize-Outputs {
    param(
        [string]$Root,
        [string]$Summary,
        [string]$Times
    )
    New-Item -ItemType Directory -Force -Path $Root, $Summary, $logDir | Out-Null
    if (-not (Test-Path $Times)) {
        "kind,run_name,split_name,start_iso,end_iso,duration_seconds,exit_code" | Set-Content -Encoding UTF8 $Times
    }
}

function Assert-Python {
    if (-not (Test-Path $Python)) {
        throw "Python interpreter not found: $Python"
    }
}

function Invoke-Step {
    param(
        [string]$Kind,
        [string]$RunName,
        [string]$SplitName,
        [string]$TimesPath,
        [scriptblock]$Command
    )
    $start = Get-Date
    Write-Host "[run-start] $($start.ToString("o")) kind=$Kind run=$RunName split=$SplitName"
    & $Command
    $status = $LASTEXITCODE
    $end = Get-Date
    $duration = [int](New-TimeSpan -Start $start -End $end).TotalSeconds
    "$Kind,$RunName,$SplitName,$($start.ToString("o")),$($end.ToString("o")),$duration,$status" | Add-Content -Encoding UTF8 $TimesPath
    Write-Host "[run-done] $($end.ToString("o")) kind=$Kind run=$RunName split=$SplitName duration=${duration}s status=$status"
    if ($status -ne 0) {
        throw "Command failed: kind=$Kind run=$RunName split=$SplitName exit_code=$status"
    }
}

function Invoke-Splits {
    Write-Host "[split-build] source_table=$sourceTable prefix=$splitPrefix"
    & $Python -m qsar_tl.cli generate-experiment-splits `
        --config $config `
        --db $db `
        --source-table $sourceTable `
        --split-name-prefix $splitPrefix `
        --split-codes B E `
        --seed 42
    if ($LASTEXITCODE -ne 0) {
        throw "Split generation failed with exit code $LASTEXITCODE"
    }
}

function Get-FoldSplit {
    param([int]$Fold)
    return "${splitPrefix}E_random_5fold_fold$Fold"
}

function Test-RunExists {
    param(
        [string]$Root,
        [string]$RunName,
        [string]$SplitName
    )
    $runDir = Join-Path $Root "$runVersion`_$RunName\deep\full\$SplitName"
    return (
        (Test-Path (Join-Path $runDir "predictions.csv")) -and
        (Test-Path (Join-Path $runDir "manifest.json")) -and
        (Test-Path (Join-Path $runDir "history.csv"))
    )
}

function Invoke-TrainOne {
    param(
        [string]$Root,
        [string]$TimesPath,
        [string]$SplitName,
        [int]$Seed,
        [string]$Label,
        [int]$EpochCount,
        [string]$RunNamePrefix = ""
    )
    $runName = "${RunNamePrefix}soil_mgkg_no_adapter_${Label}_seed${Seed}_cebin_lw0025_no_censored"
    if (Test-RunExists -Root $Root -RunName $runName -SplitName $SplitName) {
        Write-Host "[skip-existing] $runName $SplitName"
        return
    }

    $trainArgs = @(
        "-m", "qsar_tl.training.train",
        "--config", $config,
        "--db", $db,
        "--out-dir", $Root,
        "--run-version", $runVersion,
        "--run-name-zh", $runName,
        "--split-name", $SplitName,
        "--source-table", $sourceTable,
        "--seed", "$Seed",
        "--epochs", "$EpochCount",
        "--finetune-epochs", "0",
        "--batch-size", "$BatchSize",
        "--scheduler", "cosine",
        "--target-standardization", "per_task_target",
        "--no-medium-adapters",
        "--device", $Device,
        "--metric-min-n", "5",
        "--early-stopping",
        "--early-stopping-patience", "15",
        "--early-stopping-min-delta", "0.0",
        # Select the direct mg/kg baseline only from its training rows; the
        # held-out mg/kg test partition is never used for model selection.
        "--monitor-split", "internal_train_fraction",
        "--validation-fraction", "0.1",
        "--source-weighting-method", "none",
        "--source-weighting-alpha", "1.0",
        "--toxicity-binning",
        "--toxicity-binning-mode", "aux_classification",
        "--toxicity-binning-scheme", "authority_v1",
        "--toxicity-binning-loss-weight", "0.025",
        # Match the v1.2.39 final-stage protocol: neither paired mg/kg model
        # claims a censored-loss contribution.
        "--no-censored-loss",
        "--no-effect-level-weighting",
        "--learning-rate", "0.0005",
        "--dropout", "0.10",
        "--weight-decay", "0.000009856751793848817",
        "--ablation", "full"
    )

    Invoke-Step -Kind "soil_mgkg" -RunName $runName -SplitName $SplitName -TimesPath $TimesPath -Command {
        & $Python @trainArgs
    }
}

function Invoke-Matrix {
    param(
        [string]$Root,
        [string]$TimesPath,
        [int[]]$RunSeeds,
        [int[]]$RunFolds,
        [int]$EpochCount,
        [string]$RunNamePrefix = ""
    )
    foreach ($seed in $RunSeeds) {
        if ($RunRandom8) {
            Invoke-TrainOne -Root $Root -TimesPath $TimesPath -SplitName $random8Split -Seed $seed -Label "random8_2" -EpochCount $EpochCount -RunNamePrefix $RunNamePrefix
        }
        if ($Run5Fold) {
            foreach ($fold in $RunFolds) {
                Invoke-TrainOne -Root $Root -TimesPath $TimesPath -SplitName (Get-FoldSplit -Fold $fold) -Seed $seed -Label "random5fold_fold$fold" -EpochCount $EpochCount -RunNamePrefix $RunNamePrefix
            }
        }
    }
}

function Invoke-Summary {
    param(
        [string]$Root,
        [string]$Summary,
        [string]$TimesPath
    )
    & $Python scripts\summarize_deep_runs.py `
        --root $Root `
        --out-dir $Summary `
        --run-times $TimesPath
    if ($LASTEXITCODE -ne 0) {
        throw "summarize_deep_runs.py failed with exit code $LASTEXITCODE"
    }
}

Assert-Python
Initialize-Outputs -Root $outRoot -Summary $summaryOut -Times $runTimes

Write-Host "[start] $((Get-Date).ToString("o")) v1.2.38 local soil mg/kg mode=$Mode seeds=$($Seeds -join ',') folds=$($Folds -join ',')"

switch ($Mode) {
    "splits" {
        Invoke-Splits
    }
    "formal" {
        Invoke-Matrix -Root $outRoot -TimesPath $runTimes -RunSeeds $Seeds -RunFolds $Folds -EpochCount $Epochs
    }
    "summarize" {
        Invoke-Summary -Root $outRoot -Summary $summaryOut -TimesPath $runTimes
    }
    "smoke" {
        $smokeRoot = "${outRoot}_smoke"
        $smokeSummary = "${summaryOut}_smoke"
        $smokeTimes = Join-Path $logDir "run_v1_2_38_soil_mgkg_random_local_smoke_times.csv"
        Initialize-Outputs -Root $smokeRoot -Summary $smokeSummary -Times $smokeTimes
        Invoke-Splits
        Invoke-Matrix -Root $smokeRoot -TimesPath $smokeTimes -RunSeeds @($Seeds[0]) -RunFolds @(1) -EpochCount $SmokeEpochs -RunNamePrefix "smoke_"
        Invoke-Summary -Root $smokeRoot -Summary $smokeSummary -TimesPath $smokeTimes
    }
    "all" {
        Invoke-Splits
        Invoke-Matrix -Root $outRoot -TimesPath $runTimes -RunSeeds $Seeds -RunFolds $Folds -EpochCount $Epochs
        Invoke-Summary -Root $outRoot -Summary $summaryOut -TimesPath $runTimes
    }
}

Write-Host "[done] $((Get-Date).ToString("o")) v1.2.38 local soil mg/kg mode=$Mode"
