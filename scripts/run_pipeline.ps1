<#
.SYNOPSIS
    End-to-end execution pipeline for Medical Image Compression Benchmark.

.DESCRIPTION
    Runs all pipeline stages in sequence or individually.
    Automatically checks for existing output files and skips execution unless -Force is supplied.
    Outputs execution logs to logs/<stage>.log.

.PARAMETER Stage
    Specific stage(s) to run: data, baselines, dct, dwt, lossless, roi,
    segmentation, classifier, benchmark, graded, controller, assets, all. Default is 'all'.

.PARAMETER Force
    Re-run the stage even if expected output files already exist.

.EXAMPLE
    .\scripts\run_pipeline.ps1 -Stage assets
    .\scripts\run_pipeline.ps1 -Force
#>

[CmdletBinding()]
param (
    [Parameter(Position = 0)]
    [ValidateSet('all', 'data', 'baselines', 'dct', 'dwt', 'lossless', 'roi', 'segmentation', 'classifier', 'benchmark', 'graded', 'controller', 'assets')]
    [string[]]$Stage = @('all'),

    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptRoot
Set-Location $ProjectRoot

# Ensure logs directory exists
$LogDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

function Run-StageStep {
    param (
        [string]$StageName,
        [string]$Description,
        [string[]]$ExpectedOutputs,
        [string]$CommandLine
    )

    Write-Host "`n========================================================" -ForegroundColor Cyan
    Write-Host " STAGE: $StageName" -ForegroundColor Cyan
    Write-Host " $Description" -ForegroundColor DarkGray
    Write-Host "========================================================" -ForegroundColor Cyan

    $OutputsExist = $true
    foreach ($out in $ExpectedOutputs) {
        $fullPath = Join-Path $ProjectRoot $out
        if (-not (Test-Path $fullPath)) {
            $OutputsExist = $false
            break
        }
    }

    if ($OutputsExist -and (-not $Force)) {
        Write-Host " [SKIP] Expected outputs already exist. (Use -Force to rerun)" -ForegroundColor Yellow
        foreach ($out in $ExpectedOutputs) {
            Write-Host "        -> $out" -ForegroundColor DarkGray
        }
        return
    }

    $LogFile = Join-Path $LogDir "$($StageName.ToLower()).log"
    $Timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    "[$Timestamp] Starting stage: $StageName ($CommandLine)" | Out-File $LogFile -Append -Encoding utf8

    Write-Host " [EXEC] Running: $CommandLine" -ForegroundColor Green
    Write-Host " [LOG]  Writing to $LogFile" -ForegroundColor DarkGray

    try {
        $process = Start-Process -FilePath "powershell.exe" -ArgumentList "-Command `"$CommandLine`"" -NoNewWindow -PassThru -Wait -RedirectStandardOutput $LogFile -RedirectStandardError (Join-Path $LogDir "$($StageName.ToLower())_err.log")
        
        if ($process.ExitCode -ne 0) {
            Write-Error "Stage $StageName failed with exit code $($process.ExitCode). Check log: $LogFile"
        } else {
            Write-Host " [DONE] Stage $StageName completed successfully." -ForegroundColor Green
        }
    }
    catch {
        Write-Error "Exception occurred running $StageName : $_"
    }
}

$StagesToRun = if ($Stage -contains 'all') {
    @('data', 'baselines', 'dct', 'dwt', 'lossless', 'roi', 'segmentation', 'classifier', 'benchmark', 'graded', 'controller', 'assets')
} else {
    $Stage
}

Write-Host "Medical Image Compression Pipeline Execution" -ForegroundColor Magenta
Write-Host "Target Stages: $($StagesToRun -join ', ')" -ForegroundColor DarkGray
Write-Host "Force Re-run:  $Force" -ForegroundColor DarkGray

# 1. DATA
if ($StagesToRun -contains 'data') {
    Run-StageStep -StageName "data" `
        -Description "Build manifest, preprocess images to 512x512, and generate stratified folds" `
        -ExpectedOutputs @("data/manifest.csv", "data/splits/folds.json") `
        -CommandLine "python scripts/build_manifest.py; python scripts/preprocess.py; python scripts/make_folds.py"
}

# 2. BASELINES
if ($StagesToRun -contains 'baselines') {
    Run-StageStep -StageName "baselines" `
        -Description "Run PNG, JPEG, and JPEG 2000 standard baselines and summaries" `
        -ExpectedOutputs @("results/baselines.csv", "results/baseline_summary.csv") `
        -CommandLine "python scripts/run_baselines.py; python scripts/summarize_baselines.py"
}

# 3. DCT
if ($StagesToRun -contains 'dct') {
    Run-StageStep -StageName "dct" `
        -Description "Run custom 8x8 DCT transform codec and generate rate-distortion summaries" `
        -ExpectedOutputs @("results/dct_results.csv", "results/dct_summary.csv") `
        -CommandLine "python scripts/run_dct.py; python scripts/summarize_dct.py"
}

# 4. DWT
if ($StagesToRun -contains 'dwt') {
    Run-StageStep -StageName "dwt" `
        -Description "Run custom bior2.2 DWT codec and tuning grid search across wavelets" `
        -ExpectedOutputs @("results/dwt_results.csv", "results/dwt_summary.csv", "results/dwt_tuning_summary.csv") `
        -CommandLine "python scripts/run_dwt.py; python scripts/summarize_dwt.py; python scripts/tune_dwt.py"
}

# 5. LOSSLESS
if ($StagesToRun -contains 'lossless') {
    Run-StageStep -StageName "lossless" `
        -Description "Run bit-exact MED predictor lossless lung coding on masked cohort" `
        -ExpectedOutputs @("results/lossless_results.csv", "results/lossless_summary.csv") `
        -CommandLine "python scripts/run_lossless.py"
}

# 6. ROI
if ($StagesToRun -contains 'roi') {
    Run-StageStep -StageName "roi" `
        -Description "Run hybrid ROI codecs (ground truth and U-Net predicted masks)" `
        -ExpectedOutputs @("results/roi_codec_results.csv", "results/roi_codec_summary.csv", "results/roi_codec_predicted_summary.csv") `
        -CommandLine "python scripts/run_roi_codec.py; python scripts/run_roi_codec_predicted.py"
}

# 7. SEGMENTATION
if ($StagesToRun -contains 'segmentation') {
    Run-StageStep -StageName "segmentation" `
        -Description "Train and evaluate 5-fold out-of-fold U-Net lung segmentation" `
        -ExpectedOutputs @("results/segmentation_results.csv", "results/segmentation_summary.csv") `
        -CommandLine "python scripts/train_unet.py; python scripts/evaluate_segmentation.py"
}

# 8. CLASSIFIER
if ($StagesToRun -contains 'classifier') {
    Run-StageStep -StageName "classifier" `
        -Description "Train ResNet-18 classifier and EfficientNet-B0 judge across 5 folds" `
        -ExpectedOutputs @("results/classifier_metrics.csv", "results/judge_oof.csv") `
        -CommandLine "python scripts/train_classifier.py; python scripts/train_judge.py; python scripts/evaluate_classifier.py"
}

# 9. BENCHMARK
if ($StagesToRun -contains 'benchmark') {
    Run-StageStep -StageName "benchmark" `
        -Description "Run downstream diagnostic fidelity benchmark across all compression ladders" `
        -ExpectedOutputs @("results/benchmark_summary.csv", "results/benchmark_operating_points.csv") `
        -CommandLine "python scripts/run_benchmark.py; python scripts/evaluate_benchmark.py"
}

# 10. GRADED
if ($StagesToRun -contains 'graded') {
    Run-StageStep -StageName "graded" `
        -Description "Run near-lossless graded multi-zone pilot experiments" `
        -ExpectedOutputs @("results/graded_pilot_results.csv", "results/graded_pilot_summary.csv") `
        -CommandLine "python scripts/run_graded_pilot.py; python scripts/summarize_graded_pilot.py"
}

# 11. CONTROLLER
if ($StagesToRun -contains 'controller') {
    Run-StageStep -StageName "controller" `
        -Description "Evaluate adaptive closed-loop bitrate controller and variants against judge" `
        -ExpectedOutputs @("results/controller_summary.csv", "results/controller_vs_fixed.csv") `
        -CommandLine "python scripts/evaluate_controller.py; python scripts/evaluate_controller_variants.py"
}

# 12. ASSETS
if ($StagesToRun -contains 'assets') {
    Run-StageStep -StageName "assets" `
        -Description "Generate audited tables, publication figures, claims register, report docs and number checks" `
        -ExpectedOutputs @("report_assets/claims_register.md", "report/report.md", "report/report.docx", "report/report.pdf") `
        -CommandLine "python scripts/make_report_assets.py; python scripts/build_report_docs.py; python scripts/check_report_numbers.py"
}

Write-Host "`nPipeline run finished." -ForegroundColor Cyan
