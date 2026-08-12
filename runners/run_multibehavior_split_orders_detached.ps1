[CmdletBinding()]
param(
    [switch]$ForceTrain,
    [int]$MaxEpochs = 100,
    [int]$Patience = 4,
    [int]$BatchSize = 64,
    [int]$NumWorkers = 4,
    [int]$Seed = 42,
    [double]$LearningRate = 0.00005,
    [double]$LrPlateauFactor = 0.5,
    [int]$LrPlateauPatience = 1,
    [double]$MinLearningRate = 0.000001,
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [string]$DataRoot = "C:\Realtime\brayden_data",
    [string]$OutputRoot,
    [string]$PythonExecutable = "C:\Users\Jimi\anaconda3\envs\realtime_train_lightning\python.exe",
    [string]$NotifyEmail = "braydenk@andrew.cmu.edu",
    [switch]$NoNotify,
    [switch]$NoProgressBar,
    [switch]$Worker,
    [string]$RunDirectory
)

$ErrorActionPreference = "Stop"

<#
Runs six sequential multibehavioral experiments, one for each global
chronological permutation of train/val/test. Every recording uses the same
order within a fold. Each fold gets its own checkpoint, metrics, predictions,
confusion matrices, and split config. The launcher starts a hidden worker and
returns immediately.

Example:
  powershell -ExecutionPolicy Bypass -File _models/run_multibehavior_split_orders_detached.ps1
#>

$RepoRoot = Split-Path -Parent $PSScriptRoot
$TrainScript = Join-Path $RepoRoot "benchmarking\code\train_multibehavior.py"
$TraditionalScript = Join-Path $RepoRoot "benchmarking\code\evaluate_multibehavior_traditional.py"
$SegmentScript = Join-Path $RepoRoot "benchmarking\code\evaluate_multibehavior_segments.py"
$NotificationScript = Join-Path $RepoRoot "benchmarking\code\send_behavior_benchmark_email.py"
$BaseSplitConfig = Join-Path $RepoRoot "benchmarking\configs\brayden_binary_chronological_splits.json"
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $RepoRoot "benchmarking\figures\multibehavior_split_orders_lr_fixed"
}

foreach ($EmailVariableName in @("GMAIL_ADDRESS", "GMAIL_APP_PASSWORD")) {
    $SavedEmailValue = [Environment]::GetEnvironmentVariable($EmailVariableName, "User")
    if ($SavedEmailValue) {
        Set-Item -LiteralPath "Env:$EmailVariableName" -Value $SavedEmailValue
    }
}

function ConvertTo-ProcessArgument {
    param([AllowEmptyString()][string]$Value)
    if ($Value -notmatch '[\s"]') { return $Value }
    return '"' + ($Value -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    "[$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))] START $Name" |
        Add-Content -LiteralPath (Join-Path $RunDirectory "pipeline.log") -Encoding UTF8
    & $PythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
    "[$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))] DONE  $Name" |
        Add-Content -LiteralPath (Join-Path $RunDirectory "pipeline.log") -Encoding UTF8
}

function Send-Notification {
    param(
        [ValidateSet("success", "failure")][string]$Status,
        [ValidateSet("model", "pipeline")][string]$Scope,
        [string]$Name,
        [string]$ModelOutput,
        [int]$Completed
    )
    if ($NoNotify) { return }
    $Arguments = @(
        "-u", $NotificationScript, "--recipient", $NotifyEmail,
        "--status", $Status, "--scope", $Scope, "--name", $Name,
        "--run-directory", $RunDirectory, "--completed", $Completed.ToString(),
        "--total", "6"
    )
    if ($ModelOutput) { $Arguments += @("--output-dir", $ModelOutput) }
    & $PythonExecutable @Arguments
}

foreach ($RequiredPath in @(
    $PythonExecutable, $TrainScript, $TraditionalScript, $SegmentScript,
    $NotificationScript, $BaseSplitConfig, $DataRoot
)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) { throw "Required path not found: $RequiredPath" }
}
if ($MaxEpochs -lt 1 -or $Patience -lt 0 -or $BatchSize -lt 1 -or $NumWorkers -lt 0) {
    throw "MaxEpochs/BatchSize must be positive; Patience/NumWorkers cannot be negative."
}
if ($LearningRate -le 0 -or $LrPlateauFactor -le 0 -or $LrPlateauFactor -ge 1 -or
    $LrPlateauPatience -lt 0 -or $MinLearningRate -le 0) {
    throw "Learning-rate settings are invalid."
}

$Orders = @(
    [pscustomobject]@{ Name = "train_val_test"; Values = @("train", "val", "test") },
    [pscustomobject]@{ Name = "train_test_val"; Values = @("train", "test", "val") },
    [pscustomobject]@{ Name = "val_train_test"; Values = @("val", "train", "test") },
    [pscustomobject]@{ Name = "val_test_train"; Values = @("val", "test", "train") },
    [pscustomobject]@{ Name = "test_train_val"; Values = @("test", "train", "val") },
    [pscustomobject]@{ Name = "test_val_train"; Values = @("test", "val", "train") }
)

if ($Worker) {
    if ([string]::IsNullOrWhiteSpace($RunDirectory)) { throw "The internal worker requires -RunDirectory." }
    Set-Location -LiteralPath $RepoRoot
    New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
    $StatusPath = Join-Path $RunDirectory "pipeline.log"
    $ExitCodePath = Join-Path $RunDirectory "exit_code.txt"
    "[$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))] Six-order worker started." |
        Set-Content -LiteralPath $StatusPath -Encoding UTF8
    $Completed = 0
    $CurrentName = "initialization"
    $CurrentOutput = $null
    $Rows = @()
    $PipelineExitCode = 0
    try {
        foreach ($Order in $Orders) {
            $CurrentName = $Order.Name
            $FoldRoot = Join-Path $OutputRoot $Order.Name
            $CurrentOutput = Join-Path $FoldRoot "multibehavioral"
            New-Item -ItemType Directory -Path $FoldRoot -Force | Out-Null

            $Config = Get-Content -LiteralPath $BaseSplitConfig -Raw | ConvertFrom-Json
            $SplitOrderByDataset = @()
            foreach ($DatasetName in $Config.dataset_names) {
                $SplitOrderByDataset += ,([string[]]$Order.Values)
            }
            $Config.multibehavioral.split_order_by_dataset = $SplitOrderByDataset
            $ConfigPath = Join-Path $FoldRoot "split_config.json"
            $ConfigJson = $Config | ConvertTo-Json -Depth 20
            [System.IO.File]::WriteAllText(
                $ConfigPath,
                $ConfigJson,
                [System.Text.UTF8Encoding]::new($false)
            )

            $Checkpoint = Join-Path $CurrentOutput "model.ckpt"
            if ((Test-Path -LiteralPath $Checkpoint -PathType Leaf) -and -not $ForceTrain) {
                "[$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))] SKIP  train $($Order.Name) (checkpoint exists)" |
                    Add-Content -LiteralPath $StatusPath -Encoding UTF8
            } else {
                $TrainArguments = @(
                    "-u", $TrainScript, "--data-root", $DataRoot, "--output-root", $FoldRoot,
                    "--split-config", $ConfigPath, "--max-epochs", $MaxEpochs.ToString(),
                    "--patience", $Patience.ToString(), "--batch-size", $BatchSize.ToString(),
                    "--num-workers", $NumWorkers.ToString(), "--seed", $Seed.ToString(),
                    "--accelerator", $Device, "--learning-rate", $LearningRate.ToString("R"),
                    "--lr-plateau-factor", $LrPlateauFactor.ToString("R"),
                    "--lr-plateau-patience", $LrPlateauPatience.ToString(),
                    "--min-learning-rate", $MinLearningRate.ToString("R")
                )
                if ($ForceTrain) { $TrainArguments += "--force" }
                if ($NoProgressBar) { $TrainArguments += "--no-progress-bar" }
                Invoke-PythonStep "train $($Order.Name)" $TrainArguments
            }

            $Predictions = Join-Path $CurrentOutput "test_predictions.npz"
            $TraditionalOutput = Join-Path $CurrentOutput "traditional_metrics.json"
            $SegmentOutput = Join-Path $CurrentOutput "segment_metrics.json"
            $ConfusionFigure = Join-Path $CurrentOutput "confusion_matrices.png"
            Invoke-PythonStep "traditional metrics $($Order.Name)" @(
                "-u", $TraditionalScript, "--checkpoint", $Checkpoint,
                "--output", $TraditionalOutput, "--confusion-figure", $ConfusionFigure,
                "--predictions-output", $Predictions, "--data-root", $DataRoot,
                "--split-config", $ConfigPath, "--batch-size", $BatchSize.ToString(),
                "--num-workers", $NumWorkers.ToString(), "--device", $Device,
                "--seed", $Seed.ToString()
            )
            Invoke-PythonStep "segment metrics $($Order.Name)" @(
                "-u", $SegmentScript, "--predictions", $Predictions,
                "--output", $SegmentOutput, "--data-root", $DataRoot,
                "--split-config", $ConfigPath, "--seed", $Seed.ToString()
            )

            $Metrics = Get-Content -LiteralPath $TraditionalOutput -Raw | ConvertFrom-Json
            $Rows += [pscustomobject]@{
                split_order = ($Order.Values -join "->")
                accuracy = $Metrics.accuracy
                mcc = $Metrics.mcc
                macro_precision = $Metrics.macro_precision
                macro_recall = $Metrics.macro_recall
                macro_f1 = $Metrics.macro_f1
                output_directory = $CurrentOutput
            }
            $Rows | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $OutputRoot "combined_results.json") -Encoding UTF8
            $Rows | Export-Csv -LiteralPath (Join-Path $OutputRoot "combined_results.csv") -NoTypeInformation
            $Completed += 1
            Send-Notification "success" "model" "multibehavioral $($Order.Name)" $CurrentOutput $Completed
        }
        Send-Notification "success" "pipeline" "six multibehavior split orders" $OutputRoot $Completed
    } catch {
        $_ | Out-String | Add-Content -LiteralPath $StatusPath -Encoding UTF8
        $PipelineExitCode = 1
        Send-Notification "failure" "model" "multibehavioral $CurrentName" $CurrentOutput $Completed
    }
    "[$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))] Worker finished with exit code $PipelineExitCode." |
        Add-Content -LiteralPath $StatusPath -Encoding UTF8
    $PipelineExitCode | Set-Content -LiteralPath $ExitCodePath
    exit $PipelineExitCode
}

$Timestamp = [DateTime]::Now.ToString("yyyyMMdd_HHmmss")
$NewRunDirectory = Join-Path (Join-Path $PSScriptRoot "benchmark_runs") "${Timestamp}_multibehavior_six_split_orders"
New-Item -ItemType Directory -Path $NewRunDirectory -Force | Out-Null
$PowerShellHost = (Get-Process -Id $PID).Path
$ChildArguments = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath,
    "-Worker", "-RunDirectory", $NewRunDirectory,
    "-MaxEpochs", $MaxEpochs.ToString(), "-Patience", $Patience.ToString(),
    "-BatchSize", $BatchSize.ToString(), "-NumWorkers", $NumWorkers.ToString(),
    "-Seed", $Seed.ToString(), "-Device", $Device, "-DataRoot", $DataRoot,
    "-OutputRoot", $OutputRoot, "-PythonExecutable", $PythonExecutable,
    "-NotifyEmail", $NotifyEmail, "-LearningRate", $LearningRate.ToString("R"),
    "-LrPlateauFactor", $LrPlateauFactor.ToString("R"),
    "-LrPlateauPatience", $LrPlateauPatience.ToString(),
    "-MinLearningRate", $MinLearningRate.ToString("R")
)
if ($ForceTrain) { $ChildArguments += "-ForceTrain" }
if ($NoProgressBar) { $ChildArguments += "-NoProgressBar" }
if ($NoNotify) { $ChildArguments += "-NoNotify" }
$QuotedChildArguments = $ChildArguments | ForEach-Object { ConvertTo-ProcessArgument $_ }
$OutputLog = Join-Path $NewRunDirectory "pipeline.out.log"
$ErrorLog = Join-Path $NewRunDirectory "pipeline.err.log"
$WorkerProcess = Start-Process -FilePath $PowerShellHost -ArgumentList $QuotedChildArguments `
    -WorkingDirectory $RepoRoot -RedirectStandardOutput $OutputLog `
    -RedirectStandardError $ErrorLog -WindowStyle Hidden -PassThru
$WorkerProcess.Id | Set-Content -LiteralPath (Join-Path $NewRunDirectory "worker.pid")
@(
    "Started: $([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))"
    "Orders: all 6 global train/val/test permutations"
    "Data: $DataRoot"
    "Results: $OutputRoot"
    "LR: $LearningRate; ReduceLROnPlateau factor=$LrPlateauFactor patience=$LrPlateauPatience min=$MinLearningRate"
) | Set-Content -LiteralPath (Join-Path $NewRunDirectory "run_info.txt")

Write-Host "Six-order multibehavior benchmark started independently of VS Code."
Write-Host "Worker PID:    $($WorkerProcess.Id)"
Write-Host "Run directory: $NewRunDirectory"
Write-Host "Results:       $OutputRoot"
Write-Host "Follow output: Get-Content -LiteralPath `"$OutputLog`" -Tail 30 -Wait"
Write-Host "Stop command:  taskkill.exe /PID $($WorkerProcess.Id) /T /F"
