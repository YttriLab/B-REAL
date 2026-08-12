[CmdletBinding(DefaultParameterSetName = "Selected")]
param(
    [Parameter(ParameterSetName = "Selected", Mandatory = $true)]
    [string[]]$Behaviors,
    [Parameter(ParameterSetName = "All", Mandatory = $true)]
    [switch]$AllBehaviors,
    [Parameter(ParameterSetName = "Multi", Mandatory = $true)]
    [switch]$Multibehavioral,
    [switch]$ForceTrain,
    [switch]$ShuffleSplit,
    [int]$MaxEpochs = 100,
    [int]$Patience = 4,
    [int]$BatchSize = 64,
    [int]$NumWorkers = 4,
    [int]$Seed = 42,
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
Trains either independent binary TSM models sequentially or one joint
multibehavioral TSM model. The launcher immediately starts a hidden worker, so
closing this terminal or VS Code does not stop the pipeline.

Examples:
  powershell -ExecutionPolicy Bypass -File _models/run_binary_behavior_benchmarks_detached.ps1 -AllBehaviors
  powershell -ExecutionPolicy Bypass -File _models/run_binary_behavior_benchmarks_detached.ps1 -Behaviors locomotion,rear,itch
  powershell -ExecutionPolicy Bypass -File _models/run_binary_behavior_benchmarks_detached.ps1 -Multibehavioral
  powershell -ExecutionPolicy Bypass -File _models/run_binary_behavior_benchmarks_detached.ps1 -Behaviors body_groom -ForceTrain

By default, an existing <behavior>/model.ckpt (or multibehavioral/model.ckpt)
skips training. -ForceTrain replaces it. Metrics are always recomputed from
the held-out test split.
#>

$RepoRoot = Split-Path -Parent $PSScriptRoot
$TrainScript = Join-Path $RepoRoot "benchmarking\code\train_binary_behavior.py"
$TraditionalScript = Join-Path $RepoRoot "benchmarking\code\evaluate_binary_traditional.py"
$SegmentScript = Join-Path $RepoRoot "benchmarking\code\evaluate_binary_segments.py"
$MultiTrainScript = Join-Path $RepoRoot "benchmarking\code\train_multibehavior.py"
$MultiTraditionalScript = Join-Path $RepoRoot "benchmarking\code\evaluate_multibehavior_traditional.py"
$MultiSegmentScript = Join-Path $RepoRoot "benchmarking\code\evaluate_multibehavior_segments.py"
$NotificationScript = Join-Path $RepoRoot "benchmarking\code\send_behavior_benchmark_email.py"
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $RepoRoot "benchmarking\figures\binary_behaviors"
}

# Import credentials persisted by _models/setup_test_email.ps1. This matters
# when the launcher terminal was opened before those user variables were set.
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

function ConvertTo-BehaviorSlug {
    param([string]$Value)
    return (($Value.Trim().ToLowerInvariant() -replace '[^a-z0-9]+', '_').Trim('_'))
}

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    "[$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))] START $Name" |
        Add-Content -LiteralPath (Join-Path $RunDirectory "pipeline.log") -Encoding UTF8
    & $PythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    "[$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))] DONE  $Name" |
        Add-Content -LiteralPath (Join-Path $RunDirectory "pipeline.log") -Encoding UTF8
}

function Send-BenchmarkNotification {
    param(
        [ValidateSet("success", "failure")][string]$Status,
        [ValidateSet("model", "pipeline")][string]$Scope,
        [string]$Name,
        [string]$ModelOutput,
        [int]$Completed,
        [int]$Total
    )
    if ($NoNotify) { return }
    $Arguments = @(
        "-u", $NotificationScript, "--recipient", $NotifyEmail,
        "--status", $Status, "--scope", $Scope, "--name", $Name,
        "--run-directory", $RunDirectory, "--completed", $Completed.ToString(),
        "--total", $Total.ToString()
    )
    if (-not [string]::IsNullOrWhiteSpace($ModelOutput)) {
        $Arguments += @("--output-dir", $ModelOutput)
    }
    # Notification failures are deliberately non-fatal. The helper catches
    # SMTP errors itself; this protects against an unexpected helper failure.
    & $PythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        "[$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))] WARNING notification helper exited $LASTEXITCODE" |
            Add-Content -LiteralPath (Join-Path $RunDirectory "pipeline.log") -Encoding UTF8
    }
}

foreach ($RequiredPath in @(
    $PythonExecutable, $TrainScript, $TraditionalScript, $SegmentScript,
    $MultiTrainScript, $MultiTraditionalScript, $MultiSegmentScript,
    $NotificationScript, $DataRoot
)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required path not found: $RequiredPath"
    }
}
if ($MaxEpochs -lt 1 -or $Patience -lt 0 -or $BatchSize -lt 1 -or $NumWorkers -lt 0) {
    throw "MaxEpochs/BatchSize must be positive; Patience/NumWorkers cannot be negative."
}

if ($AllBehaviors) {
    $Behaviors = @(& $PythonExecutable $TrainScript --data-root $DataRoot --list-behaviors)
    if ($LASTEXITCODE -ne 0 -or $Behaviors.Count -eq 0) {
        throw "Could not discover behaviors from $DataRoot"
    }
}
if (-not $Multibehavioral) {
    $Behaviors = @(
        $Behaviors |
            ForEach-Object { $_ -split ',' } |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ } |
            Select-Object -Unique
    )
    if ($Behaviors.Count -eq 0) { throw "No behaviors were selected." }
}

if ($Worker) {
    if ([string]::IsNullOrWhiteSpace($RunDirectory)) { throw "The internal worker requires -RunDirectory." }
    Set-Location -LiteralPath $RepoRoot
    $StatusPath = Join-Path $RunDirectory "pipeline.log"
    $ExitCodePath = Join-Path $RunDirectory "exit_code.txt"
    "[$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))] Behavior benchmark worker started." |
        Set-Content -LiteralPath $StatusPath -Encoding UTF8
    $PipelineExitCode = 0
    $CurrentModelName = if ($Multibehavioral) { "multibehavioral" } else { "behavior queue" }
    $CurrentModelOutput = $null
    $TotalModels = if ($Multibehavioral) { 1 } else { $Behaviors.Count }
    $CompletedModels = 0
    try {
        if ($Multibehavioral) {
            $CurrentModelName = "multibehavioral"
            $BehaviorOutput = Join-Path $OutputRoot "multibehavioral"
            $CurrentModelOutput = $BehaviorOutput
            $Checkpoint = Join-Path $BehaviorOutput "model.ckpt"
            $Predictions = Join-Path $BehaviorOutput "test_predictions.npz"
            $TraditionalOutput = Join-Path $BehaviorOutput "traditional_metrics.json"
            $SegmentOutput = Join-Path $BehaviorOutput "segment_metrics.json"
            $ConfusionFigure = Join-Path $BehaviorOutput "confusion_matrices.png"
            if ((Test-Path -LiteralPath $Checkpoint -PathType Leaf) -and -not $ForceTrain) {
                "[$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))] SKIP  train multibehavioral (checkpoint exists)" |
                    Add-Content -LiteralPath $StatusPath -Encoding UTF8
            }
            else {
                $TrainArguments = @(
                    "-u", $MultiTrainScript, "--data-root", $DataRoot, "--output-root", $OutputRoot,
                    "--max-epochs", $MaxEpochs.ToString(), "--patience", $Patience.ToString(),
                    "--batch-size", $BatchSize.ToString(), "--num-workers", $NumWorkers.ToString(),
                    "--seed", $Seed.ToString(), "--accelerator", $Device
                )
                if ($ForceTrain) { $TrainArguments += "--force" }
                if ($ShuffleSplit) { $TrainArguments += "--shuffle-split" }
                if ($NoProgressBar) { $TrainArguments += "--no-progress-bar" }
                Invoke-PythonStep "train multibehavioral" $TrainArguments
            }
            $EvaluationArguments = @(
                "-u", $MultiTraditionalScript, "--checkpoint", $Checkpoint,
                "--output", $TraditionalOutput, "--confusion-figure", $ConfusionFigure,
                "--predictions-output", $Predictions, "--data-root", $DataRoot,
                "--batch-size", $BatchSize.ToString(), "--num-workers", $NumWorkers.ToString(),
                "--device", $Device, "--seed", $Seed.ToString()
            )
            if ($ShuffleSplit) { $EvaluationArguments += "--shuffle-split" }
            Invoke-PythonStep "traditional metrics multibehavioral" $EvaluationArguments
            $SegmentArguments = @(
                "-u", $MultiSegmentScript, "--predictions", $Predictions,
                "--output", $SegmentOutput, "--data-root", $DataRoot, "--seed", $Seed.ToString()
            )
            if ($ShuffleSplit) { $SegmentArguments += "--shuffle-split" }
            Invoke-PythonStep "segment metrics multibehavioral" $SegmentArguments
            $CompletedModels = 1
            Send-BenchmarkNotification "success" "model" "multibehavioral" $BehaviorOutput 1 1
        }
        else {
          foreach ($Behavior in $Behaviors) {
            $CurrentModelName = $Behavior
            $Slug = ConvertTo-BehaviorSlug $Behavior
            $BehaviorOutput = Join-Path $OutputRoot $Slug
            $CurrentModelOutput = $BehaviorOutput
            $Checkpoint = Join-Path $BehaviorOutput "model.ckpt"
            $Predictions = Join-Path $BehaviorOutput "test_predictions.npz"
            $TraditionalOutput = Join-Path $BehaviorOutput "traditional_metrics.json"
            $SegmentOutput = Join-Path $BehaviorOutput "segment_metrics.json"

            if ((Test-Path -LiteralPath $Checkpoint -PathType Leaf) -and -not $ForceTrain) {
                "[$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))] SKIP  train $Behavior (checkpoint exists)" |
                    Add-Content -LiteralPath $StatusPath -Encoding UTF8
            }
            else {
                $TrainArguments = @(
                    "-u", $TrainScript, "--behavior", $Behavior,
                    "--data-root", $DataRoot, "--output-root", $OutputRoot,
                    "--max-epochs", $MaxEpochs.ToString(), "--patience", $Patience.ToString(),
                    "--batch-size", $BatchSize.ToString(), "--num-workers", $NumWorkers.ToString(),
                    "--seed", $Seed.ToString(), "--accelerator", $Device
                )
                if ($ForceTrain) { $TrainArguments += "--force" }
                if ($ShuffleSplit) { $TrainArguments += "--shuffle-split" }
                if ($NoProgressBar) { $TrainArguments += "--no-progress-bar" }
                Invoke-PythonStep "train $Behavior" $TrainArguments
            }

            $EvaluationArguments = @(
                "-u", $TraditionalScript, "--behavior", $Behavior,
                "--checkpoint", $Checkpoint, "--output", $TraditionalOutput,
                "--predictions-output", $Predictions, "--data-root", $DataRoot,
                "--batch-size", $BatchSize.ToString(), "--num-workers", $NumWorkers.ToString(),
                "--device", $Device, "--seed", $Seed.ToString()
            )
            if ($ShuffleSplit) { $EvaluationArguments += "--shuffle-split" }
            Invoke-PythonStep "traditional metrics $Behavior" $EvaluationArguments

            $SegmentArguments = @(
                "-u", $SegmentScript, "--behavior", $Behavior,
                "--predictions", $Predictions, "--output", $SegmentOutput,
                "--data-root", $DataRoot, "--seed", $Seed.ToString()
            )
            if ($ShuffleSplit) { $SegmentArguments += "--shuffle-split" }
            Invoke-PythonStep "segment metrics $Behavior" $SegmentArguments
            $CompletedModels += 1
            Send-BenchmarkNotification "success" "model" $Behavior $BehaviorOutput $CompletedModels $TotalModels
          }
        }
    }
    catch {
        $_ | Out-String | Add-Content -LiteralPath $StatusPath -Encoding UTF8
        $PipelineExitCode = 1
    }
    if ($PipelineExitCode -ne 0) {
        Send-BenchmarkNotification "failure" "model" $CurrentModelName $CurrentModelOutput $CompletedModels $TotalModels
    }
    elseif (-not $Multibehavioral -and $TotalModels -gt 1) {
        Send-BenchmarkNotification "success" "pipeline" "binary behavior queue" $OutputRoot $CompletedModels $TotalModels
    }
    "[$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))] Worker finished with exit code $PipelineExitCode." |
        Add-Content -LiteralPath $StatusPath -Encoding UTF8
    $PipelineExitCode | Set-Content -LiteralPath $ExitCodePath
    exit $PipelineExitCode
}

$Timestamp = [DateTime]::Now.ToString("yyyyMMdd_HHmmss")
$RunSuffix = if ($Multibehavioral) { "multibehavioral" } else { "binary_behaviors" }
$NewRunDirectory = Join-Path (Join-Path $PSScriptRoot "benchmark_runs") "${Timestamp}_${RunSuffix}"
New-Item -ItemType Directory -Path $NewRunDirectory -Force | Out-Null

$PowerShellHost = (Get-Process -Id $PID).Path
$ChildArguments = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath,
    "-Worker", "-RunDirectory", $NewRunDirectory,
    "-MaxEpochs", $MaxEpochs.ToString(), "-Patience", $Patience.ToString(),
    "-BatchSize", $BatchSize.ToString(), "-NumWorkers", $NumWorkers.ToString(),
    "-Seed", $Seed.ToString(), "-Device", $Device,
    "-DataRoot", $DataRoot, "-OutputRoot", $OutputRoot,
    "-PythonExecutable", $PythonExecutable, "-NotifyEmail", $NotifyEmail
)
if ($Multibehavioral) {
    $ChildArguments += "-Multibehavioral"
}
else {
    $ChildArguments += @("-Behaviors", ($Behaviors -join ','))
}
if ($ForceTrain) { $ChildArguments += "-ForceTrain" }
if ($ShuffleSplit) { $ChildArguments += "-ShuffleSplit" }
if ($NoProgressBar) { $ChildArguments += "-NoProgressBar" }
if ($NoNotify) { $ChildArguments += "-NoNotify" }
$QuotedChildArguments = $ChildArguments | ForEach-Object { ConvertTo-ProcessArgument $_ }

$OutputLog = Join-Path $NewRunDirectory "pipeline.out.log"
$ErrorLog = Join-Path $NewRunDirectory "pipeline.err.log"
$WorkerProcess = Start-Process `
    -FilePath $PowerShellHost `
    -ArgumentList $QuotedChildArguments `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $OutputLog `
    -RedirectStandardError $ErrorLog `
    -WindowStyle Hidden `
    -PassThru
$WorkerProcess.Id | Set-Content -LiteralPath (Join-Path $NewRunDirectory "worker.pid")
@(
    "Started:   $([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))"
    "Mode:      $(if ($Multibehavioral) { 'multibehavioral' } else { $Behaviors -join ', ' })"
    "Data:      $DataRoot"
    "Results:   $OutputRoot"
    "Force:     $ForceTrain"
    "Email:     $(if ($NoNotify) { 'disabled' } else { $NotifyEmail })"
) | Set-Content -LiteralPath (Join-Path $NewRunDirectory "run_info.txt")

Write-Host "Behavior benchmark pipeline started independently of VS Code."
Write-Host "Worker PID:    $($WorkerProcess.Id)"
Write-Host "Mode:          $(if ($Multibehavioral) { 'multibehavioral' } else { $Behaviors -join ', ' })"
Write-Host "Run directory: $NewRunDirectory"
Write-Host "Results:       $OutputRoot"
if ($NoNotify) {
    Write-Host "Email:         disabled"
}
elseif ($env:GMAIL_ADDRESS -and $env:GMAIL_APP_PASSWORD) {
    Write-Host "Email:         $NotifyEmail (credentials loaded)"
}
else {
    Write-Warning "Email credentials are missing. Run _models/setup_test_email.ps1 before the next launch."
}
Write-Host "Follow output: Get-Content -LiteralPath `"$OutputLog`" -Tail 30 -Wait"
Write-Host "Check status:  Get-Content -LiteralPath `"$(Join-Path $NewRunDirectory 'pipeline.log')`" -Tail 30"
