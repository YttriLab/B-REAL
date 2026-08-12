[CmdletBinding()]
param(
    [switch]$Worker,
    [string]$QueueDirectory
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExecutable = "C:\Users\Jimi\anaconda3\envs\realtime_train_lightning\python.exe"
$CurrentRunDirectory = Join-Path $PSScriptRoot "benchmark_runs\20260719_223432_size_learning_curve_concat_10ep"
$BenchmarkScript = Join-Path $RepoRoot "benchmarking\code\benchmark_size_learning_curve.py"
$EvaluationScript = Join-Path $RepoRoot "benchmarking\code\evaluate_checkpoint.py"
$SchedulerConfig = Join-Path $RepoRoot "benchmarking\configs\concat_chronological_tmax30.yaml"
$GuiParityConfig = Join-Path $RepoRoot "benchmarking\configs\gui_session_parity.yaml"
$GuiCheckpoint = "D:\Realtime\realtime_app_output\models\checkpoints\face-head-paw-groom-epoch=05-val_mcc=0.839.ckpt"

function ConvertTo-ProcessArgument {
    param([AllowEmptyString()][string]$Value)
    if ($Value -notmatch '[\s"]') {
        return $Value
    }
    return '"' + ($Value -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}

function Add-QueueLog {
    param([string]$Message)
    "[$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))] $Message" |
        Add-Content -LiteralPath (Join-Path $QueueDirectory "queue.log") -Encoding UTF8
}

function Invoke-QueuedPythonJob {
    param(
        [string]$Name,
        [string]$Script,
        [string[]]$Arguments
    )

    $JobDirectory = Join-Path $QueueDirectory $Name
    New-Item -ItemType Directory -Path $JobDirectory -Force | Out-Null
    $StdoutPath = Join-Path $JobDirectory "stdout.log"
    $StderrPath = Join-Path $JobDirectory "stderr.log"
    $ExitCodePath = Join-Path $JobDirectory "exit_code.txt"
    $CommandArguments = @("-u", $Script) + $Arguments
    $QuotedArguments = $CommandArguments | ForEach-Object {
        ConvertTo-ProcessArgument -Value $_
    }

    Add-QueueLog "Starting $Name"
    $Process = Start-Process `
        -FilePath $PythonExecutable `
        -ArgumentList $QuotedArguments `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath `
        -WindowStyle Hidden `
        -PassThru `
        -Wait
    $Process.ExitCode | Set-Content -LiteralPath $ExitCodePath
    Add-QueueLog "Finished $Name with exit code $($Process.ExitCode)"
    if ($Process.ExitCode -ne 0) {
        throw "$Name failed with exit code $($Process.ExitCode). See $StderrPath"
    }
}

if ($Worker) {
    if ([string]::IsNullOrWhiteSpace($QueueDirectory)) {
        throw "The queue worker requires -QueueDirectory."
    }
    Set-Location -LiteralPath $RepoRoot
    "[$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))] Follow-up queue worker started." |
        Set-Content -LiteralPath (Join-Path $QueueDirectory "queue.log") -Encoding UTF8

    try {
        $CurrentWorkerPidPath = Join-Path $CurrentRunDirectory "worker.pid"
        $CurrentExitCodePath = Join-Path $CurrentRunDirectory "exit_code.txt"
        if (-not (Test-Path -LiteralPath $CurrentWorkerPidPath -PathType Leaf)) {
            throw "Current experiment worker PID not found: $CurrentWorkerPidPath"
        }
        $CurrentWorkerPid = [int](Get-Content -LiteralPath $CurrentWorkerPidPath)
        Add-QueueLog "Waiting for current experiment worker PID $CurrentWorkerPid"
        Wait-Process -Id $CurrentWorkerPid -ErrorAction SilentlyContinue

        if (-not (Test-Path -LiteralPath $CurrentExitCodePath -PathType Leaf)) {
            throw "Current experiment exited without writing $CurrentExitCodePath"
        }
        $CurrentExitCode = [int](Get-Content -LiteralPath $CurrentExitCodePath)
        if ($CurrentExitCode -ne 0) {
            throw "Current experiment failed with exit code $CurrentExitCode; follow-ups were not started."
        }
        Add-QueueLog "Current experiment completed successfully"

        Invoke-QueuedPythonJob `
            -Name "01_scheduler_pilot_frac80_tmax30" `
            -Script $BenchmarkScript `
            -Arguments @(
                "--config", $SchedulerConfig,
                "--fractions", "80",
                "--max-epochs", "10",
                "--repeats", "1",
                "--output-dir", (Join-Path $RepoRoot "benchmarking\figures\scheduler_pilot_frac80_tmax30")
            )

        Invoke-QueuedPythonJob `
            -Name "02_gui_checkpoint_heldout_test" `
            -Script $EvaluationScript `
            -Arguments @(
                "--config", $GuiParityConfig,
                "--checkpoint", $GuiCheckpoint,
                "--output", (Join-Path $RepoRoot "benchmarking\figures\gui_checkpoint_heldout_test\results.json")
            )

        Invoke-QueuedPythonJob `
            -Name "03_gui_session_parity_100" `
            -Script $BenchmarkScript `
            -Arguments @(
                "--config", $GuiParityConfig,
                "--fractions", "100",
                "--max-epochs", "100",
                "--patience", "3",
                "--repeats", "1",
                "--output-dir", (Join-Path $RepoRoot "benchmarking\figures\gui_session_parity_100")
            )

        Add-QueueLog "All follow-up jobs completed successfully"
        0 | Set-Content -LiteralPath (Join-Path $QueueDirectory "exit_code.txt")
        exit 0
    } catch {
        Add-QueueLog -Message ($_ | Out-String)
        1 | Set-Content -LiteralPath (Join-Path $QueueDirectory "exit_code.txt")
        exit 1
    }
}

foreach ($RequiredPath in @(
    $PythonExecutable,
    $CurrentRunDirectory,
    $BenchmarkScript,
    $EvaluationScript,
    $SchedulerConfig,
    $GuiParityConfig,
    $GuiCheckpoint
)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required queue path not found: $RequiredPath"
    }
}

$Timestamp = [DateTime]::Now.ToString("yyyyMMdd_HHmmss")
$QueueDirectory = Join-Path $PSScriptRoot "benchmark_runs\${Timestamp}_followup_queue"
New-Item -ItemType Directory -Path $QueueDirectory -Force | Out-Null

$PowerShellHost = (Get-Process -Id $PID).Path
$ChildArguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", ('"{0}"' -f $PSCommandPath),
    "-Worker",
    "-QueueDirectory", ('"{0}"' -f $QueueDirectory)
)
$Process = Start-Process `
    -FilePath $PowerShellHost `
    -ArgumentList $ChildArguments `
    -WorkingDirectory $RepoRoot `
    -WindowStyle Hidden `
    -PassThru
$Process.Id | Set-Content -LiteralPath (Join-Path $QueueDirectory "worker.pid")

Write-Host "Follow-up queue armed."
Write-Host "Queue PID: $($Process.Id)"
Write-Host "Queue directory: $QueueDirectory"
Write-Host "The queue will wait for the current experiment, then run three GPU jobs sequentially."
Write-Host "Monitor with:"
Write-Host ('  Get-Content -LiteralPath "{0}" -Tail 30 -Wait' -f (Join-Path $QueueDirectory "queue.log"))
