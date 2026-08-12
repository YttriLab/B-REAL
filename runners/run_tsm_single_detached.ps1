[CmdletBinding()]
param(
    [switch]$Worker,
    [string]$RunDirectory,
    [int]$Repeats = 1,
    [int]$MaxEpochs = 0,
    [int]$Patience = 0,
    [string]$NotifyEmail = "braydenk@andrew.cmu.edu",
    [switch]$Resume,
    [switch]$KeepCheckpoints,
    [switch]$NoProgressBar,
    [switch]$NoNotify
)

$ErrorActionPreference = "Stop"

<#
Runs the controlled comparison between:
  1. Three-frame TSM-EfficientNet
  2. The matched EfficientNet using only the most recent frame

The benchmark uses the five recordings from the last split-order experiment
and forces a chronological train -> val -> test split.

The launcher starts a hidden worker process that remains independent of the
calling terminal and VS Code. Closing the terminal will not stop training.

Examples:
  powershell -ExecutionPolicy Bypass -File _models/run_tsm_single_detached.ps1
  powershell -ExecutionPolicy Bypass -File _models/run_tsm_single_detached.ps1 -Repeats 3
  powershell -ExecutionPolicy Bypass -File _models/run_tsm_single_detached.ps1 -MaxEpochs 5
  powershell -ExecutionPolicy Bypass -File _models/run_tsm_single_detached.ps1 -Resume

Benchmark results:
  benchmarking/figures/tsm_vs_single_frame/

Per-launch logs and process metadata:
  _models/benchmark_runs/<timestamp>_tsm_vs_single_frame/

Pressing Ctrl+C while following a log only stops the log viewer. It does not
stop the benchmark.
#>

$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExecutable = "C:\Users\Jimi\anaconda3\envs\realtime_train_lightning\python.exe"
$BenchmarkScript = Join-Path $RepoRoot "benchmarking/code/benchmark_tsm_vs_single_frame.py"
$ConfigPath = Join-Path $RepoRoot "benchmarking/configs/gui_global_split_order_folds.yaml"
$OutputDirectory = Join-Path $RepoRoot "benchmarking/figures/tsm_vs_single_frame"

# A launcher opened before setup_test_email.ps1 ran may not have inherited
# persisted user variables. Import them without printing their values.
foreach ($EmailVariableName in @("GMAIL_ADDRESS", "GMAIL_APP_PASSWORD")) {
    $SavedEmailValue = [Environment]::GetEnvironmentVariable($EmailVariableName, "User")
    if ($SavedEmailValue) {
        Set-Item -LiteralPath "Env:$EmailVariableName" -Value $SavedEmailValue
    }
}

function ConvertTo-ProcessArgument {
    param([AllowEmptyString()][string]$Value)

    # Start-Process joins ArgumentList into one Windows command line. Preserve
    # whitespace and embedded quotes when it does so.
    if ($Value -notmatch '[\s"]') {
        return $Value
    }
    return '"' + ($Value -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}

function Get-BenchmarkArguments {
    $Arguments = @(
        "-u",
        $BenchmarkScript,
        "--config", $ConfigPath,
        "--output-dir", $OutputDirectory,
        "--repeats", $Repeats.ToString(),
        "--notify-email", $NotifyEmail
    )
    if ($MaxEpochs -gt 0) {
        $Arguments += @("--max-epochs", $MaxEpochs.ToString())
    }
    if ($Patience -gt 0) {
        $Arguments += @("--patience", $Patience.ToString())
    }
    if ($Resume) {
        $Arguments += "--resume"
    }
    if ($KeepCheckpoints) {
        $Arguments += "--keep-checkpoints"
    }
    if ($NoProgressBar) {
        $Arguments += "--no-progress-bar"
    }
    if ($NoNotify) {
        $Arguments += "--no-notify"
    }
    return $Arguments
}

if ($Worker) {
    if ([string]::IsNullOrWhiteSpace($RunDirectory)) {
        throw "The internal worker requires -RunDirectory."
    }

    $StatusLogPath = Join-Path $RunDirectory "benchmark.log"
    $OutputLogPath = Join-Path $RunDirectory "benchmark.out.log"
    $ErrorLogPath = Join-Path $RunDirectory "benchmark.err.log"
    $ExitCodePath = Join-Path $RunDirectory "exit_code.txt"

    try {
        Set-Location -LiteralPath $RepoRoot
        "[$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))] Benchmark worker started." |
            Set-Content -LiteralPath $StatusLogPath -Encoding UTF8

        $PythonArguments = Get-BenchmarkArguments
        $QuotedPythonArguments = $PythonArguments | ForEach-Object {
            ConvertTo-ProcessArgument -Value $_
        }
        $PythonProcess = Start-Process `
            -FilePath $PythonExecutable `
            -ArgumentList $QuotedPythonArguments `
            -WorkingDirectory $RepoRoot `
            -RedirectStandardOutput $OutputLogPath `
            -RedirectStandardError $ErrorLogPath `
            -WindowStyle Hidden `
            -PassThru `
            -Wait
        $BenchmarkExitCode = $PythonProcess.ExitCode
    }
    catch {
        $_ | Out-String | Add-Content -LiteralPath $StatusLogPath -Encoding UTF8
        $BenchmarkExitCode = 1
    }

    "[$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))] Benchmark worker finished with exit code $BenchmarkExitCode." |
        Add-Content -LiteralPath $StatusLogPath -Encoding UTF8
    $BenchmarkExitCode | Set-Content -LiteralPath $ExitCodePath
    exit $BenchmarkExitCode
}

if ($Repeats -lt 1) {
    throw "-Repeats must be at least 1."
}
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw "Python executable not found: $PythonExecutable"
}
if (-not (Test-Path -LiteralPath $BenchmarkScript -PathType Leaf)) {
    throw "Benchmark script not found: $BenchmarkScript"
}
if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    throw "Benchmark config not found: $ConfigPath"
}

$Timestamp = [DateTime]::Now.ToString("yyyyMMdd_HHmmss")
$RunsRoot = Join-Path $PSScriptRoot "benchmark_runs"
$NewRunDirectory = Join-Path $RunsRoot "${Timestamp}_tsm_vs_single_frame"
New-Item -ItemType Directory -Path $NewRunDirectory -Force | Out-Null

$DisplayArguments = Get-BenchmarkArguments | ForEach-Object {
    if ($_ -match '\s') { '"{0}"' -f ($_ -replace '"', '\"') } else { $_ }
}
@(
    "Started:  $([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))"
    "Root:     $RepoRoot"
    "Python:   $PythonExecutable"
    "Command:  $PythonExecutable $($DisplayArguments -join ' ')"
) | Set-Content -LiteralPath (Join-Path $NewRunDirectory "run_info.txt")

# Start a separate hidden PowerShell worker. The worker then owns and waits for
# the Python child, so closing this terminal or VS Code does not end training.
$PowerShellHost = (Get-Process -Id $PID).Path
$ChildArguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", ('"{0}"' -f $PSCommandPath),
    "-Worker",
    "-RunDirectory", ('"{0}"' -f $NewRunDirectory),
    "-Repeats", $Repeats.ToString(),
    "-NotifyEmail", ('"{0}"' -f $NotifyEmail)
)
if ($MaxEpochs -gt 0) {
    $ChildArguments += @("-MaxEpochs", $MaxEpochs.ToString())
}
if ($Patience -gt 0) {
    $ChildArguments += @("-Patience", $Patience.ToString())
}
if ($Resume) {
    $ChildArguments += "-Resume"
}
if ($KeepCheckpoints) {
    $ChildArguments += "-KeepCheckpoints"
}
if ($NoProgressBar) {
    $ChildArguments += "-NoProgressBar"
}
if ($NoNotify) {
    $ChildArguments += "-NoNotify"
}

$WorkerProcess = Start-Process `
    -FilePath $PowerShellHost `
    -ArgumentList $ChildArguments `
    -WorkingDirectory $RepoRoot `
    -WindowStyle Hidden `
    -PassThru

$WorkerProcess.Id | Set-Content -LiteralPath (Join-Path $NewRunDirectory "worker.pid")

$OutputLog = Join-Path $NewRunDirectory "benchmark.out.log"
$ErrorLog = Join-Path $NewRunDirectory "benchmark.err.log"
$StatusLog = Join-Path $NewRunDirectory "benchmark.log"
$ExitCodeFile = Join-Path $NewRunDirectory "exit_code.txt"
$GitTail = "C:\Program Files\Git\usr\bin\tail.exe"

$LaunchDescription = if ($Resume) { "Benchmark resume" } else { "Benchmark" }
Write-Host "$LaunchDescription started independently of VS Code."
Write-Host "Worker PID:    $($WorkerProcess.Id)"
Write-Host "Run directory: $NewRunDirectory"
Write-Host "Results:       $OutputDirectory"
if ($NoNotify) {
    Write-Host "Email:         disabled"
}
elseif ($env:GMAIL_ADDRESS -and $env:GMAIL_APP_PASSWORD) {
    Write-Host "Email:         $NotifyEmail (credentials loaded)"
}
else {
    Write-Warning "Email credentials are missing; completion email will be skipped."
}
Write-Host ""

if (Test-Path -LiteralPath $GitTail -PathType Leaf) {
    Write-Host "Follow training with:"
    Write-Host ('  & "{0}" -c 0 -f "{1}"' -f $GitTail, $OutputLog)
}
else {
    Write-Host "Follow training with:"
    Write-Host ('  Get-Content -LiteralPath "{0}" -Tail 30 -Wait' -f $OutputLog)
}

Write-Host ""
Write-Host "Inspect warnings and errors:"
Write-Host ('  Get-Content -LiteralPath "{0}" -Tail 30' -f $ErrorLog)
Write-Host ""
Write-Host "Check whether the worker is still running:"
Write-Host ('  Get-Process -Id {0} -ErrorAction SilentlyContinue' -f $WorkerProcess.Id)
Write-Host ""
Write-Host "Check completion status:"
Write-Host ('  Get-Content -LiteralPath "{0}"' -f $StatusLog)
Write-Host ('  Get-Content -LiteralPath "{0}"' -f $ExitCodeFile)
Write-Host ""
Write-Host "Stop the benchmark and its Python child:"
Write-Host ('  taskkill.exe /PID {0} /T /F' -f $WorkerProcess.Id)
