<#
.SYNOPSIS
Configure the benchmark Gmail credentials and send a real test email.

.EXAMPLE
& .\_models\setup_test_email.ps1

Use the Gmail address and App Password already present in this PowerShell
process or saved as Windows user environment variables, then send a test.

.EXAMPLE
& .\_models\setup_test_email.ps1 -PromptForPassword -PersistUser

Prompt for a new App Password, test it, and save it for future benchmark runs.
User environment variables are outside the repository, but Windows does not
encrypt their values at rest.
#>

[CmdletBinding()]
param(
    [string]$EmailAddress,
    [string]$Recipient,
    [string]$PythonPath,
    [switch]$PersistUser,
    [switch]$PromptForPassword
)

$ErrorActionPreference = "Stop"

if (-not $EmailAddress) {
    $EmailAddress = if ($env:GMAIL_ADDRESS) {
        $env:GMAIL_ADDRESS
    } elseif ([Environment]::GetEnvironmentVariable("GMAIL_ADDRESS", "User")) {
        [Environment]::GetEnvironmentVariable("GMAIL_ADDRESS", "User")
    } else {
        "braydenk@andrew.cmu.edu"
    }
}

if (-not $Recipient) {
    $Recipient = $EmailAddress
}

if (-not $PythonPath) {
    $candidates = @(@(
        $env:RTASOID_PYTHON,
        "C:\Users\Jimi\anaconda3\envs\realtime_train_lightning\python.exe"
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) })

    if ($candidates.Count -gt 0) {
        $PythonPath = $candidates[0]
    } else {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCommand) {
            $PythonPath = $pythonCommand.Source
        }
    }
}

if (-not $PythonPath -or -not (Test-Path -LiteralPath $PythonPath)) {
    throw "Could not find Python. Pass -PythonPath with the training environment's python.exe."
}

$passwordPointer = [IntPtr]::Zero
$testRecipientVariable = "GMAIL_TEST_RECIPIENT"

try {
    $plainPassword = if (-not $PromptForPassword -and $env:GMAIL_APP_PASSWORD) {
        $env:GMAIL_APP_PASSWORD
    } elseif (-not $PromptForPassword -and [Environment]::GetEnvironmentVariable("GMAIL_APP_PASSWORD", "User")) {
        [Environment]::GetEnvironmentVariable("GMAIL_APP_PASSWORD", "User")
    } else {
        $securePassword = Read-Host "Paste the NEW Gmail App Password" -AsSecureString
        $passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
        [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer)
    }
    $plainPassword = $plainPassword -replace "\s", ""
    if (-not $plainPassword) {
        throw "The Gmail App Password was empty."
    }

    # These are inherited by the benchmark when it is launched from this shell.
    $env:GMAIL_ADDRESS = $EmailAddress
    $env:GMAIL_APP_PASSWORD = $plainPassword
    Set-Item -Path "Env:$testRecipientVariable" -Value $Recipient

    if ($PersistUser) {
        [Environment]::SetEnvironmentVariable("GMAIL_ADDRESS", $EmailAddress, "User")
        [Environment]::SetEnvironmentVariable("GMAIL_APP_PASSWORD", $plainPassword, "User")
        Write-Host "Saved Gmail settings as Windows user environment variables."
    }

    $testScript = @'
import os
import smtplib
from email.message import EmailMessage

sender = os.environ.get("GMAIL_ADDRESS")
password = os.environ.get("GMAIL_APP_PASSWORD")
recipient = os.environ.get("GMAIL_TEST_RECIPIENT") or sender

if not sender:
    raise RuntimeError("GMAIL_ADDRESS is not set")
if not password:
    raise RuntimeError("GMAIL_APP_PASSWORD is not set")

message = EmailMessage()
message["Subject"] = "Behavior classifier email test"
message["From"] = sender
message["To"] = recipient
message.set_content(
    "The realtime behavior classifier benchmark email configuration is working."
)

with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
    server.starttls()
    server.login(sender, password)
    server.send_message(message)

print(f"Test email sent successfully from {sender} to {recipient}")
'@

    $testScript | & $PythonPath -
    if ($LASTEXITCODE -ne 0) {
        throw "The Python SMTP test failed with exit code $LASTEXITCODE."
    }

    Write-Host "Email variables are ready for benchmarks launched from this PowerShell window."
}
finally {
    Remove-Item -Path "Env:$testRecipientVariable" -ErrorAction SilentlyContinue
    if ($passwordPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer)
    }
    Remove-Variable plainPassword -ErrorAction SilentlyContinue
}
