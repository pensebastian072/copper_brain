# Register Copper Brain's weekly self-update as a Windows Scheduled Task.
# Trigger: every Sunday 09:00 (local). Runs scripts\run_update.ps1 which pulls
# fresh data, rebuilds the v1 score, reruns the v2 overfit gate, and republishes
# data/regime/copper_regime.json. LIMITED privilege (no admin / UAC).
# Re-run anytime to re-register; the existing task is replaced.
#
# Mirrors hq-trading-system/analytics/register_autostart.ps1.

$ErrorActionPreference = "Stop"

$taskName = "CopperBrainWeeklyUpdate"
$repo     = Split-Path -Parent $PSScriptRoot
$runner   = Join-Path $repo "scripts\run_update.ps1"

if (-not (Test-Path $runner)) {
    Write-Error "missing $runner"
    exit 1
}

# Remove any prior registration so re-runs are idempotent.
try {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction Stop
    Write-Host "Removed existing task '$taskName'."
} catch {
    # Task did not exist - fine.
}

# Action: powershell.exe runs the runner script (with retrain).
$psExe   = (Get-Command powershell.exe).Source
$action  = New-ScheduledTaskAction -Execute $psExe `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$runner`"" `
    -WorkingDirectory $repo

# Trigger: weekly, Sunday 09:00.
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "09:00"

# Settings: tolerate battery, catch up if the machine was off, retry on failure,
# cap runtime at 1h (a stuck pull shouldn't pin the box).
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 10) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

# Principal: current user, LIMITED privilege (no admin elevation = no UAC).
$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

try {
    Register-ScheduledTask `
        -TaskName $taskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description "Copper Brain weekly self-update: pull free data, rebuild v1 score, rerun v2 overfit gate, republish copper_regime.json. Advisory only." | Out-Null

    Write-Host "Registered scheduled task '$taskName' (LIMITED privilege)."
    Write-Host "Trigger: every Sunday 09:00 for $env:USERNAME."
    Write-Host ""

    $t = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
    Write-Host "Verification:"
    Write-Host "  Name:     $($t.TaskName)"
    Write-Host "  State:    $($t.State)"
    Write-Host ""
    Write-Host "Test now without waiting for Sunday:"
    Write-Host "  Start-ScheduledTask -TaskName '$taskName'"
    Write-Host ""
    Write-Host "Remove later:"
    Write-Host "  Unregister-ScheduledTask -TaskName '$taskName' -Confirm:`$false"
} catch {
    Write-Error "Register-ScheduledTask FAILED: $($_.Exception.Message)"
    Write-Host ""
    Write-Host "Fallback via schtasks.exe (user-level):"
    Write-Host "  schtasks /Create /SC WEEKLY /D SUN /ST 09:00 /TN $taskName ``"
    Write-Host "    /TR `"powershell -NoProfile -ExecutionPolicy Bypass -File '$runner'`" /RL LIMITED /F"
    exit 1
}
