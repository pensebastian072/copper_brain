# Copper Brain - self-update runner (called by the weekly Scheduled Task, or by
# hand). Resolves the repo + venv from this script's own location, runs the full
# pipeline WITH retrain (ingest -> features -> score -> v2 gate -> publish), and
# tees all output to a timestamped log under journal/runs/.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\run_update.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_update.ps1 -NoRetrain
param(
    [switch]$NoRetrain
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$vpy  = Join-Path $repo ".venv\Scripts\python.exe"
$runs = Join-Path $repo "journal\runs"

if (-not (Test-Path $vpy)) {
    Write-Error "venv python not found at $vpy - create it with: python -m venv .venv; .venv\Scripts\pip install -r requirements.txt"
    exit 1
}
if (-not (Test-Path $runs)) { New-Item -ItemType Directory -Force $runs | Out-Null }

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log   = Join-Path $runs "update_$stamp.log"

$pyArgs = @("-m", "copper_brain.update")
if (-not $NoRetrain) { $pyArgs += "--retrain" }

Write-Host "Copper Brain update -> $log"
& $vpy @pyArgs *>&1 | Tee-Object -FilePath $log
$code = $LASTEXITCODE

# Prune logs older than 90 days so journal/runs doesn't grow unbounded.
Get-ChildItem $runs -Filter "update_*.log" -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-90) } |
    Remove-Item -Force -ErrorAction SilentlyContinue

exit $code
