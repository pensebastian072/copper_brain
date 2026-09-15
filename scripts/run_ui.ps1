# Launch the Copper Brain dashboard (127.0.0.1 only) and open it as an Edge
# app-window. Starts the Flask server if it isn't already up, then opens the UI.
# Mirrors the hq-trading-system desktop launcher.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_ui.ps1 [-Port 8077]
param(
    [int]$Port = 8077
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$vpy  = Join-Path $repo ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $vpy)) { $vpy = Join-Path $repo ".venv\Scripts\python.exe" }
$url  = "http://127.0.0.1:$Port/"

function Test-Up {
    try { (Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200 }
    catch { $false }
}

if (-not (Test-Up)) {
    Write-Host "Starting dashboard server on $url ..."
    Start-Process -FilePath $vpy `
        -ArgumentList "-m","copper_brain.ui_server","--port","$Port" `
        -WorkingDirectory $repo -WindowStyle Hidden
    $n = 0
    while (-not (Test-Up) -and $n -lt 20) { Start-Sleep -Milliseconds 500; $n++ }
}

if (-not (Test-Up)) { Write-Error "server did not come up at $url"; exit 1 }

# Prefer an Edge app-window; fall back to the default browser.
$edge = "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe"
$edge86 = "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
if (Test-Path $edge) {
    Start-Process $edge "--app=$url"
} elseif (Test-Path $edge86) {
    Start-Process $edge86 "--app=$url"
} else {
    Start-Process $url
}
Write-Host "Dashboard open: $url"
