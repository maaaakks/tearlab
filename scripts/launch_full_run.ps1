# Launch the generalist full-run training (TODO jalon B / E1).
#
# One command to start the first real natural multi-room run. Requires:
#   - Isaac open with the IsaacRL mod enabled (see README "Premier run généraliste")
#   - options.ini configured while the game is closed
#
# Usage (from the repo root):
#   .\scripts\launch_full_run.ps1                  # fresh run
#   .\scripts\launch_full_run.ps1 -Resume latest   # continue last checkpoint

param(
    [string]$Resume = ""
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
$pythonDir = Join-Path $repoRoot "python"
$config = Join-Path $repoRoot "configs\full_game.yaml"

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pythonExe = $venvPython
} else {
    $pythonExe = "python"
}

$pyArgs = @("train.py", "--config", $config)
if ($Resume) {
    $pyArgs += @("--resume", $Resume)
}

Write-Host "Launching full-run generalist training:" -ForegroundColor Cyan
Write-Host "  config : $config"
Write-Host "  resume : $(if ($Resume) { $Resume } else { '(nouveau modèle)' })"

Push-Location $pythonDir
try {
    & $pythonExe @pyArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}
