param(
    [string]$OptionsPath = "$env:USERPROFILE\Documents\My Games\Binding of Isaac Repentance\options.ini"
)

$ErrorActionPreference = "Stop"

if (Get-Process -Name "isaac-ng" -ErrorAction SilentlyContinue) {
    throw "Isaac is running. Close the game before changing options.ini, then run this script again."
}

if (-not (Test-Path -LiteralPath $OptionsPath)) {
    throw "Isaac options file not found: $OptionsPath"
}

$resolvedOptions = (Resolve-Path -LiteralPath $OptionsPath).Path
$content = Get-Content -Raw -LiteralPath $resolvedOptions

if ($content -match '(?m)^PauseOnFocusLost=') {
    $updated = $content -replace '(?m)^PauseOnFocusLost=.*$', 'PauseOnFocusLost=0'
} else {
    $updated = $content -replace '(?m)^\[Options\]\s*$', "[Options]`r`nPauseOnFocusLost=0"
}

if ($updated -eq $content) {
    Write-Host "Background training is already enabled (PauseOnFocusLost=0)."
    exit 0
}

Set-Content -LiteralPath $resolvedOptions -Value $updated -NoNewline
Write-Host "Background training enabled: PauseOnFocusLost=0"
Write-Host "Options file: $resolvedOptions"
