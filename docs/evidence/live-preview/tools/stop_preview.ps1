$root=$PSScriptRoot
$currentPath=Join-Path $root 'current-session.json'
if(Test-Path -LiteralPath $currentPath){
 $current=Get-Content -LiteralPath $currentPath -Raw | ConvertFrom-Json
 New-Item -ItemType File -Path (Join-Path $current.output 'STOP') -Force | Out-Null
 Write-Output 'Arret demande. Dans le jeu, C rend aussi immediatement le controle.'
}
