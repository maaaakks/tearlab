$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$python='C:\Users\maxbr\Desktop\game\IsaacRL\binding-of-ai\.venv\Scripts\python.exe'
$checkpoint=Join-Path (Split-Path $root) 'isaac-crossfire\focus-seed129773\pool-best.pt'
$currentPath=Join-Path $root 'current-session.json'
$reuse=$false
if(Test-Path -LiteralPath $currentPath){
 $current=Get-Content -LiteralPath $currentPath -Raw | ConvertFrom-Json
 $worker=Get-CimInstance Win32_Process -Filter ('ProcessId='+$current.agent_pid) -ErrorAction SilentlyContinue
 if($worker -and $worker.CommandLine -like '*live_agent.py*' -and $worker.CommandLine -like ('*'+$current.output+'*')){$reuse=$true}
}
if(-not $reuse){
 $output=Join-Path $root ('sessions\'+(Get-Date -Format 'yyyyMMdd-HHmmss'))
 New-Item -ItemType Directory -Path $output -Force | Out-Null
 $agentArguments=@('-B',('"'+(Join-Path $root 'live_agent.py')+'"'),'--checkpoint',('"'+$checkpoint+'"'),'--output',('"'+$output+'"'))
 $env:PYTHONDONTWRITEBYTECODE='1'
 $worker=Start-Process -FilePath $python -ArgumentList $agentArguments -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $output 'stdout.log') -RedirectStandardError (Join-Path $output 'stderr.log')
 @{agent_pid=$worker.Id;output=$output;checkpoint=$checkpoint} | ConvertTo-Json | Set-Content -LiteralPath $currentPath -Encoding UTF8
}
if(-not(Get-Process -Name 'isaac-ng','isaac' -ErrorAction SilentlyContinue)){
 Start-Process -FilePath 'D:\steam\steam.exe' -ArgumentList @('-applaunch','250900','-luadebug')
}
Write-Output 'IsaacRL pret : lancer une nouvelle partie avec Isaac. C active / coupe le controle.'
Write-Output 'Console : rlpreview horf 1 (ou 2/4), rlpreview gaper 1. Commande volontaire seulement.'
Write-Output ('Rapport : '+$currentPath)
