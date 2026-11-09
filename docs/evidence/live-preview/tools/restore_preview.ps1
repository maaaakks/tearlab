$ErrorActionPreference='Stop'
if(Get-Process -Name 'isaac-ng','isaac' -ErrorAction SilentlyContinue){throw 'Fermer Isaac avant restauration.'}
$installation=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'installation.json') -Raw | ConvertFrom-Json
foreach($link in $installation.links){
 $parent=Split-Path $link.path
 $entry=Get-ChildItem -LiteralPath $parent -Force | Where-Object {$_.Name -eq 'IsaacRL'}
 if(-not $entry -or $entry.LinkType -ne 'Junction' -or @($entry.Target)[0] -ne $installation.profile){throw 'Lien modifie depuis installation; restauration arretee.'}
 [IO.Directory]::Delete($link.path)
 if($link.old_target){New-Item -ItemType Junction -Path $link.path -Target $link.old_target | Out-Null}
}
Copy-Item -LiteralPath (Join-Path $installation.backup 'documents-saves\options.ini') -Destination $installation.options -Force
Write-Output 'Mod et options precedents restaures. Les sauvegardes de jeu ne sont pas ecrasees; leurs copies restent dans le backup.'
