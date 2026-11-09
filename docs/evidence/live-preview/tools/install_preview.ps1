$ErrorActionPreference='Stop'
if(Get-Process -Name 'isaac-ng','isaac' -ErrorAction SilentlyContinue){throw 'Fermer Isaac avant installation.'}
$root=$PSScriptRoot
$game='D:\steam\steamapps\common\The Binding of Isaac Rebirth'
$docs='C:\Users\maxbr\Documents\My Games\Binding of Isaac Repentance'
$profile=Join-Path $root 'installed-mod'
$stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
$backup=Join-Path $root ('backups\'+$stamp)
New-Item -ItemType Directory -Path $backup -Force | Out-Null
$docsBackup=Join-Path $backup 'documents-saves'
New-Item -ItemType Directory -Path $docsBackup | Out-Null
Get-ChildItem -LiteralPath $docs -File | ForEach-Object {Copy-Item -LiteralPath $_.FullName -Destination $docsBackup}
$steamCopies=@()
Get-ChildItem -LiteralPath 'D:\steam\userdata' -Directory | ForEach-Object {
 $source=Join-Path $_.FullName '250900'
 if(Test-Path -LiteralPath $source){
  $target=Join-Path $backup ('steam-saves\'+$_.Name+'\250900')
  New-Item -ItemType Directory -Path (Split-Path $target) -Force | Out-Null
  Copy-Item -LiteralPath $source -Destination $target -Recurse
  $steamCopies+=@{source=$source;backup=$target}
 }
}
$oldTarget='C:\Users\maxbr\Desktop\IsaacRL\binding-of-ai\mod'
$links=@()
foreach($parent in @((Join-Path $game 'mods'),(Join-Path $docs 'mods'))){
 $link=Join-Path $parent 'IsaacRL'
 $entry=Get-ChildItem -LiteralPath $parent -Force | Where-Object {$_.Name -eq 'IsaacRL'}
 if($entry){
  if($entry.LinkType -ne 'Junction'){throw ('Refus de remplacer un dossier non-junction: '+$link)}
  $target=@($entry.Target)[0]
  if($target -ne $oldTarget){throw ('Cible inattendue, aucune suppression: '+$target)}
  $links+=@{path=$link;old_target=$target;new_target=$profile}
 } else {$links+=@{path=$link;old_target=$null;new_target=$profile}}
}
$manifest=@{backup=$backup;profile=$profile;links=$links;steam_saves=$steamCopies;options=(Join-Path $docs 'options.ini')}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $backup 'manifest.json') -Encoding UTF8
foreach($entry in $links){
 # Delete the verified junction entry only. No recursion and no target traversal.
 if($entry.old_target){[IO.Directory]::Delete($entry.path)}
 New-Item -ItemType Junction -Path $entry.path -Target $profile | Out-Null
}
if(Get-Process -Name 'isaac-ng','isaac' -ErrorAction SilentlyContinue){throw 'Isaac ouvert entretemps : options non modifiees.'}
$options=Join-Path $docs 'options.ini'
$text=[IO.File]::ReadAllText($options)
if($text -notmatch '(?m)^EnableMods=1\r?$'){throw 'EnableMods=1 attendu; configuration inattendue.'}
if($text -notmatch '(?m)^EnableDebugConsole=[01]\r?$'){throw 'Option console absente.'}
$text=$text.Replace('EnableDebugConsole=0','EnableDebugConsole=1')
[IO.File]::WriteAllText($options,$text,(New-Object Text.UTF8Encoding($false)))
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $root 'installation.json') -Encoding UTF8
$desktop=Join-Path $env:USERPROFILE 'Desktop\IsaacRL - Jouer agent.cmd'
$command='@echo off'+"`r`n"+'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "'+(Join-Path $root 'launch_preview.ps1')+'"'+"`r`n"
[IO.File]::WriteAllText($desktop,$command,[Text.Encoding]::ASCII)
Write-Output ('Installed preview profile. Backup: '+$backup)
Write-Output ('Launcher: '+$desktop)
