param([string[]]$Files,[switch]$RunTests)
$ErrorActionPreference='Stop'
if(-not $Files){$Files=Get-ChildItem (Join-Path $PSScriptRoot 'installed-mod') -Filter '*.lua' | Select-Object -ExpandProperty FullName}
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class PreviewLua {
 [DllImport(@"D:\steam\steamapps\common\The Binding of Isaac Rebirth\Lua5.3.3f.dll", CallingConvention=CallingConvention.Cdecl)] public static extern IntPtr luaL_newstate();
 [DllImport(@"D:\steam\steamapps\common\The Binding of Isaac Rebirth\Lua5.3.3f.dll", CallingConvention=CallingConvention.Cdecl)] public static extern int luaL_loadfilex(IntPtr s,string file,string mode);
 [DllImport(@"D:\steam\steamapps\common\The Binding of Isaac Rebirth\Lua5.3.3f.dll", CallingConvention=CallingConvention.Cdecl)] public static extern IntPtr lua_tolstring(IntPtr s,int index,IntPtr len);
 [DllImport(@"D:\steam\steamapps\common\The Binding of Isaac Rebirth\Lua5.3.3f.dll", CallingConvention=CallingConvention.Cdecl)] public static extern void lua_close(IntPtr s);
 [DllImport(@"D:\steam\steamapps\common\The Binding of Isaac Rebirth\Lua5.3.3f.dll", CallingConvention=CallingConvention.Cdecl)] public static extern void luaL_openlibs(IntPtr s);
 [DllImport(@"D:\steam\steamapps\common\The Binding of Isaac Rebirth\Lua5.3.3f.dll", CallingConvention=CallingConvention.Cdecl)] public static extern int lua_pcallk(IntPtr s,int nargs,int nresults,int err,IntPtr ctx,IntPtr fn);
}
'@
foreach($file in $Files){
 $state=[PreviewLua]::luaL_newstate()
 try {
  $code=[PreviewLua]::luaL_loadfilex($state,$file,'t')
  if($code -ne 0){throw [Runtime.InteropServices.Marshal]::PtrToStringAnsi([PreviewLua]::lua_tolstring($state,-1,[IntPtr]::Zero))}
  Write-Output "Lua syntax OK: $file"
 } finally {[PreviewLua]::lua_close($state)}
}
if($RunTests){
 $state=[PreviewLua]::luaL_newstate();$previousDirectory=[Environment]::CurrentDirectory
 try {
  [Environment]::CurrentDirectory=$PSScriptRoot
  [PreviewLua]::luaL_openlibs($state)
  $code=[PreviewLua]::luaL_loadfilex($state,(Join-Path $PSScriptRoot 'test_auto_mock.lua'),'t')
  if($code -eq 0){$code=[PreviewLua]::lua_pcallk($state,0,-1,0,[IntPtr]::Zero,[IntPtr]::Zero)}
  if($code -ne 0){throw [Runtime.InteropServices.Marshal]::PtrToStringAnsi([PreviewLua]::lua_tolstring($state,-1,[IntPtr]::Zero))}
 } finally {[PreviewLua]::lua_close($state);[Environment]::CurrentDirectory=$previousDirectory}
}
