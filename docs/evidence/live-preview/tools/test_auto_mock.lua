-- Mocked game and sockets. No connection to the running game.
local callbacks,sent,incoming,entities={},{},{},{}
local frame,alive,pressC=0,false,false
local mt={};function Vector(x,y)return setmetatable({X=x,Y=y},mt)end
mt.__add=function(a,b)return Vector(a.X+b.X,a.Y+b.Y)end
local p={Position=Vector(320,280),Velocity=Vector(0,0),FireDelay=0,Damage=3.5}
function p:IsDead()return not alive end
function p:ToPlayer()return self end
for _,n in ipairs({'GetHearts','GetSoulHearts','GetPlayerType','GetCollectibleCount','GetActiveItem'})do p[n]=function()return 0 end end
local room={GetTopLeftPos=function()return Vector(60,140)end,GetBottomRightPos=function()return Vector(580,420)end,GetCenterPos=function()return Vector(320,280)end,GetRoomShape=function()return 1 end,IsClear=function()return false end,SetClear=function()end,GetDoor=function()end}
function Game()return {GetFrameCount=function()return frame end,GetRoom=function()return room end,IsPaused=function()return false end}end
local mod={};function mod:AddCallback(key,fn)callbacks[key]=fn end
function RegisterMod()return mod end
function Font()return {Load=function()end,IsLoaded=function()return false end}end
Keyboard={KEY_C=67};Input={IsButtonTriggered=function()local v=pressC;pressC=false;return v end}
ModCallbacks={MC_POST_UPDATE=1,MC_POST_RENDER=2,MC_POST_NEW_ROOM=3,MC_POST_GAME_STARTED=4,MC_ENTITY_TAKE_DMG=5,MC_INPUT_ACTION=6,MC_EXECUTE_CMD=7,MC_POST_PICKUP_INIT=8}
Isaac={GetPlayer=function()return alive and p or nil end,ConsoleOutput=function()end}
function Isaac.GetRoomEntities()local out={};for _,e in ipairs(entities)do if not e.removed then out[#out+1]=e end end;return out end
function Isaac.Spawn(kind,variant,sub,pos,vel)
 local e={Type=kind,Variant=variant,Position=pos,Velocity=vel,InitSeed=#entities+1,HitPoints=10,MaxHitPoints=10,FrameCount=0}
 function e:IsActiveEnemy()return self.Type==10 or self.Type==12 end;function e:IsDead()return false end
 function e:IsVulnerableEnemy()return true end;function e:Remove()self.removed=true end
 entities[#entities+1]=e;if kind==5 and callbacks[8] then callbacks[8](mod,e) end;return e
end
ButtonAction={ACTION_LEFT=0,ACTION_RIGHT=1,ACTION_UP=2,ACTION_DOWN=3,ACTION_SHOOTLEFT=4,ACTION_SHOOTRIGHT=5,ACTION_SHOOTUP=6,ACTION_SHOOTDOWN=7,ACTION_BOMB=8,ACTION_ITEM=9,ACTION_PILLCARD=10}
InputHook={GET_ACTION_VALUE=0,IS_ACTION_PRESSED=1,IS_ACTION_TRIGGERED=2}
local inj=dofile('installed-mod/action_injector.lua')
package.preload.config=function()return {LIVE_PREVIEW=true}end
package.preload.action_injector=function()return inj end
package.preload.json=function()return {encode=function(v)sent[#sent+1]=v;return '{}'end,decode=function()return table.remove(incoming,1)end}end
local client={settimeout=function()end,send=function(_,s)return #s end,close=function()end}
function client:receive()if #incoming>0 then return 'message' end;return nil,'timeout' end
package.preload.socket=function()local accepted=false;return {tcp=function()return {setoption=function()end,bind=function()return true end,listen=function()end,settimeout=function()end,accept=function()if not accepted then accepted=true;return client end end,close=function()end}end}end
assert(dofile('live_preview.lua').start());callbacks[2](mod)
incoming[1]={command='auto_start'};callbacks[2](mod);assert(#Isaac.GetRoomEntities()==0)
alive=true;callbacks[3](mod);callbacks[4](mod)
local oldLoot=Isaac.Spawn(5,100,0,Vector(1,1),Vector(0,0))
assert(not oldLoot.removed,'manual pickup should be preserved')
local function step(to)while frame<to do frame=frame+1;callbacks[1](mod)end end
step(6);assert(#Isaac.GetRoomEntities()==1 and Isaac.GetRoomEntities()[1].Type==12,'pending auto-start lost');assert(inj.isEnabled())
assert(oldLoot.removed,'existing loot not cleared at round reset')
for _,variant in ipairs({10,20,30,40,50,100,300,350}) do
 local loot=Isaac.Spawn(5,variant,0,Vector(1,1),Vector(0,0))
 assert(loot.removed,'new demo loot not suppressed: '..variant)
end
local state=sent[#sent]
incoming[1]={action={action_id=1,room_id=state.room_id,control_id=state.control_id,move=7,shoot=1}}
callbacks[2](mod)
assert(callbacks[6](mod,p,InputHook.GET_ACTION_VALUE,ButtonAction.ACTION_LEFT)==1,'movement action ignored')
assert(callbacks[6](mod,p,InputHook.IS_ACTION_PRESSED,ButtonAction.ACTION_DOWN)==true,'diagonal movement ignored')
assert(callbacks[6](mod,p,InputHook.GET_ACTION_VALUE,ButtonAction.ACTION_SHOOTUP)==1,'shooting action ignored')
assert(callbacks[6](mod,p,InputHook.GET_ACTION_VALUE,ButtonAction.ACTION_RIGHT)==0,'opposite movement must be released')
assert(callbacks[5](mod,p,1,0,nil)==false,'damage must count but protect hearts')
step(53);assert(#Isaac.GetRoomEntities()==1 and Isaac.GetRoomEntities()[1].Type==10,'next scenario did not start');assert(inj.isEnabled())
pressC=true;step(54);assert(not inj.isEnabled());assert(callbacks[6](mod,p,InputHook.GET_ACTION_VALUE,ButtonAction.ACTION_LEFT)==nil,'manual input not released');local count=#entities;step(120);assert(#entities==count,'C stop restarted')
local manualLoot=Isaac.Spawn(5,100,0,Vector(1,1),Vector(0,0));assert(not manualLoot.removed,'manual pickup incorrectly suppressed')
print('PASS: real injector, auto sequence, contact/C stop, existing loot cleanup and new demo pickup suppression; manual pickups preserved')
