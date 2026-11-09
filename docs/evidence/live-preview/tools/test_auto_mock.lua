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
ModCallbacks={MC_POST_UPDATE=1,MC_POST_RENDER=2,MC_POST_NEW_ROOM=3,MC_POST_GAME_STARTED=4,MC_ENTITY_TAKE_DMG=5,MC_INPUT_ACTION=6,MC_EXECUTE_CMD=7}
Isaac={GetPlayer=function()return alive and p or nil end,ConsoleOutput=function()end}
function Isaac.GetRoomEntities()local out={};for _,e in ipairs(entities)do if not e.removed then out[#out+1]=e end end;return out end
function Isaac.Spawn(kind,variant,sub,pos,vel)
 local e={Type=kind,Variant=variant,Position=pos,Velocity=vel,InitSeed=#entities+1,HitPoints=10,MaxHitPoints=10,FrameCount=0}
 function e:IsActiveEnemy()return true end;function e:IsDead()return false end
 function e:IsVulnerableEnemy()return true end;function e:Remove()self.removed=true end
 entities[#entities+1]=e;return e
end
local inj={enabled=false,setAction=function()end,tick=function()end,reset=function()end,onInputAction=function()end}
function inj.setEnabled(v)inj.enabled=v end
package.preload.config=function()return {LIVE_PREVIEW=true}end
package.preload.action_injector=function()return inj end
package.preload.json=function()return {encode=function(v)sent[#sent+1]=v;return '{}'end,decode=function()return table.remove(incoming,1)end}end
local client={settimeout=function()end,send=function(_,s)return #s end,close=function()end}
function client:receive()if #incoming>0 then return 'message' end;return nil,'timeout' end
package.preload.socket=function()local accepted=false;return {tcp=function()return {setoption=function()end,bind=function()return true end,listen=function()end,settimeout=function()end,accept=function()if not accepted then accepted=true;return client end end,close=function()end}end}end
assert(dofile('live_preview.lua').start());callbacks[2](mod)
incoming[1]={command='auto_start'};callbacks[2](mod);assert(#Isaac.GetRoomEntities()==0)
alive=true;callbacks[3](mod);callbacks[4](mod)
local function step(to)while frame<to do frame=frame+1;callbacks[1](mod)end end
step(6);assert(#Isaac.GetRoomEntities()==1 and Isaac.GetRoomEntities()[1].Type==12,'pending auto-start lost');assert(inj.enabled)
assert(callbacks[5](mod,p,1,0,nil)==false,'damage must count but protect hearts')
step(53);assert(#Isaac.GetRoomEntities()==1 and Isaac.GetRoomEntities()[1].Type==10,'next scenario did not start');assert(inj.enabled)
pressC=true;step(54);assert(not inj.enabled);local count=#entities;step(120);assert(#entities==count,'C stop restarted')
print('PASS: queued auto-start, room/game callbacks, Horf -> contact -> Gaper, protection and C stop')
