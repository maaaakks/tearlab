-- Realtime frozen-policy preview. No training, automatic restart, or save edits.
local M = {}
function M.start()
    local Config = require("config")
    if not Config.LIVE_PREVIEW then return false end
    local json = require("json")
    local Injector = require("action_injector")
    local mod = RegisterMod("IsaacRL", 1)
    local socketOK, socket = pcall(require, "socket")
    local listener, client = nil, nil
    local enabled, reason = false, "C: activer / couper"
    local buffer, outgoing = "", ""
    local tick, lastActionTick, decision, roomSerial, startTick, controlId = 0, -1000, 0, 0, 0, 0
    local lastAction = {action_id=0,move=0,shoot=0,bomb=0,active=0,pillcard=0}
    local events = {}
    local font=Font();font:Load("font/terminus.fnt")
    Injector.setEnabled(false)
    local function stop(why)
        enabled=false;reason=why or "MANUEL";Injector.setEnabled(false);Injector.reset()
        lastAction={action_id=0,move=0,shoot=0,bomb=0,active=0,pillcard=0}
    end
    local function safe(fn,default) local ok,v=pcall(fn);if ok then return v end;return default end
    local function send(message)
        if not client or #outgoing>0 then return false end
        outgoing=json.encode(message).."\n"
        return true
    end
    local function flush()
        if not client or #outgoing==0 then return end
        local sent,err,partial=client:send(outgoing)
        local count=sent or partial or 0
        outgoing=string.sub(outgoing,count+1)
        if err=="closed" then client:close();client=nil;outgoing="";buffer="";stop("Connexion perdue") end
    end
    local function player() return safe(function() return Isaac.GetPlayer(0) end,nil) end
    local function active()
        local p=player();return p~=nil and not p:IsDead() and Game():GetFrameCount()>0
    end
    local function arm()
        if not client then stop("Agent Python non connecte");return end
        if not active() then stop("Demarrer une partie avec Isaac");return end
        enabled=true;reason="AGENT";startTick=tick;decision=0;lastActionTick=tick;controlId=controlId+1
        Injector.reset();Injector.setEnabled(true)
    end
    local function snapshot()
        local p=player();if not p then return nil end
        local room=Game():GetRoom();local left=room:GetTopLeftPos();local right=room:GetBottomRightPos()
        local enemies,projectiles={},{}
        for _,ent in ipairs(Isaac.GetRoomEntities()) do
            if ent:IsActiveEnemy(false) and not ent:IsDead() then
                table.insert(enemies,{init_seed=ent.InitSeed,type=ent.Type,variant=ent.Variant,
                    hp=ent.HitPoints,max_hp=ent.MaxHitPoints,position={ent.Position.X,ent.Position.Y},
                    velocity={ent.Velocity.X,ent.Velocity.Y},attack_warning=0,warning_observed=false,
                    vulnerable=safe(function() return ent:IsVulnerableEnemy() end,true)})
            elseif ent.Type==2 or ent.Type==9 then
                table.insert(projectiles,{init_seed=ent.InitSeed,type=ent.Type,
                    friendly=ent.Type==2,position={ent.Position.X,ent.Position.Y},
                    velocity={ent.Velocity.X,ent.Velocity.Y},age_ticks=ent.FrameCount})
            end
        end
        table.sort(enemies,function(a,b)return a.init_seed<b.init_seed end)
        table.sort(projectiles,function(a,b)return a.init_seed<b.init_seed end)
        return {protocol="isaac-live-238-v1",kind="state",tick=tick,decision=decision,
            room_id=roomSerial,control_id=controlId,control_tick=tick-startTick,assistant_enabled=enabled,
            applied_action=lastAction,events=events,player_dead=p:IsDead(),room_cleared=room:IsClear(),
            player={position={p.Position.X,p.Position.Y},velocity={p.Velocity.X,p.Velocity.Y},
                fire_delay=p.FireDelay,damage=p.Damage,hp_red=p:GetHearts(),hp_soul=p:GetSoulHearts(),
                player_type=p:GetPlayerType(),collectible_count=p:GetCollectibleCount(),
                active_item_id=p:GetActiveItem(),immunity_ticks=safe(function()return p:GetDamageCooldown()end,0)},
            room={top_left={left.X,left.Y},width=right.X-left.X,height=right.Y-left.Y,
                shape=safe(function()return room:GetRoomShape()end,-1)},enemies=enemies,projectiles=projectiles}
    end
    local function message(msg)
        if msg.command=="release" then stop(msg.reason or "MANUEL") end
        if msg.action and enabled then
            local a=msg.action
            if a.room_id~=roomSerial or a.control_id~=controlId then return end
            if tonumber(a.move)==nil or tonumber(a.shoot)==nil or a.move<0 or a.move>8 or a.shoot<0 or a.shoot>4 then stop("Action invalide");return end
            lastAction={action_id=a.action_id or 0,move=a.move,shoot=a.shoot,bomb=0,active=0,pillcard=0}
            lastActionTick=tick;Injector.setAction(lastAction)
        end
    end
    local function poll()
        if not listener then return end
        if not client then
            local c=listener:accept()
            if c then
                client=c;client:settimeout(0);buffer="";outgoing="";stop("Python connecte - C pour activer")
                send({kind="hello",protocol="isaac-live-238-v1",control="manual",mod="IsaacRL realtime preview"})
            end
        end
        if not client then return end
        for _=1,32 do
            local line,err,partial=client:receive("*l")
            if partial and #partial>0 then buffer=buffer..partial end
            if line then
                local ok,msg=pcall(json.decode,buffer..line);buffer=""
                if ok then message(msg) else stop("Message invalide") end
            elseif err=="closed" then client:close();client=nil;buffer="";outgoing="";stop("Python deconnecte");break
            else break end
        end
        flush()
    end
    local function startServer()
        if listener or not socketOK then return end
        listener=socket.tcp();listener:setoption("reuseaddr",true)
        local ok,err=listener:bind("127.0.0.1",10009)
        if not ok then reason="Port 10009 indisponible";Isaac.ConsoleOutput("IsaacRL preview: "..tostring(err).."\n");listener:close();listener=nil;return end
        listener:listen(1);listener:settimeout(0)
        Isaac.ConsoleOutput("IsaacRL preview: READY 127.0.0.1:10009 - C toggles, no automatic actions\n")
    end
    function mod:onUpdate()
        tick=Game():GetFrameCount();poll()
        if Input.IsButtonTriggered(Keyboard.KEY_C,0) then if enabled then stop("MANUEL - C pour activer") else arm() end end
        if enabled and tick-lastActionTick>15 then stop("Agent trop lent - controle coupe") end
        if tick%3==0 and client then
            decision=decision+1
            local state=snapshot()
            if state then
                if enabled and (#state.enemies==0 or #state.enemies>4 or state.player_dead) then
                    stop(#state.enemies>4 and "Plus de 4 ennemis: MANUEL" or "Fin du combat: MANUEL");state.assistant_enabled=false
                end
                if send(state) then events={} end;flush()
            end
        end
        Injector.tick()
    end
    function mod:onRender()
        startServer();poll()
        if enabled and safe(function()return Game():IsPaused()end,false) then stop("Pause - MANUEL") end
        if font:IsLoaded() then
            local text="ISAAC RL | "..(enabled and "AGENT" or "MANUEL").." | C: ON/OFF"
            font:DrawStringScaled(text,38,42,.55,.55,KColor(.4,1,.8,1),0,false)
            font:DrawStringScaled(reason,38,52,.5,.5,KColor(1,1,1,1),0,false)
        end
    end
    function mod:onRoom() roomSerial=roomSerial+1;startTick=0;decision=0;stop("Nouvelle salle - C pour activer") end
    function mod:onStart() roomSerial=roomSerial+1;tick=0;decision=0;events={};stop("Pret - C pour activer") end
    function mod:onDamage(entity,amount,flags,source)
        if entity and entity:ToPlayer() then
            table.insert(events,{name="player_damaged",amount=amount});stop("Degat recu - MANUEL (C pour reprendre)")
        elseif entity and entity:ToNPC() then table.insert(events,{name="enemy_damaged",target_seed=entity.InitSeed,amount=amount}) end
    end
    function mod:onInput(entity,hook,button)
        if not enabled or not client then return nil end
        return Injector.onInputAction(nil,entity,hook,button)
    end
    function mod:onCommand(command,args)
        if command~="rlpreview" then return end
        local kind,count=string.match(args or "","^(%a+)%s+(%d+)$")
        count=tonumber(count)
        if (kind~="horf" and kind~="gaper") or not count or count<1 or count>4 or not active() then
            Isaac.ConsoleOutput("rlpreview horf 1|2|3|4 ou rlpreview gaper 1|2|3|4 : remplace les ennemis de la salle courante. C active l'agent.\n");return
        end
        stop("Salle de test vanilla - C pour activer")
        for _,e in ipairs(Isaac.GetRoomEntities()) do if e:IsActiveEnemy(false) or e.Type==2 or e.Type==9 then e:Remove() end end
        local p=player();local room=Game():GetRoom();local center=room:GetCenterPos();p.Position=center;p.Velocity=Vector(0,0)
        for i=1,count do
            local a=(i-1)*math.pi*2/count
            local pos=center+Vector(math.cos(a)*100,math.sin(a)*100)
            Isaac.Spawn(kind=="horf" and 12 or 10,0,0,pos,Vector(0,0),nil)
        end
        room:SetClear(false);roomSerial=roomSerial+1;decision=0
        Isaac.ConsoleOutput("IsaacRL preview: monstres vanilla, IA non modifiee; pas des tireurs synthetiques du simulateur.\n")
    end
    mod:AddCallback(ModCallbacks.MC_POST_UPDATE,mod.onUpdate)
    mod:AddCallback(ModCallbacks.MC_POST_RENDER,mod.onRender)
    mod:AddCallback(ModCallbacks.MC_POST_NEW_ROOM,mod.onRoom)
    mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED,mod.onStart)
    mod:AddCallback(ModCallbacks.MC_ENTITY_TAKE_DMG,mod.onDamage)
    mod:AddCallback(ModCallbacks.MC_INPUT_ACTION,mod.onInput)
    mod:AddCallback(ModCallbacks.MC_EXECUTE_CMD,mod.onCommand)
    startServer()
    if not socketOK then reason="LuaSocket absent - aucun controle agent";Isaac.ConsoleOutput("IsaacRL preview: socket unavailable: "..tostring(socket).."\n") end
    return true
end
return M
