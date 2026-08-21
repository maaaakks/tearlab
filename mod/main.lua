local Config = require("config")
local TcpServer = require("tcp_server")
local StateSerializer = require("state_serializer")
local ActionInjector = require("action_injector")
local GameControl = require("game_control")

local mod = RegisterMod("IsaacRL", 1)

local server = TcpServer.new(Config.TCP_HOST, Config.TCP_PORT, Config.TCP_TIMEOUT)
local serverStarted = false
local tickCount = 0

-- Episode lifecycle (Lua owns)
local episodeId = 0
local episodeTick = 0
local hadEnemies = false
local episodeStartCoins = 0
local pickupsCollected = 0
local lastAction = {move = 0, shoot = 0}
local paused = false
local debugInfo = nil  -- compact training HUD payload from Python
local hudFont = Font()
hudFont:Load("font/terminus.fnt")

-- Keep the always-on HUD tucked under the minimap. Detailed metrics remain in
-- metrics.jsonl; the fourth line is reserved for short-lived useful events.
local HUD_SCALE = 0.50
local HUD_X = 335
local HUD_Y = 67
local HUD_LINE_HEIGHT = 9
local HUD_SHADOW_OFFSET = 0.75
local HUD_EVENT_FRAMES = 60
local HUD_UPDATE_FRAMES = 90

local HUD_COLOR = {
    shadow = KColor(0.0, 0.0, 0.0, 0.95),
    white = KColor(0.98, 0.98, 1.0, 1.0),
    muted = KColor(0.74, 0.80, 0.88, 1.0),
    cyan = KColor(0.30, 0.90, 1.0, 1.0),
    green = KColor(0.40, 1.0, 0.58, 1.0),
    yellow = KColor(1.0, 0.82, 0.28, 1.0),
    red = KColor(1.0, 0.42, 0.42, 1.0),
}

local hudContextText = nil
local hudContextColor = HUD_COLOR.muted
local hudContextExpiresAt = 0
local lastHudUpdateCount = 0

local function compactNumber(value)
    local number = tonumber(value) or 0
    if math.abs(number) >= 1000000 then
        return string.format("%.1fm", number / 1000000)
    elseif math.abs(number) >= 1000 then
        return string.format("%.1fk", number / 1000)
    end
    return tostring(math.floor(number + 0.5))
end

local function drawHudText(text, x, y, color)
    hudFont:DrawStringScaled(
        tostring(text), x + HUD_SHADOW_OFFSET, y + HUD_SHADOW_OFFSET,
        HUD_SCALE, HUD_SCALE, HUD_COLOR.shadow, 0, false
    )
    hudFont:DrawStringScaled(
        tostring(text), x, y, HUD_SCALE, HUD_SCALE,
        color or HUD_COLOR.white, 0, false
    )
end

local function updateHudContext(info)
    if not info then
        return
    end

    local frame = Game():GetFrameCount()
    local updateCount = tonumber(info.update_count) or 0
    local hasNewUpdate = updateCount > lastHudUpdateCount
    if updateCount > lastHudUpdateCount then
        lastHudUpdateCount = updateCount
    end

    local stepReward = tonumber(info.step_reward) or 0
    local eventLabel = nil
    local eventValue = 0
    local eventColor = HUD_COLOR.green

    if math.abs(tonumber(info.kills) or 0) > 0.001 then
        eventLabel = "KILL"
        eventValue = tonumber(info.kills) or 0
    elseif math.abs(tonumber(info.damage_taken) or 0) > 0.001 then
        eventLabel = "HURT"
        eventValue = tonumber(info.damage_taken) or 0
        eventColor = HUD_COLOR.red
    elseif math.abs(tonumber(info.damage_dealt) or 0) > 0.001 then
        eventLabel = "DMG"
        eventValue = tonumber(info.damage_dealt) or 0
    elseif math.abs(tonumber(info.pickup_collected) or 0) > 0.001 then
        eventLabel = "PICK"
        eventValue = tonumber(info.pickup_collected) or 0
    elseif math.abs(tonumber(info.wall_collision) or 0) > 0.001 then
        eventLabel = "WALL"
        eventValue = tonumber(info.wall_collision) or 0
        eventColor = HUD_COLOR.red
    end

    if eventLabel then
        hudContextText = string.format(
            "%s %+.1f  R%+.3f", eventLabel, eventValue, stepReward
        )
        hudContextColor = eventColor
        hudContextExpiresAt = frame + HUD_EVENT_FRAMES
    elseif hasNewUpdate then
        hudContextText = string.format(
            "KL %.3f H %.2f EV %+.2f",
            tonumber(info.approx_kl) or 0,
            tonumber(info.entropy) or 0,
            tonumber(info.explained_variance) or 0
        )
        hudContextColor = HUD_COLOR.cyan
        hudContextExpiresAt = frame + HUD_UPDATE_FRAMES
    end
end

local function drawTrainingHud(info)
    if not info or not hudFont:IsLoaded() then
        return
    end

    local x = HUD_X
    local y = HUD_Y
    local steps = tonumber(info.global_steps) or 0
    drawHudText(
        string.format(
            "STEP %s   EP %d",
            compactNumber(steps),
            tonumber(info.episode) or 0
        ),
        x, y, HUD_COLOR.white
    )
    y = y + HUD_LINE_HEIGHT

    drawHudText(
        string.format(
            "R %+.1f   AVG %+.1f",
            tonumber(info.episode_reward) or 0,
            tonumber(info.rolling_reward) or 0
        ),
        x, y, HUD_COLOR.white
    )
    y = y + HUD_LINE_HEIGHT

    drawHudText(
        "SAVE " .. compactNumber(info.last_checkpoint or 0),
        x, y, HUD_COLOR.muted
    )
    y = y + HUD_LINE_HEIGHT

    if hudContextText and Game():GetFrameCount() <= hudContextExpiresAt then
        drawHudText(hudContextText, x, y, hudContextColor)
    else
        hudContextText = nil
    end
end

-- Initialize on game start
function mod:onGameStart(isContinue)
    if not serverStarted then
        serverStarted = server:start()
        ActionInjector.init()
    end
    GameControl.onGameStart()

    -- New episode
    episodeId = episodeId + 1
    episodeTick = 0
    hadEnemies = false
    local player = Isaac.GetPlayer(0)
    episodeStartCoins = player and player:GetNumCoins() or 0
    pickupsCollected = 0
    lastAction = {move = 0, shoot = 0}
    hudContextText = nil
    hudContextExpiresAt = 0
    ActionInjector.reset()

    Isaac.ConsoleOutput("IsaacRL[" .. Config.INSTANCE_ID .. "]: Episode " .. episodeId .. " started (port " .. Config.TCP_PORT .. ")\n")
end

-- Handle new room
function mod:onNewRoom()
    GameControl.onNewRoom()
end

-- Process a command message from Python
local function handleMessage(message)
    if message.command == "configure" then
        if message.settings then
            GameControl.configure(message.settings)
            if message.settings.disable_shooting ~= nil then
                ActionInjector.setDisableShooting(message.settings.disable_shooting)
            end
        end
    elseif message.command == "pause" then
        paused = true
        Isaac.ConsoleOutput("IsaacRL[" .. Config.INSTANCE_ID .. "]: Received pause command\n")
    elseif message.command == "resume" then
        paused = false
        Isaac.ConsoleOutput("IsaacRL[" .. Config.INSTANCE_ID .. "]: Received resume command\n")
    elseif message.command == "start_run" then
        -- Best-effort clean restart for a newly attached trainer session.
        paused = false
        Isaac.ConsoleOutput("IsaacRL[" .. Config.INSTANCE_ID .. "]: Received start_run command\n")
        ActionInjector.reset()
        lastAction = {move = 0, shoot = 0}
        GameControl.resetEpisode()
    elseif message.command == "reset" then
        -- Manual/initial reset
        paused = false
        Isaac.ConsoleOutput("IsaacRL[" .. Config.INSTANCE_ID .. "]: Received reset command\n")
        ActionInjector.reset()
        lastAction = {move = 0, shoot = 0}
        GameControl.resetEpisode()
    elseif message.action then
        lastAction = message.action
        -- Arm one-shot interactions (bomb/active/pillcard) for this new
        -- decision only; latched re-applies must not re-trigger them.
        ActionInjector.armInteractions(lastAction)
        if message.debug then
            debugInfo = message.debug
            updateHudContext(debugInfo)
        end
    end
end

-- Main update loop - runs every game tick (30/sec)
function mod:onUpdate()
    if not serverStarted then
        return
    end

    -- Try to accept a client if not connected
    if not server.connected then
        server:acceptClient()
        return
    end

    -- Poll for commands even when paused (to receive resume)
    if paused then
        while true do
            local message = server:pollAction()
            if not message then break end
            handleMessage(message)
        end
        return
    end

    tickCount = tickCount + 1

    -- Skip frames if configured
    if tickCount % Config.FRAME_SKIP ~= 0 then
        return
    end

    -- Don't send state during reset transition
    if GameControl.isResetting() then
        return
    end

    local game = Game()
    local player = Isaac.GetPlayer(0)
    if not player then
        return
    end

    -- Process any pending pickup respawns from collision callbacks
    GameControl.processPendingSpawns()

    episodeTick = episodeTick + 1

    -- Serialize state
    local state = StateSerializer.serialize(game)

    -- Track whether enemies have appeared this episode
    if state.enemy_count > 0 then
        hadEnemies = true
    end

    -- Track pickups collected this episode
    local currentCoins = player:GetNumCoins()
    local newPickups = currentCoins - episodeStartCoins
    if newPickups > pickupsCollected then
        pickupsCollected = newPickups
    end

    -- Detect terminal conditions
    local terminal = false
    local terminalReason = nil

    if player:IsDead() then
        terminal = true
        terminalReason = "death"
    elseif Config.TERMINAL_PICKUP_COUNT > 0 and pickupsCollected >= Config.TERMINAL_PICKUP_COUNT then
        terminal = true
        terminalReason = "pickup_target_reached"
    elseif Config.TERMINAL_ON_PICKUP and pickupsCollected > 0 then
        -- Legacy single-pickup terminal (equivalent to TERMINAL_PICKUP_COUNT = 1)
        terminal = true
        terminalReason = "pickup_collected"
    elseif hadEnemies and state.enemy_count == 0 then
        terminal = true
        terminalReason = "room_cleared"
    elseif Config.MAX_EPISODE_TICKS > 0 and episodeTick >= Config.MAX_EPISODE_TICKS then
        terminal = true
        terminalReason = "timeout"
    end

    -- Add episode metadata
    state.episode_id = episodeId
    state.episode_tick = episodeTick
    state.pickups_collected = pickupsCollected
    state.terminal = terminal
    state.terminal_reason = terminalReason

    -- Send state to Python. In free-running mode the send is non-blocking:
    -- if Python is busy running a PPO update the frame is dropped and the
    -- game keeps playing at full speed with the last latched action.
    local sent
    if Config.FREE_RUNNING then
        sent = server:sendStateNonBlocking(state)
    else
        sent = server:sendState(state)
    end

    -- If terminal: restart immediately regardless of whether Python saw the
    -- final frame (Python resyncs via episode_id).
    if terminal then
        Isaac.ConsoleOutput("IsaacRL[" .. Config.INSTANCE_ID .. "]: Episode " .. episodeId .. " ended (" .. terminalReason .. ")\n")
        ActionInjector.reset()
        lastAction = {move = 0, shoot = 0}
        GameControl.resetEpisode()
        -- episodeId incremented in onGameStart
        return
    end

    if not sent then
        ActionInjector.setAction(lastAction)
        return
    end

    if Config.FREE_RUNNING then
        -- Poll for a new action without ever blocking the game loop.
        while true do
            local message = server:pollAction()
            if not message then break end
            handleMessage(message)
            if GameControl.isResetting() then
                return
            end
        end
    else
        -- Lock-step protocol: block until Python sends the next action.
        -- This prevents the game from running ahead of the agent, which would
        -- cause frame drops and corrupt the training signal.
        local message = server:waitForAction()
        if message then
            handleMessage(message)
            if GameControl.isResetting() then
                return
            end
        end
    end

    -- Expire one-shot interaction window, then apply latched action
    ActionInjector.tick()
    ActionInjector.setAction(lastAction)
end

-- Accept connections and handle configure during non-gameplay states
function mod:onRender()
    if not serverStarted then
        return
    end

    drawTrainingHud(debugInfo)

    if not server.connected then
        server:acceptClient()
        return
    end

    -- Only poll during non-gameplay states (death screen, menus, resetting)
    local player = Isaac.GetPlayer(0)
    if player and not player:IsDead() and not GameControl.isResetting() then
        return  -- onUpdate handles it
    end

    -- Drain buffered messages (configure, manual reset, start_run)
    while true do
        local message = server:pollAction()
        if not message then break end
        handleMessage(message)
    end
end

-- Input interception callback
function mod:onInputAction(entity, inputHook, buttonAction)
    if not server.connected then
        return nil
    end
    return ActionInjector.onInputAction(nil, entity, inputHook, buttonAction)
end

-- Pickup collection detection: respawn a new pickup when one is collected
function mod:onPickupCollision(pickup, collider, low)
    if not Config.RESPAWN_PICKUP then
        return
    end
    -- Only handle coins (variant 20)
    local coinVariant = (PickupVariant and PickupVariant.PICKUP_COIN) or 20
    if pickup.Variant ~= coinVariant then
        return
    end
    -- Only trigger for player collisions
    if not collider or collider.Type ~= 1 then
        return
    end
    -- Schedule a new penny spawn on next frame (can't spawn during collision)
    GameControl.schedulePickupRespawn()
end

-- Register callbacks
mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, mod.onGameStart)
mod:AddCallback(ModCallbacks.MC_POST_NEW_ROOM, mod.onNewRoom)
mod:AddCallback(ModCallbacks.MC_POST_UPDATE, mod.onUpdate)
mod:AddCallback(ModCallbacks.MC_POST_RENDER, mod.onRender)
mod:AddCallback(ModCallbacks.MC_INPUT_ACTION, mod.onInputAction)
mod:AddCallback(ModCallbacks.MC_PRE_PICKUP_COLLISION, mod.onPickupCollision, 20)  -- 20 = PickupVariant.PICKUP_COIN

Isaac.ConsoleOutput("IsaacRL[" .. Config.INSTANCE_ID .. "]: Mod loaded (port " .. Config.TCP_PORT .. ")\n")
