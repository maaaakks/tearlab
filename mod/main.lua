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
local debugInfo = nil  -- reward overlay from Python

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
        if message.debug then
            debugInfo = message.debug
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

    -- Apply latched action
    ActionInjector.setAction(lastAction)
end

-- Accept connections and handle configure during non-gameplay states
function mod:onRender()
    if not serverStarted then
        return
    end

    -- Draw debug reward overlay
    if debugInfo then
        local font = Font()
        font:Load("font/terminus.fnt")
        if font:IsLoaded() then
            local x = 40
            local y = 30
            local lineH = 12
            local white = KColor(1, 1, 1, 0.9)
            local green = KColor(0.2, 1, 0.2, 0.9)
            local red = KColor(1, 0.3, 0.3, 0.9)

            font:DrawString("-- Reward Debug --", x, y, white, 0, false)
            y = y + lineH + 2

            local function drawReward(label, value, y)
                local color = white
                if value > 0.001 then color = green
                elseif value < -0.001 then color = red end
                font:DrawString(string.format("%-18s %+.3f", label, value), x, y, color, 0, false)
                return y + lineH
            end

            y = drawReward("step_reward", debugInfo.step_reward or 0, y)
            y = drawReward("pickup_approach", debugInfo.pickup_approach or 0, y)
            y = drawReward("pickup_collected", debugInfo.pickup_collected or 0, y)
            y = drawReward("wall_collision", debugInfo.wall_collision or 0, y)
            y = drawReward("time", debugInfo.time or 0, y)
            y = y + 4
            font:DrawString(string.format("ep_reward: %.1f", debugInfo.reward or 0), x, y, white, 0, false)
        end
    end

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
