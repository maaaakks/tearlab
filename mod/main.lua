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
-- Explicit end-of-episode request from Python (victory / abandon). Consumed by
-- the terminal detection and cleared on the next run start.
local requestedTerminalReason = nil

-- B2 lifecycle events: one-shot events attached to the next successfully sent
-- state. Python consumes them for rewards and run-level metrics.
local pendingEvents = {}
local lastRoomKey = -1
local lastStage = -1
local lastStageType = -1
local lastRoomClear = false
local ROOM_TYPE_BOSS = 5  -- RoomType.ROOM_BOSS

local function queueEvent(name, fields)
    local event = {name = name, tick = episodeTick}
    if fields then
        for key, value in pairs(fields) do
            event[key] = value
        end
    end
    table.insert(pendingEvents, event)
end

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

local function shortTerminalReason(reason)
    local labels = {
        room_cleared = "CLEAR",
        death = "DEATH",
        timeout = "TIMEOUT",
        pickup_target_reached = "TARGET",
        pickup_collected = "PICKUP",
        unknown = "--",
    }
    local raw = tostring(reason or "unknown")
    return labels[raw] or string.upper(string.sub(raw, 1, 8))
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
            "R %+.1f   AVG%d %+.1f",
            tonumber(info.episode_reward) or 0,
            tonumber(info.rolling_window) or 20,
            tonumber(info.rolling_reward) or 0
        ),
        x, y, HUD_COLOR.white
    )
    y = y + HUD_LINE_HEIGHT

    local lastReason = shortTerminalReason(info.last_episode_reason)
    local lastColor = HUD_COLOR.muted
    if lastReason == "CLEAR" or lastReason == "TARGET" then
        lastColor = HUD_COLOR.green
    elseif lastReason == "DEATH" then
        lastColor = HUD_COLOR.red
    elseif lastReason == "TIMEOUT" then
        lastColor = HUD_COLOR.yellow
    end
    drawHudText(
        string.format(
            "LAST %+.1f L%d %s",
            tonumber(info.last_episode_reward) or 0,
            tonumber(info.last_episode_length) or 0,
            lastReason
        ),
        x, y, lastColor
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
    requestedTerminalReason = nil
    pendingEvents = {}
    lastRoomKey = -1
    lastStage = -1
    lastStageType = -1
    lastRoomClear = false
    hudContextText = nil
    hudContextExpiresAt = 0
    ActionInjector.reset()

    -- Per-episode logging lives on the Python side (one line per episode in
    -- live_trainer); the Lua side stays silent to keep the console clean.
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
        end    elseif message.command == "pause" then
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
    elseif message.command == "controlled_reset" then
        -- E6/E7: rare-scenario start request. Scenario injection (A4) is not
        -- implemented yet, so this is currently a hook that performs a plain
        -- episode reset; the scheduler in Python still manages the fraction.
        paused = false
        Isaac.ConsoleOutput(
            "IsaacRL[" .. Config.INSTANCE_ID .. "]: controlled_reset ("
                .. tostring(message.scenario or "natural") .. ") -> plain reset\n"
        )
        ActionInjector.reset()
        lastAction = {move = 0, shoot = 0}
        GameControl.resetEpisode()
    elseif message.command == "end_episode" then
        -- Explicit terminal from Python: "run_won" (victory/final objective)
        -- or "abandoned". Ends the current episode without waiting for death
        -- or timeout. In full_run mode this is how a run is closed cleanly.
        requestedTerminalReason = message.reason or "abandoned"
        Isaac.ConsoleOutput(
            "IsaacRL[" .. Config.INSTANCE_ID .. "]: Received end_episode ("
                .. requestedTerminalReason .. ")\n"
        )
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

    -- Human demo mode: attach the live input encoding so Python can record
    -- (obs, action) pairs aligned with the agent's action vocabulary.
    if Config.HUMAN_MODE then
        state.action = ActionInjector.readHumanAction()
    end

    -- B2 lifecycle events: one-shot diffs against the last seen room/floor.
    local roomKey = state.run.room_list_index
    if roomKey == nil or roomKey < 0 then
        roomKey = state.run.room_grid_index
    end
    if lastRoomKey == -1 and roomKey >= 0 then
        queueEvent("room_entered", {
            room_list_index = state.run.room_list_index,
            room_type = state.run.room_type,
        })
    elseif lastRoomKey >= 0 and roomKey >= 0 and roomKey ~= lastRoomKey then
        queueEvent("room_entered", {
            room_list_index = state.run.room_list_index,
            room_type = state.run.room_type,
        })
    end

    local stage = state.run.stage
    local stageType = state.run.stage_type
    if lastStage == -1 and stage >= 0 then
        queueEvent("floor_entered", {stage = stage, stage_type = stageType})
    elseif lastStage >= 0 and (stage ~= lastStage or stageType ~= lastStageType) then
        queueEvent("floor_cleared", {stage = lastStage, stage_type = lastStageType})
        queueEvent("floor_entered", {stage = stage, stage_type = stageType})
    end

    if state.run.room_clear and not lastRoomClear then
        queueEvent("room_cleared", {
            room_list_index = state.run.room_list_index,
            room_type = state.run.room_type,
        })
        if state.run.room_type == ROOM_TYPE_BOSS then
            queueEvent("boss_killed", {
                room_list_index = state.run.room_list_index,
            })
        end
    end

    lastRoomKey = roomKey
    lastStage = stage
    lastStageType = stageType
    lastRoomClear = state.run.room_clear

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
    elseif requestedTerminalReason then
        terminal = true
        terminalReason = requestedTerminalReason
    elseif not Config.FULL_RUN and Config.TERMINAL_PICKUP_COUNT > 0
        and pickupsCollected >= Config.TERMINAL_PICKUP_COUNT then
        terminal = true
        terminalReason = "pickup_target_reached"
    elseif not Config.FULL_RUN and Config.TERMINAL_ON_PICKUP and pickupsCollected > 0 then
        -- Legacy single-pickup terminal (equivalent to TERMINAL_PICKUP_COUNT = 1)
        terminal = true
        terminalReason = "pickup_collected"
    elseif not Config.FULL_RUN and hadEnemies and state.enemy_count == 0 then
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

    -- Explicit victory closes the run with a run_won event on the final state.
    if terminal and terminalReason == "run_won" then
        queueEvent("run_won", {})
    end

    -- Attach queued one-shot lifecycle events to this state.
    state.events = pendingEvents

    -- Send state to Python. In free-running mode the send is non-blocking:
    -- if Python is busy running a PPO update the frame is dropped and the
    -- game keeps playing at full speed with the last latched action.
    local sent
    if Config.FREE_RUNNING then
        sent = server:sendStateNonBlocking(state)
    else
        sent = server:sendState(state)
    end

    -- Events are only cleared when Python actually received the state; a
    -- dropped frame keeps them queued for the next successful send.
    if sent then
        pendingEvents = {}
    end

    -- If terminal: restart immediately regardless of whether Python saw the
    -- final frame (Python resyncs via episode_id).
    if terminal then
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

    if Config.HUMAN_MODE then
        -- Human drives: no action to receive, nothing to inject. The next
        -- decision tick re-reads the keyboard.
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

-- Speedhack: run extra full game updates per rendered frame. Each nested
-- Game():Update() advances one real game tick (entities, physics, timers,
-- MC_POST_UPDATE), so x8 processes ~8x ticks per wall-clock second. Render
-- callbacks do not fire inside nested updates, so this is not re-entrant.
local function applySpeedHack()
    local mult = Config.SPEED_MULTIPLIER
    if mult <= 1 or paused or GameControl.isResetting() then
        return
    end
    local player = Isaac.GetPlayer(0)
    if not player or player:IsDead() then
        return
    end
    for _ = 1, mult - 1 do
        Game():Update()
        -- A terminal/reset may have fired inside this nested tick (episode
        -- ends call ExecuteCommand("restart")). Never keep updating a game
        -- that is mid-restart or has no living player.
        local nestedPlayer = Isaac.GetPlayer(0)
        if GameControl.isResetting() or not nestedPlayer or nestedPlayer:IsDead() then
            break
        end
    end
end

-- Accept connections and handle configure during non-gameplay states
function mod:onRender()
    if not serverStarted then
        return
    end

    drawTrainingHud(debugInfo)

    applySpeedHack()

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

-- Input interception callback. In human demo mode we return nil everywhere:
-- the real keyboard drives Isaac and Python only records.
function mod:onInputAction(entity, inputHook, buttonAction)
    if not server.connected or Config.HUMAN_MODE then
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
