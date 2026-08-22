local Config = require("config")

local GameControl = {}

local waitingForReset = false
local resetFrame = 0
local RESET_DELAY = 5  -- frames to wait after reset command before sending state
local pendingPickupRespawn = false
local spawnedObstacleIndices = {}

local function shuffleInPlace(items)
    for i = #items, 2, -1 do
        local j = math.random(i)
        items[i], items[j] = items[j], items[i]
    end
end

local function clearSpawnedObstacles(game)
    local room = game and game:GetRoom()
    if not room then
        spawnedObstacleIndices = {}
        return
    end

    for _, idx in ipairs(spawnedObstacleIndices) do
        if room:GetGridEntity(idx) then
            room:RemoveGridEntity(idx, 0, false)
        end
    end
    spawnedObstacleIndices = {}
end

function GameControl.resetEpisode()
    clearSpawnedObstacles(Game())
    pendingPickupRespawn = false
    Isaac.ExecuteCommand("restart")
    waitingForReset = true
    resetFrame = 0
end

function GameControl.isResetting()
    return waitingForReset
end

function GameControl.onGameStart()
    if waitingForReset then
        resetFrame = Game():GetFrameCount()
    end
end

function GameControl.onNewRoom()
    if waitingForReset then
        local game = Game()
        GameControl.spawnObstacles(game)
        -- Spawn configured enemies
        if Config.SPAWN_ENEMIES then
            GameControl.spawnEnemies(game)
        end
        if Config.SPAWN_PICKUP_PENNY then
            GameControl.spawnPenny(game)
        end
        -- Teleport player to room center to avoid spawn-position bias
        local player = Isaac.GetPlayer(0)
        local room = game:GetRoom()
        player.Position = room:GetCenterPos()
        waitingForReset = false
    end
end

function GameControl.spawnObstacles(game)
    spawnedObstacleIndices = {}
    if not Config.SPAWN_OBSTACLES or Config.OBSTACLE_COUNT <= 0 then
        return
    end

    local room = game:GetRoom()
    local gridWidth = room:GetGridWidth()
    local gridSize = room:GetGridSize()
    local gridHeight = math.max(1, math.floor(gridSize / gridWidth))
    local centerIdx = room:GetClampedGridIndex(room:GetCenterPos())
    local centerGX = centerIdx % gridWidth
    local centerGY = math.floor(centerIdx / gridWidth)
    local spacing = Config.OBSTACLE_MIN_SPACING or 2
    local candidates = {}

    for gy = 1, gridHeight - 2 do
        for gx = 1, gridWidth - 2 do
            if math.abs(gx - centerGX) > 1 or math.abs(gy - centerGY) > 1 then
                local idx = gy * gridWidth + gx
                if not room:GetGridEntity(idx) then
                    table.insert(candidates, {gx = gx, gy = gy, idx = idx})
                end
            end
        end
    end

    shuffleInPlace(candidates)

    local placed = {}
    for _, candidate in ipairs(candidates) do
        if #placed >= Config.OBSTACLE_COUNT then
            break
        end

        local tooClose = false
        for _, existing in ipairs(placed) do
            if math.abs(candidate.gx - existing.gx) < spacing
                and math.abs(candidate.gy - existing.gy) < spacing then
                tooClose = true
                break
            end
        end

        if not tooClose then
            room:SpawnGridEntity(candidate.idx, Config.OBSTACLE_TYPE, 0, 0, 0)
            if room:GetGridEntity(candidate.idx) then
                table.insert(placed, candidate)
                table.insert(spawnedObstacleIndices, candidate.idx)
            end
        end
    end
end

local function getSpawnRadius()
    local minRadius = Config.SPAWN_RADIUS_MIN or 80
    local maxRadius = Config.SPAWN_RADIUS_MAX or minRadius
    if maxRadius < minRadius then
        minRadius, maxRadius = maxRadius, minRadius
    end
    return minRadius, maxRadius
end

local function getRandomSpawnPos(room, centerPos, minRadiusOverride, maxRadiusOverride)
    local minRadius, maxRadius
    if minRadiusOverride ~= nil or maxRadiusOverride ~= nil then
        minRadius = minRadiusOverride or maxRadiusOverride or 80
        maxRadius = maxRadiusOverride or minRadius
        if maxRadius < minRadius then
            minRadius, maxRadius = maxRadius, minRadius
        end
    else
        minRadius, maxRadius = getSpawnRadius()
    end
    local angle = math.random() * (math.pi * 2)
    local radius = minRadius
    if maxRadius > minRadius then
        radius = minRadius + math.random() * (maxRadius - minRadius)
    end

    local candidate = Vector(
        centerPos.X + math.cos(angle) * radius,
        centerPos.Y + math.sin(angle) * radius
    )
    return room:FindFreePickupSpawnPosition(candidate, 0, true)
end

function GameControl.spawnEnemies(game)
    local room = game:GetRoom()
    local centerPos = room:GetCenterPos()

    for i = 1, Config.ENEMY_COUNT do
        local spawnPos
        if Config.RANDOM_SPAWN_POSITIONS then
            spawnPos = getRandomSpawnPos(room, centerPos)
        else
            -- Offset enemies so they don't stack
            local offset = Vector(
                (i - 1) * 60 - (Config.ENEMY_COUNT - 1) * 30,
                0
            )
            spawnPos = Vector(centerPos.X + offset.X, centerPos.Y + offset.Y)
        end

        local enemy = Isaac.Spawn(
            Config.ENEMY_TYPE,
            Config.ENEMY_VARIANT,
            0,          -- subtype
            spawnPos,
            Vector(0, 0), -- velocity
            nil         -- spawner
        )
        if enemy and Config.ENEMY_COLLISION_DAMAGE ~= nil then
            enemy.CollisionDamage = Config.ENEMY_COLLISION_DAMAGE
        end
    end
end

function GameControl.spawnPenny(game)
    local room = game:GetRoom()
    local centerPos = room:GetCenterPos()
    local spawnPos

    if Config.PICKUP_RANDOM_POSITION then
        spawnPos = getRandomSpawnPos(
            room,
            centerPos,
            Config.PICKUP_RADIUS_MIN,
            Config.PICKUP_RADIUS_MAX
        )
    else
        local targetPos = Vector(
            centerPos.X + (Config.PICKUP_OFFSET_X or 180),
            centerPos.Y + (Config.PICKUP_OFFSET_Y or 0)
        )
        spawnPos = room:FindFreePickupSpawnPosition(targetPos, 0, true)
    end

    local pickupType = (EntityType and EntityType.ENTITY_PICKUP) or 5
    local coinVariant = (PickupVariant and PickupVariant.PICKUP_COIN) or 20
    local pennySubType = (CoinSubType and CoinSubType.COIN_PENNY) or 1

    local spawned = Isaac.Spawn(
        pickupType,
        coinVariant,
        pennySubType,
        spawnPos,
        Vector(0, 0),
        nil
    )
    if spawned then
        Isaac.ConsoleOutput(
            string.format(
                "IsaacRL: Spawned penny at (%.1f, %.1f)\n",
                spawnPos.X,
                spawnPos.Y
            )
        )
    else
        Isaac.ConsoleOutput("IsaacRL: Failed to spawn penny\n")
    end
end

function GameControl.schedulePickupRespawn()
    pendingPickupRespawn = true
end

function GameControl.processPendingSpawns()
    if pendingPickupRespawn then
        pendingPickupRespawn = false
        GameControl.spawnPenny(Game())
    end
end

function GameControl.configure(settings)
    if settings.enemy_type then
        Config.ENEMY_TYPE = settings.enemy_type
    end
    if settings.enemy_variant then
        Config.ENEMY_VARIANT = settings.enemy_variant
    end
    if settings.enemy_count then
        Config.ENEMY_COUNT = settings.enemy_count
    end
    if settings.enemy_collision_damage ~= nil then
        Config.ENEMY_COLLISION_DAMAGE = settings.enemy_collision_damage
    end
    if settings.spawn_pickup_penny ~= nil then
        Config.SPAWN_PICKUP_PENNY = settings.spawn_pickup_penny
    end
    if settings.pickup_random_position ~= nil then
        Config.PICKUP_RANDOM_POSITION = settings.pickup_random_position
    end
    if settings.pickup_offset_x ~= nil then
        Config.PICKUP_OFFSET_X = settings.pickup_offset_x
    end
    if settings.pickup_offset_y ~= nil then
        Config.PICKUP_OFFSET_Y = settings.pickup_offset_y
    end
    if settings.pickup_radius_min ~= nil then
        Config.PICKUP_RADIUS_MIN = settings.pickup_radius_min
    end
    if settings.pickup_radius_max ~= nil then
        Config.PICKUP_RADIUS_MAX = settings.pickup_radius_max
    end
    if settings.terminal_on_pickup ~= nil then
        Config.TERMINAL_ON_PICKUP = settings.terminal_on_pickup
    end
    if settings.terminal_pickup_count ~= nil then
        Config.TERMINAL_PICKUP_COUNT = settings.terminal_pickup_count
    end
    if settings.respawn_pickup ~= nil then
        Config.RESPAWN_PICKUP = settings.respawn_pickup
    end
    if settings.random_spawn_positions ~= nil then
        Config.RANDOM_SPAWN_POSITIONS = settings.random_spawn_positions
    end
    if settings.spawn_radius_min then
        Config.SPAWN_RADIUS_MIN = settings.spawn_radius_min
    end
    if settings.spawn_radius_max then
        Config.SPAWN_RADIUS_MAX = settings.spawn_radius_max
    end
    if settings.frame_skip then
        Config.FRAME_SKIP = settings.frame_skip
    end
    if settings.speed_multiplier ~= nil then
        local mult = math.floor(tonumber(settings.speed_multiplier) or 1)
        if mult < 1 then mult = 1 end
        if mult > 64 then mult = 64 end
        Config.SPEED_MULTIPLIER = mult
    end
    if settings.spawn_enemies ~= nil then
        Config.SPAWN_ENEMIES = settings.spawn_enemies
    end
    if settings.spawn_obstacles ~= nil then
        Config.SPAWN_OBSTACLES = settings.spawn_obstacles
    end
    if settings.obstacle_count ~= nil then
        Config.OBSTACLE_COUNT = settings.obstacle_count
    end
    if settings.obstacle_type ~= nil then
        Config.OBSTACLE_TYPE = settings.obstacle_type
    end
    if settings.obstacle_min_spacing ~= nil then
        Config.OBSTACLE_MIN_SPACING = settings.obstacle_min_spacing
    end
    if settings.max_episode_ticks then
        Config.MAX_EPISODE_TICKS = settings.max_episode_ticks
    end
    if settings.full_run ~= nil then
        Config.FULL_RUN = settings.full_run
    end
    if settings.human_mode ~= nil then
        local newHuman = settings.human_mode and true or false
        if newHuman ~= Config.HUMAN_MODE then
            Isaac.ConsoleOutput(
                "IsaacRL: human_mode = " .. tostring(newHuman) .. "\n")
        end
        Config.HUMAN_MODE = newHuman
    end
end

return GameControl
