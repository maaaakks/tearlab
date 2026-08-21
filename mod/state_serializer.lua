local Config = require("config")

local StateSerializer = {}

-- Call fn, falling back to default if the API is missing or errors.
-- Keeps the serializer compatible across game versions.
local function safe(fn, default)
    local ok, result = pcall(fn)
    if ok then
        return result
    end
    return default
end

-- Grid entity type constants
local GRID_WALL = 15
local GRID_ROCK = 2
local GRID_ROCKB = 3
local GRID_ROCKT = 4
local GRID_ROCK_ALT = 22
local GRID_PIT = 7
local GRID_SPIKES = 8
local GRID_SPIKES_ONOFF = 9
local GRID_POOP = 14
local GRID_TNT = 12
local GRID_FIREPLACE = 13
local GRID_SPIDERWEB = 10

-- Channel indices for the grid observation
StateSerializer.CHANNEL_WALLS = 1
StateSerializer.CHANNEL_OBSTACLES = 2
StateSerializer.CHANNEL_PITS = 3
StateSerializer.CHANNEL_PLAYER = 4
StateSerializer.CHANNEL_ENEMIES = 5
StateSerializer.CHANNEL_PROJECTILES = 6
StateSerializer.CHANNEL_PICKUPS = 7
StateSerializer.CHANNEL_DOORS = 8
-- Projectile velocity direction, encoded as 0.5 + 0.5 * (v / PROJ_VEL_NORM)
-- so 0.5 = stationary/no velocity, 0.0 = full speed one way, 1.0 = the other.
StateSerializer.CHANNEL_PROJ_VX = 9
StateSerializer.CHANNEL_PROJ_VY = 10
StateSerializer.NUM_CHANNELS = 10

-- Projectile speeds in Isaac are roughly 0-16 px/tick; normalize by ~8 so
-- typical projectiles map to a good chunk of the 0-1 range.
local PROJ_VEL_NORM = 8.0

local function createGrid()
    local grid = {}
    for c = 1, StateSerializer.NUM_CHANNELS do
        grid[c] = {}
        for y = 1, Config.GRID_HEIGHT do
            grid[c][y] = {}
            for x = 1, Config.GRID_WIDTH do
                grid[c][y][x] = 0
            end
        end
    end
    return grid
end

local function worldToGrid(room, pos)
    local idx = room:GetClampedGridIndex(pos)
    local w = room:GetGridWidth()
    local x = (idx % w) + 1  -- 1-indexed
    local y = math.floor(idx / w) + 1
    -- Clamp to valid range
    x = math.max(1, math.min(x, Config.GRID_WIDTH))
    y = math.max(1, math.min(y, Config.GRID_HEIGHT))
    return x, y
end

local function isObstacle(gridType)
    return gridType == GRID_ROCK
        or gridType == GRID_ROCKB
        or gridType == GRID_ROCKT
        or gridType == GRID_ROCK_ALT
        or gridType == GRID_POOP
        or gridType == GRID_TNT
        or gridType == GRID_FIREPLACE
        or gridType == GRID_SPIDERWEB
end

function StateSerializer.serialize(game)
    local room = game:GetRoom()
    local level = game:GetLevel()
    local player = Isaac.GetPlayer(0)

    local roomTopLeft = room:GetTopLeftPos()
    local roomBottomRight = room:GetBottomRightPos()
    local roomWidth = math.max(1.0, roomBottomRight.X - roomTopLeft.X)
    local roomHeight = math.max(1.0, roomBottomRight.Y - roomTopLeft.Y)

    local function normalizePos(pos)
        local x = (pos.X - roomTopLeft.X) / roomWidth
        local y = (pos.Y - roomTopLeft.Y) / roomHeight
        x = math.max(0.0, math.min(1.0, x))
        y = math.max(0.0, math.min(1.0, y))
        return x, y
    end

    local function normalizeDelta(dx, dy)
        local x = dx / roomWidth
        local y = dy / roomHeight
        x = math.max(-1.0, math.min(1.0, x))
        y = math.max(-1.0, math.min(1.0, y))
        return x, y
    end

    local grid = createGrid()

    -- Encode grid entities (walls, obstacles, pits)
    local gridWidth = room:GetGridWidth()
    local gridSize = room:GetGridSize()
    local specialRocks = {}  -- tinted rocks / marked skulls (identity matters)
    for idx = 0, gridSize - 1 do
        local gx = (idx % gridWidth) + 1
        local gy = math.floor(idx / gridWidth) + 1
        if gx >= 1 and gx <= Config.GRID_WIDTH and gy >= 1 and gy <= Config.GRID_HEIGHT then
            local gridEntity = room:GetGridEntity(idx)
            if gridEntity then
                local gType = gridEntity:GetType()
                if gType == GRID_WALL then
                    grid[StateSerializer.CHANNEL_WALLS][gy][gx] = 1
                elseif gType == GRID_PIT then
                    grid[StateSerializer.CHANNEL_PITS][gy][gx] = 1
                elseif gType == GRID_SPIKES or gType == GRID_SPIKES_ONOFF then
                    grid[StateSerializer.CHANNEL_OBSTACLES][gy][gx] = 1
                elseif isObstacle(gType) then
                    grid[StateSerializer.CHANNEL_OBSTACLES][gy][gx] = 1
                    -- Rock variants: 1 = marked skull, 2 = tinted rock.
                    -- Exposed as identity-bearing objects for the planner.
                    if gType == GRID_ROCK or gType == GRID_ROCK_ALT then
                        local variant = safe(function() return gridEntity.Variant end, 0)
                        if variant == 1 or variant == 2 then
                            local wpos = room:GetGridPosition(idx)
                            table.insert(specialRocks, {
                                kind = variant == 1 and "marked_skull" or "tinted_rock",
                                variant = variant,
                                position = {wpos.X, wpos.Y},
                                destroyed = false,
                            })
                        end
                    end
                end
            end
        end
    end

    -- Player position on grid
    local px, py = worldToGrid(room, player.Position)
    grid[StateSerializer.CHANNEL_PLAYER][py][px] = 1

    -- Enemies, projectiles, pickups
    local enemies = {}
    local pickups = {}       -- structured records (identity + price)
    local pedestalItems = {} -- Type 5 Variant 100 collectibles
    local nearestPickupDist = nil
    local nearestPickupDx = 0.0
    local nearestPickupDy = 0.0
    local nearestEnemyDist = nil
    local nearestEnemyDx = 0.0
    local nearestEnemyDy = 0.0
    local nearestProjectileDist = nil
    local nearestProjectileDx = 0.0
    local nearestProjectileDy = 0.0
    local entities = Isaac.GetRoomEntities()
    for i = 1, #entities do
        local ent = entities[i]
        local ex, ey = worldToGrid(room, ent.Position)
        local dx = ent.Position.X - player.Position.X
        local dy = ent.Position.Y - player.Position.Y
        local distSq = dx * dx + dy * dy

        if ent:IsActiveEnemy(false) and not ent:IsDead() then
            -- Enemy channel: normalized HP (0-1)
            local maxHp = ent.MaxHitPoints
            local hp = ent.HitPoints
            local normalized = 1.0
            if maxHp > 0 then
                normalized = hp / maxHp
            end
            grid[StateSerializer.CHANNEL_ENEMIES][ey][ex] = math.max(
                grid[StateSerializer.CHANNEL_ENEMIES][ey][ex],
                normalized
            )
            table.insert(enemies, {
                init_seed = ent.InitSeed,
                type = ent.Type,
                variant = ent.Variant,
                hp = hp,
                max_hp = maxHp,
                position = {ent.Position.X, ent.Position.Y},
                velocity = {ent.Velocity.X, ent.Velocity.Y}
            })
            if nearestEnemyDist == nil or distSq < nearestEnemyDist then
                nearestEnemyDist = distSq
                nearestEnemyDx = dx
                nearestEnemyDy = dy
            end

        elseif ent.Type == 9 then
            -- Enemy projectile: position + velocity direction channels
            grid[StateSerializer.CHANNEL_PROJECTILES][ey][ex] = 1
            local vx = math.max(-1.0, math.min(1.0, ent.Velocity.X / PROJ_VEL_NORM))
            local vy = math.max(-1.0, math.min(1.0, ent.Velocity.Y / PROJ_VEL_NORM))
            local cellVx = 0.5 + 0.5 * vx
            local cellVy = 0.5 + 0.5 * vy
            grid[StateSerializer.CHANNEL_PROJ_VX][ey][ex] = cellVx
            grid[StateSerializer.CHANNEL_PROJ_VY][ey][ex] = cellVy
            if nearestProjectileDist == nil or distSq < nearestProjectileDist then
                nearestProjectileDist = distSq
                nearestProjectileDx = dx
                nearestProjectileDy = dy
            end

        elseif ent.Type == 5 and ent.Variant == 100 then
            -- Pedestal item: identity + price (0 = free, negative = heart cost)
            local pxn, pyn = normalizePos(ent.Position)
            table.insert(pedestalItems, {
                item_id = ent.SubType,
                price = safe(function() return ent.Price end, 0),
                position = {ent.Position.X, ent.Position.Y},
                pos_x = pxn,
                pos_y = pyn,
            })

        elseif ent.Type == 5 then
            -- Pickup (coin/bomb/key/pill/card/...): structured record
            local pxn, pyn = normalizePos(ent.Position)
            table.insert(pickups, {
                variant = ent.Variant,
                subtype = ent.SubType,
                price = safe(function() return ent.Price end, 0),
                position = {ent.Position.X, ent.Position.Y},
                pos_x = pxn,
                pos_y = pyn,
            })
            grid[StateSerializer.CHANNEL_PICKUPS][ey][ex] = 1
            if nearestPickupDist == nil or distSq < nearestPickupDist then
                nearestPickupDist = distSq
                nearestPickupDx = dx
                nearestPickupDy = dy
            end
        end
    end

    local playerPosX, playerPosY = normalizePos(player.Position)
    local nearestPickupDxNorm, nearestPickupDyNorm = normalizeDelta(nearestPickupDx, nearestPickupDy)
    local nearestEnemyDxNorm, nearestEnemyDyNorm = normalizeDelta(nearestEnemyDx, nearestEnemyDy)
    local nearestProjectileDxNorm, nearestProjectileDyNorm = normalizeDelta(nearestProjectileDx, nearestProjectileDy)

    -- Doors (with destination info for the room-graph memory)
    local doors = {}
    for slot = 0, 7 do
        local door = room:GetDoor(slot)
        if door then
            local dx, dy = worldToGrid(room, door.Position)
            if dx >= 1 and dx <= Config.GRID_WIDTH and dy >= 1 and dy <= Config.GRID_HEIGHT then
                local doorVal = 0.5  -- open
                if door:IsLocked() then
                    doorVal = 0.25
                end
                if room:IsClear() then
                    doorVal = 1.0
                end
                grid[StateSerializer.CHANNEL_DOORS][dy][dx] = doorVal

                table.insert(doors, {
                    slot = slot,
                    locked = door:IsLocked(),
                    open = safe(function() return door:IsOpen() end, not door:IsLocked()),
                    target_room_index = safe(function() return door.TargetRoomIndex end, -1),
                    position = {door.Position.X, door.Position.Y},
                })
            end
        end
    end

    -- Player state vector
    local playerState = {
        hp_red = player:GetHearts(),
        hp_soul = player:GetSoulHearts(),
        hp_black = player:GetBlackHearts(),
        hp_bone = safe(function() return player:GetBoneHearts() end, 0),
        hp_golden = safe(function() return player:GetGoldenHearts() end, 0),
        speed = player.MoveSpeed,
        damage = player.Damage,
        range = player.TearRange,
        fire_rate = player.MaxFireDelay,
        shot_speed = player.ShotSpeed,
        luck = player.Luck,
        num_bombs = player:GetNumBombs(),
        num_keys = player:GetNumKeys(),
        num_coins = player:GetNumCoins(),
        has_active_item = player:GetActiveItem() ~= 0,
        active_item_id = safe(function() return player:GetActiveItem() end, 0),
        active_charge = player:GetActiveCharge(),
        card_slot = safe(function() return player:GetCard(0) end, 0),
        pill_slot = safe(function() return player:GetPill(0) end, 0),
        trinket_id = safe(function() return player:GetTrinket(0) end, 0),
        pos_x = playerPosX,
        pos_y = playerPosY,
        nearest_pickup_dx = nearestPickupDxNorm,
        nearest_pickup_dy = nearestPickupDyNorm,
        nearest_enemy_dx = nearestEnemyDxNorm,
        nearest_enemy_dy = nearestEnemyDyNorm,
        nearest_projectile_dx = nearestProjectileDxNorm,
        nearest_projectile_dy = nearestProjectileDyNorm,
        position = {player.Position.X, player.Position.Y}
    }

    -- Run-level info: floor identity, room identity, deal history.
    -- All additive; the planner and run memory consume these.
    local roomDesc = safe(function() return level:GetCurrentRoomDesc() end, nil)
    local run = {
        stage = level:GetStage(),
        stage_type = level:GetStageType(),
        curse = safe(function() return level:GetCurses() end, 0),
        room_type = room:GetType(),
        room_shape = room:GetShape(),
        room_list_index = roomDesc and roomDesc.ListIndex or -1,
        room_grid_index = safe(function() return level:GetCurrentRoomIndex() end, -1),
        room_clear = room:IsClear(),
        devil_deal_taken = safe(
            function() return game:GetStateFlag(GameStateFlag.STATE_DEVIL_TAKEN) end, false),
        angel_room_taken = safe(
            function()
                return game:GetStateFlag(GameStateFlag.STATE_ANGEL_TAKEN)
                    or level:GetStateFlag(LevelStateFlag.STATE_ANGEL_TAKEN)
            end, false),
    }

    local state = {
        grid = grid,
        player = playerState,
        enemies = enemies,
        pickups = pickups,
        pedestal_items = pedestalItems,
        special_rocks = specialRocks,
        doors = doors,
        run = run,
        room_cleared = room:IsClear(),
        player_dead = player:IsDead(),
        tick = game:GetFrameCount(),
        enemy_count = #enemies
    }

    return state
end

return StateSerializer
