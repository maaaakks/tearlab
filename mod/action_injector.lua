local ActionInjector = {}

-- Movement directions: 0=none, 1=up, 2=down, 3=left, 4=right,
-- 5=up-left, 6=up-right, 7=down-left, 8=down-right
-- Shooting directions: 0=none, 1=up, 2=down, 3=left, 4=right,
-- 5=up-left, 6=up-right, 7=down-left, 8=down-right
-- Interaction flags: 0=no, 1=trigger once per action decision
--   bomb     -> ACTION_BOMB    (E)
--   active   -> ACTION_ITEM    (Space)
--   pillcard -> ACTION_PILLCARD (Q)

local currentAction = {
    move = 0,
    shoot = 0,
    bomb = 0,
    active = 0,
    pillcard = 0
}

-- One-shot interaction state: pressed for ONE_SHOT_TICKS game ticks after
-- the action arrives, then cleared. Prevents a latched action from placing
-- a bomb every frame while Python is busy.
local ONE_SHOT_TICKS = 2
local oneShotTicksLeft = 0
local oneShotButtons = {}

local MOVE_MAP = {
    [0] = {},                           -- stand still
    [1] = {ACTION_UP = true},
    [2] = {ACTION_DOWN = true},
    [3] = {ACTION_LEFT = true},
    [4] = {ACTION_RIGHT = true},
    [5] = {ACTION_UP = true, ACTION_LEFT = true},
    [6] = {ACTION_UP = true, ACTION_RIGHT = true},
    [7] = {ACTION_DOWN = true, ACTION_LEFT = true},
    [8] = {ACTION_DOWN = true, ACTION_RIGHT = true},
}

local SHOOT_MAP = {
    [0] = {},                           -- don't shoot
    [1] = {ACTION_SHOOTUP = true},
    [2] = {ACTION_SHOOTDOWN = true},
    [3] = {ACTION_SHOOTLEFT = true},
    [4] = {ACTION_SHOOTRIGHT = true},
    [5] = {ACTION_SHOOTUP = true, ACTION_SHOOTLEFT = true},
    [6] = {ACTION_SHOOTUP = true, ACTION_SHOOTRIGHT = true},
    [7] = {ACTION_SHOOTDOWN = true, ACTION_SHOOTLEFT = true},
    [8] = {ACTION_SHOOTDOWN = true, ACTION_SHOOTRIGHT = true},
}

-- Map ButtonAction enum values to our string keys
local BUTTON_TO_KEY = {}
local KEY_TO_BUTTON = {}     -- reverse map for reading live human input
local INTERACTION_KEYS = {}  -- set of string keys treated as one-shot

function ActionInjector.init()
    BUTTON_TO_KEY[ButtonAction.ACTION_LEFT] = "ACTION_LEFT"
    BUTTON_TO_KEY[ButtonAction.ACTION_RIGHT] = "ACTION_RIGHT"
    BUTTON_TO_KEY[ButtonAction.ACTION_UP] = "ACTION_UP"
    BUTTON_TO_KEY[ButtonAction.ACTION_DOWN] = "ACTION_DOWN"
    BUTTON_TO_KEY[ButtonAction.ACTION_SHOOTLEFT] = "ACTION_SHOOTLEFT"
    BUTTON_TO_KEY[ButtonAction.ACTION_SHOOTRIGHT] = "ACTION_SHOOTRIGHT"
    BUTTON_TO_KEY[ButtonAction.ACTION_SHOOTUP] = "ACTION_SHOOTUP"
    BUTTON_TO_KEY[ButtonAction.ACTION_SHOOTDOWN] = "ACTION_SHOOTDOWN"
    for button, key in pairs(BUTTON_TO_KEY) do
        KEY_TO_BUTTON[key] = button
    end
    if ButtonAction.ACTION_BOMB ~= nil then
        BUTTON_TO_KEY[ButtonAction.ACTION_BOMB] = "ACTION_BOMB"
        INTERACTION_KEYS["ACTION_BOMB"] = true
    end
    if ButtonAction.ACTION_ITEM ~= nil then
        BUTTON_TO_KEY[ButtonAction.ACTION_ITEM] = "ACTION_ITEM"
        INTERACTION_KEYS["ACTION_ITEM"] = true
    end
    if ButtonAction.ACTION_PILLCARD ~= nil then
        BUTTON_TO_KEY[ButtonAction.ACTION_PILLCARD] = "ACTION_PILLCARD"
        INTERACTION_KEYS["ACTION_PILLCARD"] = true
    end
    local mapped = 0
    for _ in pairs(KEY_TO_BUTTON) do
        mapped = mapped + 1
    end
    Isaac.ConsoleOutput(
        "IsaacRL: ActionInjector.init mapped " .. tostring(mapped)
            .. " button actions\n")
end

-- DEBUG: set to true to disable shooting (for death testing)
ActionInjector.DISABLE_SHOOTING = false

function ActionInjector.setDisableShooting(disableShooting)
    ActionInjector.DISABLE_SHOOTING = disableShooting and true or false
    if ActionInjector.DISABLE_SHOOTING then
        currentAction.shoot = 0
    end
end

function ActionInjector.setAction(action)
    if action and action.move then
        currentAction.move = action.move
    end
    if action and action.shoot then
        if ActionInjector.DISABLE_SHOOTING then
            currentAction.shoot = 0
        else
            currentAction.shoot = action.shoot
        end
    end
end

-- Called ONCE per new action decision from Python (not on latched re-applies).
-- Interactions are edge-triggered: armed for one ONE_SHOT_TICKS window.
function ActionInjector.armInteractions(action)
    if not action then
        return
    end
    local armed = false
    if action.bomb == 1 then
        oneShotButtons["ACTION_BOMB"] = true
        armed = true
    end
    if action.active == 1 then
        oneShotButtons["ACTION_ITEM"] = true
        armed = true
    end
    if action.pillcard == 1 then
        oneShotButtons["ACTION_PILLCARD"] = true
        armed = true
    end
    if armed then
        oneShotTicksLeft = ONE_SHOT_TICKS
    end
end

-- Called once per game update; expires the one-shot window.
function ActionInjector.tick()
    if oneShotTicksLeft > 0 then
        oneShotTicksLeft = oneShotTicksLeft - 1
        if oneShotTicksLeft == 0 then
            oneShotButtons = {}
        end
    end
end

function ActionInjector.getAction()
    return currentAction
end

-- Encode four direction booleans into the 0-8 vocabulary
-- (0=none, 1=up, 2=down, 3=left, 4=right, 5=UL, 6=UR, 7=DL, 8=DR).
local function encodeDir(up, down, left, right)
    if up and left then return 5 end
    if up and right then return 6 end
    if down and left then return 7 end
    if down and right then return 8 end
    if up then return 1 end
    if down then return 2 end
    if left then return 3 end
    if right then return 4 end
    return 0
end

-- Read the live (human) input state and encode it into the same action
-- vocabulary Python uses, so demonstrations align with the agent's action
-- space. Sampled once per decision tick while HUMAN_MODE is active.
--
-- NOTE: must use Input.IsActionPressed here — IsButtonPressed expects
-- Keyboard enum scancodes, not ButtonAction values (it silently returns
-- false otherwise; see wofsauge/IsaacDocs Input.md).
function ActionInjector.readHumanAction()
    local function pressed(key)
        local button = KEY_TO_BUTTON[key]
        if button == nil then
            return false
        end
        return Input.IsActionPressed(button, 0)
    end

    local move = encodeDir(
        pressed("ACTION_UP"), pressed("ACTION_DOWN"),
        pressed("ACTION_LEFT"), pressed("ACTION_RIGHT")
    )
    local shoot = encodeDir(
        pressed("ACTION_SHOOTUP"), pressed("ACTION_SHOOTDOWN"),
        pressed("ACTION_SHOOTLEFT"), pressed("ACTION_SHOOTRIGHT")
    )

    -- Interaction dims: report "pressed now". A short human key tap can fall
    -- between two decision ticks; holding the key ~0.1s guarantees capture.
    local bomb = pressed("ACTION_BOMB") and 1 or 0
    local active = pressed("ACTION_ITEM") and 1 or 0
    local pillcard = false
    if ButtonAction.ACTION_PILLCARD ~= nil then
        pillcard = Input.IsActionPressed(ButtonAction.ACTION_PILLCARD, 0)
    end

    return {
        move = move,
        shoot = ActionInjector.DISABLE_SHOOTING and 0 or shoot,
        bomb = bomb,
        active = active,
        pillcard = pillcard and 1 or 0,
    }
end

function ActionInjector.reset()
    currentAction.move = 0
    currentAction.shoot = 0
    currentAction.bomb = 0
    currentAction.active = 0
    currentAction.pillcard = 0
    oneShotTicksLeft = 0
    oneShotButtons = {}
end

function ActionInjector.onInputAction(_, entity, inputHook, buttonAction)
    local key = BUTTON_TO_KEY[buttonAction]
    if not key then
        return nil
    end

    local isPressed
    if INTERACTION_KEYS[key] then
        isPressed = oneShotTicksLeft > 0 and oneShotButtons[key] == true
    else
        local moveButtons = MOVE_MAP[currentAction.move] or {}
        local shootButtons = SHOOT_MAP[currentAction.shoot] or {}
        isPressed = moveButtons[key] or shootButtons[key] or false
    end

    if inputHook == InputHook.GET_ACTION_VALUE then
        if isPressed then
            return 1.0
        else
            return 0.0
        end
    elseif inputHook == InputHook.IS_ACTION_PRESSED then
        return isPressed
    elseif inputHook == InputHook.IS_ACTION_TRIGGERED then
        return isPressed
    end

    return nil
end

return ActionInjector
