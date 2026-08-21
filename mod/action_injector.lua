local ActionInjector = {}

-- Movement directions: 0=none, 1=up, 2=down, 3=left, 4=right,
-- 5=up-left, 6=up-right, 7=down-left, 8=down-right
-- Shooting directions: 0=none, 1=up, 2=down, 3=left, 4=right,
-- 5=up-left, 6=up-right, 7=down-left, 8=down-right

local currentAction = {
    move = 0,
    shoot = 0
}

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

function ActionInjector.init()
    BUTTON_TO_KEY[ButtonAction.ACTION_LEFT] = "ACTION_LEFT"
    BUTTON_TO_KEY[ButtonAction.ACTION_RIGHT] = "ACTION_RIGHT"
    BUTTON_TO_KEY[ButtonAction.ACTION_UP] = "ACTION_UP"
    BUTTON_TO_KEY[ButtonAction.ACTION_DOWN] = "ACTION_DOWN"
    BUTTON_TO_KEY[ButtonAction.ACTION_SHOOTLEFT] = "ACTION_SHOOTLEFT"
    BUTTON_TO_KEY[ButtonAction.ACTION_SHOOTRIGHT] = "ACTION_SHOOTRIGHT"
    BUTTON_TO_KEY[ButtonAction.ACTION_SHOOTUP] = "ACTION_SHOOTUP"
    BUTTON_TO_KEY[ButtonAction.ACTION_SHOOTDOWN] = "ACTION_SHOOTDOWN"
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

function ActionInjector.getAction()
    return currentAction
end

function ActionInjector.reset()
    currentAction.move = 0
    currentAction.shoot = 0
end

function ActionInjector.onInputAction(_, entity, inputHook, buttonAction)
    local key = BUTTON_TO_KEY[buttonAction]
    if not key then
        return nil
    end

    local moveButtons = MOVE_MAP[currentAction.move] or {}
    local shootButtons = SHOOT_MAP[currentAction.shoot] or {}

    local isPressed = moveButtons[key] or shootButtons[key] or false

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
