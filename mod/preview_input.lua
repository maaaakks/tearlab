-- SPDX-License-Identifier: CC0-1.0
-- Standalone frozen-preview adapter; no legacy training or human-demo logic.
local Adapter = {}
local enabled, command, buttons = false, {move=0,shoot=0}, {}
local moves = {{0,0},{0,-1},{0,1},{-1,0},{1,0},{-1,-1},{1,-1},{-1,1},{1,1}}
local shots = {{0,0},{0,-1},{0,1},{-1,0},{1,0}}
function Adapter.init()
    local specs = {
        {"ACTION_LEFT","move",1,-1}, {"ACTION_RIGHT","move",1,1},
        {"ACTION_UP","move",2,-1}, {"ACTION_DOWN","move",2,1},
        {"ACTION_SHOOTLEFT","shoot",1,-1}, {"ACTION_SHOOTRIGHT","shoot",1,1},
        {"ACTION_SHOOTUP","shoot",2,-1}, {"ACTION_SHOOTDOWN","shoot",2,1},
        {"ACTION_BOMB","blocked"}, {"ACTION_ITEM","blocked"}, {"ACTION_PILLCARD","blocked"},
    }
    buttons = {}
    for _, spec in ipairs(specs) do
        local id = ButtonAction[spec[1]]
        if id ~= nil then buttons[id] = spec end
    end
end
function Adapter.reset() command = {move=0,shoot=0} end
function Adapter.setEnabled(value)
    enabled = value == true
    if not enabled then Adapter.reset() end
end
function Adapter.isEnabled() return enabled end
function Adapter.setAction(action)
    command = {move=tonumber(action.move) or 0, shoot=tonumber(action.shoot) or 0}
end
function Adapter.tick() end
function Adapter.onInputAction(_, entity, hook, button)
    if not enabled then return nil end
    local spec = buttons[button]
    if not spec then return nil end
    local pressed = false
    if spec[2] ~= "blocked" then
        local vectors = spec[2] == "move" and moves or shots
        local vector = vectors[command[spec[2]]+1] or vectors[1]
        pressed = vector[spec[3]] == spec[4]
    end
    if hook == InputHook.GET_ACTION_VALUE then return pressed and 1.0 or 0.0 end
    if hook == InputHook.IS_ACTION_PRESSED or hook == InputHook.IS_ACTION_TRIGGERED then return pressed end
    return nil
end
return Adapter
