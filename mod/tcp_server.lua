local json = require("json")

local TcpServer = {}
TcpServer.__index = TcpServer

local socket = nil
local HAS_SOCKET = false

-- Try to load luasocket
local ok, luasocket = pcall(require, "socket")
if ok then
    socket = luasocket
    HAS_SOCKET = true
    Isaac.ConsoleOutput("IsaacRL: luasocket loaded\n")
else
    Isaac.ConsoleOutput("IsaacRL: luasocket not available, falling back to file IPC\n")
end

function TcpServer.new(host, port, timeout)
    local self = setmetatable({}, TcpServer)
    self.host = host
    self.port = port
    self.timeout = timeout or 0.001
    self.server = nil
    self.client = nil
    self.connected = false
    self.use_file_ipc = not HAS_SOCKET
    self.ipc_dir = nil
    return self
end

function TcpServer:start()
    if self.use_file_ipc then
        return self:startFileIPC()
    end
    return self:startSocket()
end

function TcpServer:startSocket()
    self.server = socket.tcp()
    self.server:setoption("reuseaddr", true)
    local ok, err = self.server:bind(self.host, self.port)
    if not ok then
        Isaac.ConsoleOutput("IsaacRL: Failed to bind: " .. tostring(err) .. "\n")
        return false
    end
    self.server:listen(1)
    self.server:settimeout(0.001)  -- non-blocking accept so game doesn't freeze waiting for connection
    Isaac.ConsoleOutput("IsaacRL: TCP server listening on " .. self.host .. ":" .. self.port .. "\n")
    return true
end

function TcpServer:startFileIPC()
    -- File-based IPC fallback: use temp files for communication
    local tmpdir = os.getenv("TMPDIR") or os.getenv("TEMP") or "/tmp"
    self.ipc_dir = tmpdir
    self.state_file = tmpdir .. "/isaacrl_state.json"
    self.action_file = tmpdir .. "/isaacrl_action.json"
    self.ready_file = tmpdir .. "/isaacrl_ready"
    -- Signal that we're ready
    local f = io.open(self.ready_file, "w")
    if f then
        f:write("ready")
        f:close()
    end
    Isaac.ConsoleOutput("IsaacRL: File IPC started in " .. tmpdir .. "\n")
    self.connected = true
    return true
end

function TcpServer:acceptClient()
    if self.use_file_ipc then
        return true
    end
    if self.connected then
        return true
    end
    if not self.server then
        return false
    end
    local client, err = self.server:accept()
    if client then
        client:settimeout(self.timeout)
        self.client = client
        self.connected = true
        Isaac.ConsoleOutput("IsaacRL: Python client connected\n")
        return true
    end
    return false
end

function TcpServer:sendState(stateTable)
    local data = json.encode(stateTable)
    if self.use_file_ipc then
        local f = io.open(self.state_file, "w")
        if f then
            f:write(data)
            f:close()
            return true
        end
        return false
    end
    if not self.client then
        return false
    end
    -- Use infinite timeout for send — in lock-step mode the buffer should
    -- always have room because Python reads before sending the next action.
    self.client:settimeout(nil)
    local ok, err = self.client:send(data .. "\n")
    self.client:settimeout(self.timeout)  -- restore
    if not ok then
        Isaac.ConsoleOutput("IsaacRL: Send error (" .. tostring(err) .. "), disconnecting\n")
        self:handleDisconnect()
        return false
    end
    return true
end

--- Non-blocking state send for free-running mode.
-- Never blocks the game: if the socket buffer is full (Python busy training),
-- the frame is silently dropped. Only a closed connection triggers disconnect.
function TcpServer:sendStateNonBlocking(stateTable)
    if self.use_file_ipc then
        return self:sendState(stateTable)
    end
    if not self.client then
        return false
    end
    local data = json.encode(stateTable)
    self.client:settimeout(0)
    local ok, err = self.client:send(data .. "\n")
    self.client:settimeout(self.timeout)
    if ok then
        return true
    end
    if err == "closed" then
        Isaac.ConsoleOutput("IsaacRL: Send error (" .. tostring(err) .. "), disconnecting\n")
        self:handleDisconnect()
    end
    -- "timeout" / "wantwrite": Python is backpressured; drop this frame
    return false
end

function TcpServer:receiveAction()
    if self.use_file_ipc then
        return self:receiveActionFile()
    end
    return self:receiveActionSocket()
end

function TcpServer:receiveActionSocket()
    if not self.client then
        return nil
    end
    local data, err = self.client:receive("*l")
    if data then
        local ok, action = pcall(json.decode, data)
        if ok then
            return action
        else
            Isaac.ConsoleOutput("IsaacRL: JSON decode error\n")
            return nil
        end
    elseif err == "closed" then
        self:handleDisconnect()
    end
    return nil
end

--- Block until Python sends a message (infinite timeout).
-- Used after sendState to lock-step Lua with Python's action loop.
function TcpServer:waitForAction()
    if self.use_file_ipc or not self.client then
        return nil
    end
    self.client:settimeout(nil)  -- block indefinitely
    local data, err = self.client:receive("*l")
    self.client:settimeout(self.timeout)  -- restore normal timeout
    if data then
        local ok, msg = pcall(json.decode, data)
        if ok then
            return msg
        else
            Isaac.ConsoleOutput("IsaacRL: JSON decode error in waitForAction\n")
        end
    elseif err == "closed" then
        self:handleDisconnect()
    end
    return nil
end

-- Non-blocking poll: returns one buffered message or nil
function TcpServer:pollAction()
    if self.use_file_ipc then
        return self:receiveActionFile()
    end
    if not self.client then
        return nil
    end
    self.client:settimeout(0)
    local data, err = self.client:receive("*l")
    self.client:settimeout(self.timeout)
    if data then
        local ok, msg = pcall(json.decode, data)
        if ok then
            return msg
        else
            Isaac.ConsoleOutput("IsaacRL: JSON decode error\n")
            return nil
        end
    elseif err == "closed" then
        self:handleDisconnect()
    end
    return nil
end

function TcpServer:receiveActionFile()
    local f = io.open(self.action_file, "r")
    if not f then
        return nil
    end
    local data = f:read("*a")
    f:close()
    os.remove(self.action_file)
    if data and #data > 0 then
        local ok, action = pcall(json.decode, data)
        if ok then
            return action
        end
    end
    return nil
end

function TcpServer:handleDisconnect()
    Isaac.ConsoleOutput("IsaacRL: Client disconnected\n")
    if self.client then
        self.client:close()
        self.client = nil
    end
    self.connected = false
end

function TcpServer:stop()
    if self.client then
        self.client:close()
    end
    if self.server then
        self.server:close()
    end
    self.connected = false
end

return TcpServer
