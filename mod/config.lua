local Config = {}

Config.TCP_HOST = "127.0.0.1"
Config.TCP_PORT = tonumber(os.getenv("ISAAC_RL_PORT")) or 9999
Config.INSTANCE_ID = os.getenv("ISAAC_RL_INSTANCE") or "0"
Config.TCP_TIMEOUT = 0.5  -- send timeout; short enough to not block game loop, long enough for Python backpressure

-- Seed RNG per-instance so multi-worker training gets diverse spawns
math.randomseed(os.time() + tonumber(Config.INSTANCE_ID) * 1000)

-- How many game ticks between sending state / receiving actions
Config.FRAME_SKIP = 3

-- Speedhack: extra game updates per render frame (1-64). 1 = normal speed.
-- Note: past ~x8 training throughput is usually limited by the Python side,
-- not the game. Set via the configure command (speed_multiplier).
Config.SPEED_MULTIPLIER = 1

-- Free-running mode: never block the game waiting for Python.
-- States are sent non-blocking (dropped if Python is busy training),
-- actions are polled non-blockingly and the last received action is latched.
Config.FREE_RUNNING = true

-- Full-run mode: a cleared room emits room_cleared but does NOT end the
-- episode. Episodes only end on death, victory, explicit abandon, or the
-- global timeout. Set via the configure command (env.full_run).
Config.FULL_RUN = false

-- Room grid dimensions (standard room)
Config.GRID_WIDTH = 13
Config.GRID_HEIGHT = 7

-- Episode limits (0 = no limit; set via configure command)
Config.MAX_EPISODE_TICKS = 3000

-- Phase 1a defaults
Config.SPAWN_ENEMIES = true
Config.ENEMY_TYPE = 10      -- EntityType 10 = Gaper
Config.ENEMY_VARIANT = 0
Config.ENEMY_COUNT = 1
Config.ENEMY_COLLISION_DAMAGE = nil
Config.SPAWN_PICKUP_PENNY = false
Config.PICKUP_RANDOM_POSITION = false
Config.PICKUP_OFFSET_X = 180
Config.PICKUP_OFFSET_Y = 0
Config.PICKUP_RADIUS_MIN = 120
Config.PICKUP_RADIUS_MAX = 200
Config.TERMINAL_ON_PICKUP = false
Config.TERMINAL_PICKUP_COUNT = 0  -- 0 = disabled; N > 0 = end episode after N pickups
Config.RESPAWN_PICKUP = false     -- spawn a new pickup when one is collected
Config.RANDOM_SPAWN_POSITIONS = false
Config.SPAWN_RADIUS_MIN = 80
Config.SPAWN_RADIUS_MAX = 160
Config.DISABLE_SHOOTING = false
Config.SPAWN_OBSTACLES = false
Config.OBSTACLE_COUNT = 0
Config.OBSTACLE_TYPE = 4
Config.OBSTACLE_MIN_SPACING = 2

return Config
