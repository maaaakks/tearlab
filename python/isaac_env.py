import logging
import socket
import threading
import time
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import Config
from game_state import GameState
from network_client import NetworkClient
from reward import RewardShaper
from run_memory import NUM_MAP_CHANNELS, MAP_HEIGHT, MAP_WIDTH, RunMemory

log = logging.getLogger("IsaacEnv")


class IsaacEnv(gym.Env):
    """Gymnasium environment wrapper for The Binding of Isaac via TCP.

    Protocol: Lua owns the game clock and episode lifecycle.
    - Lua streams observations with episode_id, terminal, terminal_reason
    - Lua restarts immediately on terminal, increments episode_id
    - Python sends actions (fire-and-forget), syncs on episode_id for reset
    """

    metadata = {"render_modes": []}
    PLAYER_FEATURE_SCALES = np.array(
        [
            24.0, 24.0, 24.0,  # red/soul/black HP in half-hearts
            3.0, 20.0, 1000.0, 30.0, 5.0, 20.0,  # combat stats
            99.0, 99.0, 99.0,  # bombs, keys, coins
            1.0, 12.0,  # active item flag and charge
            1.0, 1.0,  # normalized player position
            1.0, 1.0,  # nearest pickup delta
            1.0, 1.0,  # nearest enemy delta
            1.0, 1.0,  # nearest projectile delta
        ],
        dtype=np.float32,
    )

    def __init__(self, config: Config | None = None):
        super().__init__()
        self.config = config or Config()
        self.env_cfg = self.config.env
        self.reward_shaper = RewardShaper(self.config.reward)

        # Action space: MultiDiscrete([9, 9, 2, 2, 2])
        #   dim 0: movement (8 directions + none)
        #   dim 1: shooting (8 directions + none, incl. diagonal tears)
        #   dim 2: place bomb (0=no, 1=place)
        #   dim 3: use active item (0=no, 1=use)
        #   dim 4: use card/pill (0=no, 1=use)
        # Interaction dims are masked off unless phase.enable_interactions.
        self.action_space = spaces.MultiDiscrete([9, 9, 2, 2, 2])

        # Frame stacking: stack the last N grids along the channel axis so the
        # network can infer velocities from consecutive frames.
        self.frame_stack = max(1, int(self.env_cfg.frame_stack))
        self._grid_stack: deque[np.ndarray] = deque(maxlen=self.frame_stack)

        # Run memory: persistent room-graph map summary appended to obs.
        self.enable_map_memory = bool(getattr(self.env_cfg, "enable_map_memory", False))
        self.run_memory = RunMemory()

        # Observation space: Dict with grid (CNN input) and player (vector input)
        obs_spaces = {
            "grid": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(
                    self.env_cfg.grid_channels * self.frame_stack,
                    self.env_cfg.grid_height,
                    self.env_cfg.grid_width,
                ),
                dtype=np.float32,
            ),
            "player": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.env_cfg.player_features,),
                dtype=np.float32,
            ),
        }
        if self.enable_map_memory:
            obs_spaces["map"] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(NUM_MAP_CHANNELS, MAP_HEIGHT, MAP_WIDTH),
                dtype=np.float32,
            )
        self.observation_space = spaces.Dict(obs_spaces)

        self._client = NetworkClient(
            host=self.env_cfg.host,
            port=self.env_cfg.port,
            timeout=self.env_cfg.action_timeout,
            on_connect=self._on_client_connect,
        )
        self.step_count = 0
        self.last_state = None
        self._episode_num = 0
        self._last_episode_id = 0  # tracks Lua's episode_id

        # Throughput tracking
        self._total_steps = 0
        self._start_time = None
        self._throughput_interval = 1000

        # Episode tracking
        self._ep_reward = 0.0
        self._ep_kills = 0
        self._ep_damage_taken = 0
        self._ep_reward_components: dict[str, float] = {}
        self._last_terminal_reason = None

        # Small thread-safe snapshot forwarded to the in-game training HUD.
        # The live trainer updates it from its optimizer thread while the
        # collector keeps stepping the environment.
        self._training_metrics: dict[str, float | int | str] = {}
        self._training_metrics_lock = threading.Lock()
        self._hud_game_ticks_per_sec = 0.0

        # Auto-configure: send game settings on first reset()
        self._game_settings = self.config.to_game_settings()
        self._configured = False

        # Speed diagnostics
        self._last_episode_tick = 0
        self._ep_frames_dropped = 0
        self._ep_step_latencies = []
        self._ep_receive_times = []  # wall-clock time of each received state

    def _on_client_connect(self, client: NetworkClient) -> None:
        """Replay current game settings after every connect/reconnect."""
        client.send({"command": "configure", "settings": self._game_settings})
        self._configured = True

    def _receive_state(self) -> GameState:
        return GameState.from_dict(self._client.receive())

    def update_training_metrics(self, **metrics) -> None:
        """Merge a compact metric snapshot for the Lua HUD."""
        with self._training_metrics_lock:
            self._training_metrics.update(metrics)

    def _build_debug_info(self) -> dict:
        """Build the JSON-safe payload rendered by the Lua HUD."""
        rc = self.reward_shaper.reward_components
        debug_info = {
            "episode": self._episode_num,
            "episode_reward": round(self._ep_reward, 2),
            "episode_step": self.step_count,
            "step_reward": round(sum(rc.values()), 3),
            "damage_dealt": round(rc.get("damage_dealt", 0), 2),
            "kills": round(rc.get("kills", 0), 2),
            "damage_taken": round(rc.get("damage_taken", 0), 2),
            "pickup_approach": round(rc.get("pickup_approach", 0), 3),
            "wall_collision": round(rc.get("wall_collision", 0), 3),
            "pickup_collected": round(rc.get("pickup_collected", 0), 1),
            "time_reward": round(rc.get("time", 0), 3),
            "game_tps": round(self._hud_game_ticks_per_sec, 1),
            "frames_dropped": self._ep_frames_dropped,
        }
        with self._training_metrics_lock:
            debug_info.update(self._training_metrics)
        return debug_info

    def _push_grid(self, grid: np.ndarray) -> np.ndarray:
        """Append a grid to the frame stack and return the stacked observation."""
        self._grid_stack.append(grid)
        return np.concatenate(list(self._grid_stack), axis=0)

    def _state_to_obs(self, state: GameState) -> dict:
        grid = np.array(state.grid, dtype=np.float32)
        if grid.shape != (self.env_cfg.grid_channels, self.env_cfg.grid_height, self.env_cfg.grid_width):
            grid = np.zeros(
                (self.env_cfg.grid_channels, self.env_cfg.grid_height, self.env_cfg.grid_width),
                dtype=np.float32,
            )
        if len(self._grid_stack) == 0:
            # New episode: pre-fill the stack so all frames match
            for _ in range(self.frame_stack):
                self._grid_stack.append(grid)

        p = state.player
        player_features = [
            p.hp_red,
            p.hp_soul,
            p.hp_black,
            p.speed,
            p.damage,
            p.range,
            p.fire_rate,
            p.shot_speed,
            p.luck,
            p.num_bombs,
            p.num_keys,
            p.num_coins,
            1.0 if p.has_active_item else 0.0,
            p.active_charge,
        ]

        if self.env_cfg.mask_distance_vectors:
            player_features.extend([0.0] * 8)
        else:
            player_features.extend([
                p.pos_x,
                p.pos_y,
                p.nearest_pickup_dx,
                p.nearest_pickup_dy,
                p.nearest_enemy_dx,
                p.nearest_enemy_dy,
                p.nearest_projectile_dx,
                p.nearest_projectile_dy,
            ])

        player = np.array(player_features, dtype=np.float32)
        if player.shape != (self.env_cfg.player_features,):
            raise ValueError(
                f"Player feature shape mismatch: got {player.shape[0]}, "
                f"expected {self.env_cfg.player_features}"
            )
        if self.env_cfg.normalize_player_features:
            if player.shape != self.PLAYER_FEATURE_SCALES.shape:
                raise ValueError(
                    "Player normalization scale mismatch: "
                    f"features={player.shape}, scales={self.PLAYER_FEATURE_SCALES.shape}"
                )
            player = np.clip(
                player / self.PLAYER_FEATURE_SCALES,
                -2.0,
                2.0,
            ).astype(np.float32)

        obs: dict = {"grid": self._push_grid(grid), "player": player}
        if self.enable_map_memory:
            obs["map"] = self.run_memory.update(state)
        return obs

    def _wait_for_new_episode(self, timeout=120.0):
        """Wait for first non-terminal observation of a new episode_id."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                state = self._receive_state()
                if state.episode_id > self._last_episode_id and not state.terminal:
                    return state
            except (TimeoutError, OSError):
                continue
        raise ConnectionError(
            f"Timed out waiting for new episode (last_id={self._last_episode_id})"
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._client.connect()

        # Log previous episode summary
        if self._episode_num > 0:
            reason = self._last_terminal_reason or "unknown"
            avg_latency = (
                sum(self._ep_step_latencies) / len(self._ep_step_latencies) * 1000
                if self._ep_step_latencies else 0.0
            )
            ticks_per_sec = 0.0
            if len(self._ep_receive_times) >= 2:
                duration = self._ep_receive_times[-1] - self._ep_receive_times[0]
                if duration > 0:
                    ticks_per_sec = (len(self._ep_receive_times) - 1) / duration
            log.info(
                "EP %d ended (%s) | steps=%d reward=%.1f kills=%d dmg_taken=%d "
                "frames_dropped=%d avg_latency=%.1fms ticks/s=%.1f",
                self._episode_num, reason, self.step_count,
                self._ep_reward, self._ep_kills, self._ep_damage_taken,
                self._ep_frames_dropped, avg_latency, ticks_per_sec,
            )

        self._episode_num += 1
        self.step_count = 0
        self._ep_reward = 0.0
        self._ep_kills = 0
        self._ep_damage_taken = 0
        self._ep_reward_components = {}
        self._last_terminal_reason = None
        self._last_episode_tick = 0
        self._ep_frames_dropped = 0
        self._ep_step_latencies = []
        self._ep_receive_times = []
        self.reward_shaper.reset()
        self._grid_stack.clear()
        if self.enable_map_memory:
            # New episode = new run: wipe floor memory
            self.run_memory.reset()

        if self._last_episode_id == 0:
            # First episode in this Python process: force game back to active play
            # and request a fresh episode so stale paused state cannot leak across runs.
            self._client.send({"command": "resume"})
            self._client.send({"command": "reset"})
            log.debug("Initial reset sent, waiting for first episode...")

        # Wait for observation with a new episode_id (Lua auto-restarts)
        state = self._wait_for_new_episode()
        self._last_episode_id = state.episode_id
        self.last_state = state

        log.debug("EP %d started (game ep %d) | enemies=%d",
                  self._episode_num, self._last_episode_id,
                  state.enemy_count)

        obs = self._state_to_obs(state)
        info = {"state": state, "state_raw": state.raw, "reward_components": {}}
        return obs, info

    def step(self, action):
        self.step_count += 1
        self._total_steps += 1
        if self._start_time is None:
            self._start_time = time.monotonic()
        elif self._total_steps % self._throughput_interval == 0:
            elapsed = time.monotonic() - self._start_time
            sps = self._total_steps / elapsed
            log.info("%d steps, %.1f steps/sec", self._total_steps, sps)

        move_action = int(action[0])
        shoot_action = int(action[1])
        bomb_action = int(action[2]) if len(action) > 2 else 0
        active_action = int(action[3]) if len(action) > 3 else 0
        pillcard_action = int(action[4]) if len(action) > 4 else 0

        # Send action (fire-and-forget, Lua latches it)
        action_msg = {
            "command": "action",
            "action": {
                "move": move_action,
                "shoot": shoot_action,
                "bomb": bomb_action,
                "active": active_action,
                "pillcard": pillcard_action,
            },
        }

        # Attach both gameplay rewards and trainer metrics for the in-game HUD.
        action_msg["debug"] = self._build_debug_info()

        self._client.send(action_msg)

        # Receive next state from Lua's stream (measure wait time)
        t_before = time.monotonic()
        state = self._receive_state()
        t_after = time.monotonic()
        step_latency = t_after - t_before
        self._ep_step_latencies.append(step_latency)
        self._ep_receive_times.append(t_after)

        # Detect dropped frames via episode_tick gaps
        episode_tick = state.episode_tick
        if self._last_episode_tick > 0 and episode_tick > 0:
            expected = self._last_episode_tick + 1
            if episode_tick > expected:
                self._ep_frames_dropped += episode_tick - expected
        self._last_episode_tick = episode_tick

        # Compute reward
        reward = self.reward_shaper.compute(state, action)
        self._ep_reward += reward

        # Track episode stats
        rc = self.reward_shaper.reward_components
        for key, value in rc.items():
            self._ep_reward_components[key] = self._ep_reward_components.get(key, 0.0) + value
        if rc.get("kills", 0) > 0:
            self._ep_kills += int(rc["kills"] / self.config.reward.enemy_killed)
        if rc.get("damage_taken", 0) < 0:
            self._ep_damage_taken += 1

        # Terminal from Lua's detection
        terminated = state.terminal
        terminal_reason = state.terminal_reason

        # Gymnasium convention: timeout is truncation, not termination
        truncated = terminal_reason == "timeout"
        if truncated:
            terminated = False

        if terminated or truncated:
            self._last_terminal_reason = terminal_reason

        # Compute live ticks/sec from wall-clock receive times
        game_ticks_per_sec = 0.0
        if len(self._ep_receive_times) >= 2:
            duration = self._ep_receive_times[-1] - self._ep_receive_times[0]
            if duration > 0:
                game_ticks_per_sec = (len(self._ep_receive_times) - 1) / duration
        self._hud_game_ticks_per_sec = game_ticks_per_sec

        obs = self._state_to_obs(state)
        # Fraction of steps where state was already buffered (latency < 1ms)
        instant_ratio = 0.0
        avg_latency = step_latency
        if self._ep_step_latencies:
            instant_ratio = sum(1 for l in self._ep_step_latencies if l < 0.001) / len(self._ep_step_latencies)
            avg_latency = sum(self._ep_step_latencies) / len(self._ep_step_latencies)

        info = {
            "state": state,
            "state_raw": state.raw,
            "reward_components": self.reward_shaper.reward_components.copy(),
            "ep_reward_components": self._ep_reward_components.copy(),
            "ep_kills": self._ep_kills,
            "ep_damage_taken": self._ep_damage_taken,
            "run_metrics": self.reward_shaper.run_metrics(),
            "step_latency": step_latency,
            "avg_step_latency": avg_latency,
            "instant_ratio": instant_ratio,
            "frames_dropped": self._ep_frames_dropped,
            "game_ticks_per_sec": game_ticks_per_sec,
        }

        self.last_state = state
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return flattened action mask for MultiDiscrete([9, 9, 2, 2, 2]).

        Interaction actions are masked by actual resource availability from
        the last received state, and entirely disabled unless
        phase.enable_interactions is set.
        """
        move_mask = np.ones(9, dtype=bool)

        if self.config.phase.mask_shoot:
            # Allow only "don't shoot" (0); mask shoot directions 1..8.
            shoot_mask = np.zeros(9, dtype=bool)
            shoot_mask[0] = True
        else:
            shoot_mask = np.ones(9, dtype=bool)

        if not self.config.phase.enable_interactions or self.last_state is None:
            interaction_mask = np.array([True, False] * 3, dtype=bool)
        else:
            p = self.last_state.player
            bomb_mask = np.array([True, p.num_bombs > 0], dtype=bool)
            active_mask = np.array([True, p.has_active_item], dtype=bool)
            pillcard_mask = np.array(
                [True, p.card_slot != 0 or p.pill_slot != 0], dtype=bool
            )
            interaction_mask = np.concatenate((bomb_mask, active_mask, pillcard_mask))

        return np.concatenate((move_mask, shoot_mask, interaction_mask))

    def end_episode(self, reason: str = "abandoned"):
        """Explicitly end the current episode (victory / abandon).

        Sends the Lua `end_episode` command; the mod marks the next state
        terminal with `reason` and restarts the run. Used by tooling to close
        a full-run episode on a final objective or a manual abort.
        """
        self._client.send({"command": "end_episode", "reason": reason})

    def pause_game(self):
        """Pause the game (used between rollout collection and PPO update)."""
        self._client.send({"command": "pause"})

    def resume_and_flush(self):
        """Resume the game and flush stale TCP data from the buffer."""
        self._client.send({"command": "resume"})
        return self._client.flush()

    def configure_game(self, settings: dict):
        """Send game configuration to the Lua mod."""
        self._game_settings = settings
        was_connected = self._client.sock is not None
        self._client.connect()
        if was_connected:
            self._client.send({"command": "configure", "settings": settings})
        self._configured = True

    def close(self):
        # Best-effort cleanup: don't leave the game paused when training exits.
        if self._client.sock is not None:
            try:
                self._client.send({"command": "resume"})
            except (ConnectionError, OSError, TimeoutError, socket.timeout):
                pass
        self._client.disconnect()
