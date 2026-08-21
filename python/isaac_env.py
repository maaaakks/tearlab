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
from planner.decision_engine import DecisionEngine
from planner.goals import GOAL_VECTOR_DIM
from planner.rules import PlannerContext, RulePlanner
from reward import RewardShaper
from run_memory import NUM_MAP_CHANNELS, MAP_HEIGHT, MAP_WIDTH, RunMemory

log = logging.getLogger("IsaacEnv")

# Entity-set observation (TODO C2): a fixed-size, zero-padded table of
# identity-bearing objects. Each row packs [kind, slot/type, variant/subtype,
# value/price, normalized x, normalized y, hp/max_hp, ...] so a single
# permutation-invariant encoder can consume enemies, pickups, prices, doors
# and interactive rocks without knowing the object count in advance.
MAX_ENTITIES = 32
ENTITY_DIM = 16


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
        self.reward_shaper = RewardShaper(
            self.config.reward,
            max_episode_steps=self.env_cfg.max_episode_steps,
        )

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

        # Goal conditioning (D5): the planner emits a goal vector, the decision
        # engine emits numeric utility/constraint features; both are appended
        # to the observation for the single generalist policy.
        self.enable_goal_conditioning = bool(
            getattr(self.env_cfg, "enable_goal_conditioning", False)
        )
        if self.enable_goal_conditioning:
            from knowledge_base import KnowledgeBase

            knowledge = KnowledgeBase()
            self.planner = RulePlanner(
                PlannerContext(
                    objective=self.config.phase.objective,
                    preserve_angel=self.config.phase.preserve_angel,
                    preserve_bombs=self.config.phase.preserve_bombs,
                ),
                knowledge=knowledge,
            )
            self.decision_engine = DecisionEngine()

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
        # Identity-bearing object table (TODO C2) — fixed padded shape.
        obs_spaces["entities"] = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(MAX_ENTITIES, ENTITY_DIM),
            dtype=np.float32,
        )
        if self.enable_goal_conditioning:
            obs_spaces["goal"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(GOAL_VECTOR_DIM,),
                dtype=np.float32,
            )
            from planner.decision_engine import DecisionFeatures

            obs_spaces["decision"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(DecisionFeatures.VECTOR_DIM,),
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
        self._throughput_last_steps = 0
        self._throughput_last_time = None

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

        obs: dict = {
            "grid": self._push_grid(grid),
            "player": player,
            "entities": self._state_to_entities(state),
        }
        if self.enable_map_memory:
            obs["map"] = self.run_memory.update(state)
        if self.enable_goal_conditioning:
            obs["goal"] = np.asarray(
                self.planner.goal_vector(state), dtype=np.float32
            )
            obs["decision"] = self._state_to_decision_features(state)
        return obs

    def _state_to_decision_features(self, state: GameState) -> np.ndarray:
        """D5: numeric utility/constraint features for the current decision.

        Derived from the DecisionEngine (health ratio, stage) plus cheap
        resource pressure from the player inventory. The single generalist
        policy consumes these as part of its observation.
        """
        # Deal decision when this is a shop/devil/angel room; otherwise use a
        # neutral context so the features remain meaningful in normal rooms.
        deal_type = "shop_purchase"
        if state.run.room_type == 15:
            deal_type = "angel_room"
        elif state.run.room_type == 14:
            deal_type = "devil_deal"
        _, _, features = self.decision_engine.evaluate(deal_type, state)

        p = state.player
        # Resource pressure in [0,1]: how close the player is to running out
        # of the cheapest consumable they care about. 0 = plenty, 1 = empty.
        consumables = []
        consumables.append(p.num_bombs / max(1.0, 6.0))
        consumables.append(p.num_keys / max(1.0, 6.0))
        consumables.append(p.num_coins / max(1.0, 15.0))
        features.resource_pressure = 1.0 - min(1.0, min(consumables))

        vector = features.to_vector()
        return np.asarray(vector, dtype=np.float32)

    def _state_to_entities(self, state: GameState) -> np.ndarray:
        """Serialize identity-bearing objects into the fixed padded table.

        Row layout (ENTITY_DIM = 16):
          0 kind          1=enemy, 2=pickup, 3=pedestal item, 4=door, 5=rock
          1 type/slot     enemy type | door slot | rock variant
          2 variant/type2 pickup variant | pedestal item quality hint
          3 subtype       pickup subtype | pedestal item id | rock variant
          4 value         enemy hp | pickup price | item price
          5 value_norm    enemy hp fraction (hp/max_hp) | 0
          6 pos_x         normalized 0..1
          7 pos_y         normalized 0..1
          8 locked/cost   door locked | 0 otherwise
          9 open/active   1 if door open / enemy active | 0
         10 hp            0 for non-enemies
         11 max_hp        0 for non-enemies
         12 seed_hash     stable identity hash (0 when unavailable)
         13..15 reserved  0
        """
        table = np.zeros((MAX_ENTITIES, ENTITY_DIM), dtype=np.float32)
        rows: list[list[float]] = []

        for enemy in state.enemies:
            hp = float(enemy.hp)
            max_hp = float(enemy.max_hp) if enemy.max_hp > 0.0 else hp
            px, py = enemy.position or (0.0, 0.0)
            rows.append([
                1.0,
                float(enemy.type),
                float(enemy.variant),
                0.0,
                hp,
                hp / max_hp if max_hp > 0 else 0.0,
                float(px),
                float(py),
                0.0,
                1.0,
                hp,
                max_hp,
                float(enemy.init_seed & 0xFFFFFF),
                0.0, 0.0, 0.0,
            ])

        for pickup in state.pickups:
            px = pickup.pos_x if pickup.position is None else pickup.position[0]
            py = pickup.pos_y if pickup.position is None else pickup.position[1]
            rows.append([
                2.0,
                float(pickup.variant),
                float(pickup.subtype),
                float(pickup.price),
                1.0,
                0.0,
                float(px),
                float(py),
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ])

        for item in state.pedestal_items:
            px = item.pos_x if item.position is None else item.position[0]
            py = item.pos_y if item.position is None else item.position[1]
            rows.append([
                3.0,
                float(item.item_id),
                float(item.price),
                0.0,
                1.0,
                0.0,
                float(px),
                float(py),
                0.0, 0.0, 0.0, 0.0,
                float(item.item_id & 0xFFFFFF),
                0.0, 0.0, 0.0,
            ])

        for door in state.doors:
            px, py = door.position or (door.slot * 0.1, 0.0)
            rows.append([
                4.0,
                float(door.slot),
                float(door.target_room_index),
                0.0,
                1.0 if door.locked else 0.0,
                0.0,
                float(px),
                float(py),
                1.0 if door.locked else 0.0,
                1.0 if door.open else 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ])

        for rock in state.special_rocks:
            px, py = rock.position or (0.0, 0.0)
            rows.append([
                5.0,
                float(rock.variant),
                0.0,
                0.0,
                0.0,
                0.0,
                float(px),
                float(py),
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ])

        for i, row in enumerate(rows[:MAX_ENTITIES]):
            for j, value in enumerate(row):
                table[i, j] = min(1.0, max(-1.0, value))
        return table

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

        # Episode summaries are logged once per episode by live_trainer; the
        # env side stays silent to keep one line per episode.

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
            self._throughput_last_time = self._start_time
        elif self._total_steps % self._throughput_interval == 0:
            # Windowed rate: recent throughput, not a lifetime average that
            # startup stalls drag down forever.
            now = time.monotonic()
            elapsed = now - self._throughput_last_time
            steps_delta = self._total_steps - self._throughput_last_steps
            sps = steps_delta / elapsed if elapsed > 0 else 0.0
            self._throughput_last_time = now
            self._throughput_last_steps = self._total_steps
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
        # Throttled: JSON-encoding this payload and decoding it inside the
        # game's Lua loop every decision costs measurable time at high speed.
        if self.step_count % 10 == 0:
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

    def request_controlled_reset(self, scenario: str) -> bool:
        """E6: request a rare-scenario restart instead of a natural reset.

        The Lua mod acknowledges the scenario when state injection (A4) is
        available; until then it falls back to a plain episode reset so the
        training loop keeps running. Returns True when the scenario hook was
        accepted, False when it fell back.
        """
        try:
            self._client.send(
                {"command": "controlled_reset", "scenario": scenario}
            )
            return True
        except (ConnectionError, OSError, TimeoutError, socket.timeout):
            log.warning(
                "controlled_reset(%s) not available; falling back to reset",
                scenario,
            )
            return False

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
