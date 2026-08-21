import math
from collections.abc import Sequence

from config import RewardConfig
from game_state import GameState, PlayerState


class RewardShaper:
    """Computes reward from state diffs between consecutive ticks."""

    WALL_COLLISION_DISTANCE_THRESHOLD = 1.0
    WALL_COLLISION_AXIS_THRESHOLD = 0.5
    # Normalized pos_x/pos_y thresholds for "near a room wall"
    WALL_PROXIMITY_THRESHOLD = 0.08
    MOVE_DIRECTION_COMPONENTS = {
        1: (0, -1),
        2: (0, 1),
        3: (-1, 0),
        4: (1, 0),
        5: (-1, -1),
        6: (1, -1),
        7: (-1, 1),
        8: (1, 1),
    }

    def __init__(self, config: RewardConfig):
        self.config = config
        self.prev_state: GameState | None = None
        self.reward_components = {}
        self._nav_target: tuple[float, float] | None = None
        self._nav_reached = False
        self._damage_progress_total = 0.0
        self._took_damage_this_room = False

    def reset(self):
        self.prev_state = None
        self.reward_components = {}
        self._nav_target = None
        self._nav_reached = False
        self._damage_progress_total = 0.0
        self._took_damage_this_room = False

    def compute(self, state: GameState, action: Sequence[int] | None = None) -> float:
        reward = 0.0
        self.reward_components = {}

        if self.prev_state is None:
            self.prev_state = state
            return 0.0

        # Time penalty
        reward += self.config.time_penalty
        self.reward_components["time"] = self.config.time_penalty

        # Resource use costs (bomb / active item / card-pill)
        resource_cost = self._compute_use_costs(action)
        reward += resource_cost
        if resource_cost != 0.0:
            self.reward_components["use_cost"] = resource_cost

        # Reward staying alive while enemies are still present.
        if state.enemy_count > 0 and not state.player_dead:
            reward += self.config.survival_bonus
            self.reward_components["survival_bonus"] = self.config.survival_bonus

        # Damage dealt (enemy HP decreased), capped cumulatively per episode
        damage_reward = self._compute_damage_dealt(state)
        reward += damage_reward
        self.reward_components["damage_dealt"] = damage_reward

        # Enemy killed (optional; redundant with damage progress, default off)
        kill_reward = self._compute_kills(state)
        reward += kill_reward
        if kill_reward != 0.0:
            self.reward_components["kills"] = kill_reward

        # Damage taken (per half-heart lost)
        damage_taken_reward = self._compute_damage_taken(state)
        reward += damage_taken_reward
        self.reward_components["damage_taken"] = damage_taken_reward

        # Pickup collected (currently measured via coin count increase)
        pickup_reward = self._compute_pickups(state)
        reward += pickup_reward
        self.reward_components["pickup_collected"] = pickup_reward

        # Pickup approach shaping (reward for getting closer to nearest pickup)
        approach_reward = self._compute_pickup_approach(state)
        reward += approach_reward
        self.reward_components["pickup_approach"] = approach_reward

        wall_collision_reward = self._compute_wall_collision(state, action)
        reward += wall_collision_reward
        self.reward_components["wall_collision"] = wall_collision_reward

        # Navigation smoke-test objective (optional)
        nav_reward, nav_progress_reward, nav_reach_bonus = self._compute_nav_reward(state)
        reward += nav_reward
        if self._nav_target is not None:
            self.reward_components["nav_progress"] = nav_progress_reward
            self.reward_components["nav_reach_bonus"] = nav_reach_bonus

        # Room cleared (+ no-hit bonus if the room was cleared damage-free)
        if state.room_cleared and not self.prev_state.room_cleared:
            reward += self.config.room_cleared
            self.reward_components["room_cleared"] = self.config.room_cleared
            if (
                self.config.no_hit_bonus != 0.0
                and not self._took_damage_this_room
            ):
                reward += self.config.no_hit_bonus
                self.reward_components["no_hit_bonus"] = self.config.no_hit_bonus
            # New room starts fresh for the no-hit tracker
            self._took_damage_this_room = False
        elif state.enemy_count > 0 and self.prev_state.enemy_count == 0:
            # Combat (re)started in a new room without a clear transition
            self._took_damage_this_room = False

        # Death
        if state.player_dead:
            reward += self.config.death
            self.reward_components["death"] = self.config.death

        # Timeout (truncation): penalized at least as much as death so that
        # hiding until timeout is never the optimal strategy.
        if state.terminal_reason == "timeout":
            reward += self.config.timeout_penalty
            self.reward_components["timeout"] = self.config.timeout_penalty

        self.prev_state = state
        return reward

    def _compute_damage_dealt(self, state: GameState) -> float:
        if self._damage_progress_total >= self.config.damage_progress_cap:
            return 0.0

        prev_enemies = {
            e.init_seed: e.hp for e in self.prev_state.enemies if e.init_seed != 0
        }
        legacy_enemies = [
            e for e in self.prev_state.enemies if e.init_seed == 0
        ]
        raw_damage = 0.0
        for enemy in state.enemies:
            damage = None
            if enemy.init_seed != 0 and enemy.init_seed in prev_enemies:
                # Exact identity match via the game's InitSeed
                prev_hp = prev_enemies[enemy.init_seed]
                if enemy.hp < prev_hp:
                    damage = prev_hp - enemy.hp
            elif enemy.init_seed == 0 and legacy_enemies:
                # Fallback for mods/states without InitSeed: best-effort match
                # by type+variant with highest previous HP.
                best_match_hp = None
                for penemy in legacy_enemies:
                    if penemy.type == enemy.type and penemy.variant == enemy.variant:
                        if best_match_hp is None or penemy.hp > best_match_hp:
                            best_match_hp = penemy.hp
                if best_match_hp is not None and enemy.hp < best_match_hp:
                    damage = best_match_hp - enemy.hp
            if damage is not None:
                raw_damage += damage

        if raw_damage <= 0.0:
            return 0.0

        scaled = raw_damage * self.config.damage_dealt
        remaining = self.config.damage_progress_cap - self._damage_progress_total
        granted = min(scaled, remaining)
        self._damage_progress_total += granted
        return granted

    def _compute_kills(self, state: GameState) -> float:
        if self.config.enemy_killed == 0.0:
            return 0.0
        prev_count = self.prev_state.enemy_count
        curr_count = state.enemy_count
        kills = max(0, prev_count - curr_count)
        return kills * self.config.enemy_killed

    def _compute_damage_taken(self, state: GameState) -> float:
        prev_hp = self._total_hp(self.prev_state.player)
        curr_hp = self._total_hp(state.player)
        if curr_hp < prev_hp:
            half_hearts_lost = prev_hp - curr_hp
            self._took_damage_this_room = True
            return half_hearts_lost * self.config.damage_taken
        return 0.0

    def _compute_pickups(self, state: GameState) -> float:
        prev_coins = self.prev_state.player.num_coins
        curr_coins = state.player.num_coins
        collected = max(0, curr_coins - prev_coins)
        return collected * self.config.pickup_collected

    def _compute_pickup_approach(self, state: GameState) -> float:
        """Reward for reducing distance to the nearest pickup (potential-based shaping)."""
        if self.config.pickup_approach_scale == 0.0:
            return 0.0

        # Skip on the step a coin was collected — the nearest pickup jumps to a
        # new respawned location, creating a large spurious negative delta.
        if state.player.num_coins > self.prev_state.player.num_coins:
            return 0.0

        prev_dist = math.sqrt(
            self.prev_state.player.nearest_pickup_dx ** 2
            + self.prev_state.player.nearest_pickup_dy ** 2
        )
        curr_dist = math.sqrt(
            state.player.nearest_pickup_dx ** 2
            + state.player.nearest_pickup_dy ** 2
        )

        if prev_dist == 0.0 or curr_dist == 0.0:
            return 0.0

        return (prev_dist - curr_dist) * self.config.pickup_approach_scale

    def _compute_wall_collision(
        self,
        state: GameState,
        action: Sequence[int] | None,
    ) -> float:
        if self.config.wall_collision_penalty == 0.0:
            return 0.0

        movement_action = self._movement_action(action)
        if movement_action == 0:
            return 0.0

        prev_pos = self._player_position(self.prev_state)
        curr_pos = self._player_position(state)
        if prev_pos is None or curr_pos is None:
            return 0.0

        # Per-axis wall collision: if the player is near a wall on a given
        # axis AND barely moved on that axis, they're colliding.  This
        # correctly detects diagonal movement into walls, where the other
        # axis still has free movement.
        px, py = state.player.pos_x, state.player.pos_y
        dx = abs(curr_pos[0] - prev_pos[0])
        dy = abs(curr_pos[1] - prev_pos[1])

        near_wall_x = (
            px < self.WALL_PROXIMITY_THRESHOLD
            or px > 1.0 - self.WALL_PROXIMITY_THRESHOLD
        )
        near_wall_y = (
            py < self.WALL_PROXIMITY_THRESHOLD
            or py > 1.0 - self.WALL_PROXIMITY_THRESHOLD
        )

        stuck_x = near_wall_x and dx < self.WALL_COLLISION_AXIS_THRESHOLD
        stuck_y = near_wall_y and dy < self.WALL_COLLISION_AXIS_THRESHOLD

        if stuck_x or stuck_y:
            return self.config.wall_collision_penalty

        return 0.0

    def _compute_use_costs(self, action: Sequence[int] | None) -> float:
        if action is None or len(action) < 5:
            return 0.0
        cost = 0.0
        if int(action[2]) == 1:
            cost += self.config.bomb_use_penalty
        if int(action[3]) == 1:
            cost += self.config.active_use_penalty
        if int(action[4]) == 1:
            cost += self.config.pillcard_use_penalty
        return cost

    @staticmethod
    def _movement_action(action: Sequence[int] | None) -> int:
        if action is None:
            return 0

        try:
            if len(action) == 0:
                return 0
        except TypeError:
            return 0

        try:
            return int(action[0])
        except (IndexError, TypeError, ValueError):
            return 0

    def _total_hp(self, player: PlayerState) -> float:
        return player.total_hp

    def _compute_nav_reward(self, state: GameState) -> tuple[float, float, float]:
        nav_enabled = (
            self.config.nav_progress_scale != 0.0
            or self.config.nav_reach_bonus != 0.0
        )
        if not nav_enabled:
            return 0.0, 0.0, 0.0

        if self._nav_target is None:
            self._nav_target = self._resolve_nav_target(state)
        if self._nav_target is None:
            return 0.0, 0.0, 0.0

        prev_pos = self._player_position(self.prev_state)
        curr_pos = self._player_position(state)
        if prev_pos is None or curr_pos is None:
            return 0.0, 0.0, 0.0

        prev_dist = math.dist(prev_pos, self._nav_target)
        curr_dist = math.dist(curr_pos, self._nav_target)

        nav_progress_reward = (prev_dist - curr_dist) * self.config.nav_progress_scale

        nav_reach_bonus = 0.0
        if (
            not self._nav_reached
            and curr_dist <= self.config.nav_reach_radius
        ):
            nav_reach_bonus = self.config.nav_reach_bonus
            self._nav_reached = True

        return nav_progress_reward + nav_reach_bonus, nav_progress_reward, nav_reach_bonus

    def _resolve_nav_target(self, state: GameState) -> tuple[float, float] | None:
        if self.config.nav_target_x is not None and self.config.nav_target_y is not None:
            return float(self.config.nav_target_x), float(self.config.nav_target_y)

        player_pos = self._player_position(state)
        if player_pos is None:
            return None

        return (
            player_pos[0] + self.config.nav_target_dx,
            player_pos[1] + self.config.nav_target_dy,
        )

    @staticmethod
    def _player_position(state: GameState | None) -> tuple[float, float] | None:
        if not state:
            return None
        return state.player.position
