import math
from collections.abc import Sequence

from config import RewardConfig
from game_state import GameState, PlayerState

ROOM_TYPE_BOSS = 5


class RewardShaper:
    """Computes reward from state diffs between consecutive ticks.

    Room-scoped counters (damage budget, no-hit, enemy HP pool) reset whenever
    the player enters a new room. Run-scoped counters (visited rooms, boss
    kills, floors) survive for the whole episode. Lifecycle events emitted by
    the Lua mod (`state.events`) are consumed when present, with a diff-based
    fallback for old states that predate the event stream.
    """

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

    def __init__(self, config: RewardConfig, max_episode_steps: int = 0):
        self.config = config
        self.max_episode_steps = max(0, int(max_episode_steps))
        self.prev_state: GameState | None = None
        self.reward_components = {}
        self._nav_target: tuple[float, float] | None = None
        self._nav_reached = False

        # Room-scoped counters (reset on room change)
        self._room_damage_total = 0.0
        self._room_total_hp = 0.0
        self._room_enemy_seeds: set[int] = set()
        self._took_damage_this_room = False
        self._room_key: tuple | None = None
        self._room_start_tick = 0

        # Run-scoped counters (reset only in reset())
        self._visited_rooms: set[tuple] = set()
        self._boss_kills = 0
        self._floors_seen = 0

        # Floor-scoped counters (reset on floor change)
        self._floor_key: tuple | None = None
        self._red_heart_damage_this_floor = 0.0
        self._pending_first_visit = 0.0

    def reset(self):
        self.prev_state = None
        self.reward_components = {}
        self._nav_target = None
        self._nav_reached = False
        self._room_damage_total = 0.0
        self._room_total_hp = 0.0
        self._room_enemy_seeds = set()
        self._took_damage_this_room = False
        self._room_key = None
        self._room_start_tick = 0
        self._visited_rooms = set()
        self._boss_kills = 0
        self._floors_seen = 0
        self._floor_key = None
        self._red_heart_damage_this_floor = 0.0
        self._pending_first_visit = 0.0

    # -- state keys -----------------------------------------------------

    @staticmethod
    def _room_key_of(state: GameState) -> tuple:
        li = state.run.room_list_index
        if li is None or li < 0:
            li = state.run.room_grid_index
        return (state.run.stage, state.run.stage_type, li)

    @staticmethod
    def _floor_key_of(state: GameState) -> tuple:
        return (state.run.stage, state.run.stage_type)

    def _event_names(self, state: GameState) -> set[str]:
        return {e.get("name", "") for e in getattr(state, "events", []) or []}

    # -- main entry point ----------------------------------------------

    def compute(self, state: GameState, action: Sequence[int] | None = None) -> float:
        self.reward_components = {}

        if self.prev_state is None:
            self.prev_state = state
            self._room_key = self._room_key_of(state)
            self._room_start_tick = max(0, int(state.episode_tick))
            self._floor_key = self._floor_key_of(state)
            # Seed the room HP pool from the first sighting of this room's
            # enemies so damage fractions are correct from the very first
            # transition (the enemy HP present now is part of the pool even
            # though we only observe it once).
            self._ensure_enemies_seen(state)
            first_visit = self._room_key not in self._visited_rooms
            self._visited_rooms.add(self._room_key)
            self._floors_seen = 1
            if first_visit:
                self._pending_first_visit = self.config.room_first_visit
            return self._process_events(state)

        reward = 0.0

        # Time penalty
        reward += self.config.time_penalty
        self.reward_components["time"] = self.config.time_penalty

        # Resource use costs (bomb / active item / card-pill)
        resource_cost = self._compute_use_costs(action)
        reward += resource_cost
        if resource_cost != 0.0:
            self.reward_components["use_cost"] = resource_cost

        # Aim discipline (precision over spraying)
        aim_cost = self._compute_aim(state, action)
        reward += aim_cost
        if aim_cost != 0.0:
            self.reward_components["aim"] = aim_cost

        # Reward staying alive while enemies are still present.
        if state.enemy_count > 0 and not state.player_dead:
            reward += self.config.survival_bonus
            self.reward_components["survival_bonus"] = self.config.survival_bonus

        room_changed = self._room_key_of(state) != self._room_key
        floor_changed = self._floor_key_of(state) != self._floor_key

        # One-shot lifecycle rewards first: they close out the OLD room/floor,
        # so they must see the old room's no-hit and red-heart counters.
        reward += self._process_events(state)

        # Damage progress only within the same room; a room transition must not
        # count the old room's enemies as "killed".
        if not room_changed:
            damage_reward = self._compute_damage_dealt(state)
            reward += damage_reward
            self.reward_components["damage_dealt"] = damage_reward

            # Enemy killed (optional; redundant with damage progress)
            kill_reward = self._compute_kills(state)
            reward += kill_reward
            if kill_reward != 0.0:
                self.reward_components["kills"] = kill_reward

        # Room / floor transition bookkeeping (resets room-level counters and
        # arms the first-visit bonus for the new room).
        self._update_room_and_floor_tracking(state)
        reward += self._consume_pending_first_visit()

        # Damage taken (per half-heart lost) + red-heart deal protection signal.
        # After the tracking update this attributes to the new room on a
        # room-change step.
        damage_taken_reward = self._compute_damage_taken(state)
        reward += damage_taken_reward
        self.reward_components["damage_taken"] = damage_taken_reward

        # Pickup collected (opt-in legacy: value-aware rewards live in D4/D5)
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

        # Death is always last so a simultaneous clear cannot mask it.
        if state.player_dead:
            death_reward = self._death_reward(state)
            reward += death_reward
            self.reward_components["death"] = death_reward

        # Timeout (truncation): remaining enemy HP makes untouched-room camping
        # worse, while survival still ranks above death.
        if state.terminal_reason == "timeout":
            timeout_reward = self._timeout_reward(state)
            reward += timeout_reward
            self.reward_components["timeout"] = timeout_reward

        self.prev_state = state
        return reward

    # -- room / floor lifecycle -----------------------------------------

    def _update_room_and_floor_tracking(self, state: GameState):
        room_key = self._room_key_of(state)
        floor_key = self._floor_key_of(state)

        if room_key != self._room_key:
            # Entering a new room: reset room-level counters and pay the
            # one-shot first-visit bonus.
            self._room_damage_total = 0.0
            self._room_total_hp = 0.0
            self._room_enemy_seeds = set()
            self._took_damage_this_room = False
            self._room_key = room_key
            self._room_start_tick = max(0, int(state.episode_tick))

            first_visit = room_key not in self._visited_rooms
            self._visited_rooms.add(room_key)
            if first_visit:
                self._pending_first_visit = self.config.room_first_visit

        if floor_key != self._floor_key:
            self._floor_key = floor_key
            self._floors_seen += 1
            self._red_heart_damage_this_floor = 0.0

    def _consume_pending_first_visit(self) -> float:
        bonus = getattr(self, "_pending_first_visit", 0.0)
        if bonus != 0.0:
            self.reward_components["room_first_visit"] = bonus
            self._pending_first_visit = 0.0
            return bonus
        return 0.0

    def _process_events(self, state: GameState) -> float:
        """Consume Lua lifecycle events; falls back to state diffs."""
        reward = 0.0
        names = self._event_names(state)

        # First-visit bonus is tracked via room keys (see
        # _update_room_and_floor_tracking), not via room_entered events, so it
        # works with and without the event stream.
        reward += self._consume_pending_first_visit()

        clear_event = "room_cleared" in names
        boss_event = "boss_killed" in names
        floor_event = "floor_cleared" in names
        run_event = "run_won" in names

        # Room clear (+ no-hit) only if Isaac is alive; death wins ties.
        cleared = clear_event or (
            state.room_cleared and not self.prev_state.room_cleared
        )
        if cleared and not state.player_dead:
            reward += self.config.room_cleared
            self.reward_components["room_cleared"] = self.config.room_cleared
            speed_bonus = self._clear_speed_reward(state)
            if (
                speed_bonus != 0.0
                and not (
                    self.config.clear_speed_requires_no_hit
                    and self._took_damage_this_room
                )
            ):
                reward += speed_bonus
                self.reward_components["clear_speed_bonus"] = speed_bonus
            if (
                self.config.no_hit_bonus != 0.0
                and not self._took_damage_this_room
            ):
                reward += self.config.no_hit_bonus
                self.reward_components["no_hit_bonus"] = self.config.no_hit_bonus
            self._took_damage_this_room = False

        # Boss kill (bounded, one-shot).
        boss_killed = boss_event or (
            cleared and state.run.room_type == ROOM_TYPE_BOSS
        )
        if boss_killed:
            reward += self.config.boss_killed
            self.reward_components["boss_killed"] = self.config.boss_killed
            self._boss_kills += 1

        # Floor cleared (+ deal protection bonus for a red-heart-clean floor).
        floor_cleared = floor_event or self._detect_floor_change_from_prev(state)
        if floor_cleared:
            reward += self.config.floor_cleared
            self.reward_components["floor_cleared"] = self.config.floor_cleared
            if (
                self.config.deal_protected_floor_bonus != 0.0
                and self._red_heart_damage_this_floor == 0.0
            ):
                reward += self.config.deal_protected_floor_bonus
                self.reward_components["deal_protected"] = (
                    self.reward_components.get("deal_protected", 0.0)
                    + self.config.deal_protected_floor_bonus
                )
            self._red_heart_damage_this_floor = 0.0

        # Run won.
        if run_event or state.terminal_reason == "run_won":
            reward += self.config.run_won
            self.reward_components["run_won"] = self.config.run_won

        return reward

    def _clear_speed_reward(self, state: GameState) -> float:
        """Success-only urgency: faster clears earn more, failures pay nothing."""
        bonus = float(self.config.clear_speed_bonus)
        horizon = int(self.config.clear_speed_horizon)
        if bonus == 0.0 or horizon <= 0:
            return 0.0
        elapsed = max(0, int(state.episode_tick) - self._room_start_tick)
        progress = min(1.0, elapsed / horizon)
        return bonus * (1.0 - progress)

    def _death_reward(self, state: GameState) -> float:
        """Make early death worse without charging the agent for being alive."""
        reward = float(self.config.death)
        early = float(self.config.death_early_penalty)
        if early == 0.0 or self.max_episode_steps <= 0:
            return reward
        progress = min(
            1.0,
            max(0.0, float(state.episode_tick) / self.max_episode_steps),
        )
        return reward + early * (1.0 - progress)

    def _timeout_reward(self, state: GameState) -> float:
        """Penalize camping in proportion to unfinished enemy HP."""
        reward = float(self.config.timeout_penalty)
        hp_penalty = float(self.config.timeout_enemy_hp_penalty)
        if hp_penalty == 0.0:
            return reward
        self._ensure_enemies_seen(state)
        if self._room_total_hp <= 0.0:
            return reward
        remaining_hp = sum(max(0.0, enemy.hp) for enemy in state.enemies)
        remaining_fraction = min(1.0, remaining_hp / self._room_total_hp)
        return reward + hp_penalty * remaining_fraction

    def _detect_floor_change_from_prev(self, state: GameState) -> bool:
        if self.prev_state is None:
            return False
        return self._floor_key_of(state) != self._floor_key_of(self.prev_state)

    # -- damage ---------------------------------------------------------

    def _compute_damage_dealt(self, state: GameState) -> float:
        if self.config.damage_dealt == 0.0:
            return 0.0
        if self._room_damage_total >= self.config.damage_progress_cap:
            return 0.0
        # Never attribute cross-room disappearances to damage.
        if self._room_key_of(state) != self._room_key:
            return 0.0

        self._ensure_enemies_seen(state)
        self._ensure_enemies_seen(self.prev_state)
        raw_damage = self._damage_since_prev(state)
        if raw_damage <= 0.0:
            return 0.0

        if self._room_total_hp > 0.0:
            # Fraction of the room's total HP pool represented by this damage.
            progress = raw_damage / self._room_total_hp
        else:
            # Unknown HP pool (empty/legacy state): fall back to raw scaling.
            progress = raw_damage

        # The full room's damage is worth damage_dealt * damage_progress_cap;
        # this step earns its fraction of that. Capping per room makes reward
        # comparable across rooms regardless of enemy count or total HP.
        scaled = (
            self.config.damage_dealt
            * self.config.damage_progress_cap
            * progress
        )
        remaining = self.config.damage_progress_cap - self._room_damage_total
        granted = min(scaled, remaining)
        self._room_damage_total += granted
        return granted

    def _ensure_enemies_seen(self, state: GameState):
        for enemy in state.enemies:
            if enemy.init_seed != 0 and enemy.init_seed not in self._room_enemy_seeds:
                self._room_enemy_seeds.add(enemy.init_seed)
                hp = enemy.max_hp if enemy.max_hp > 0.0 else enemy.hp
                self._room_total_hp += hp

    def _damage_since_prev(self, state: GameState) -> float:
        prev_by_seed = {
            e.init_seed: e for e in self.prev_state.enemies if e.init_seed != 0
        }
        curr_by_seed = {
            e.init_seed: e for e in state.enemies if e.init_seed != 0
        }
        raw_damage = 0.0

        for seed, penemy in prev_by_seed.items():
            cenemy = curr_by_seed.get(seed)
            if cenemy is not None:
                if cenemy.hp < penemy.hp:
                    raw_damage += penemy.hp - cenemy.hp
            else:
                # Disappeared from the room: the killing blow removed its
                # remaining HP, so count it instead of dropping the final hit.
                raw_damage += max(0.0, penemy.hp)

        # Legacy fallback for states without InitSeed: match alive enemies by
        # type+variant against the highest previous HP of that type.
        legacy_prev = [e for e in self.prev_state.enemies if e.init_seed == 0]
        legacy_curr = [e for e in state.enemies if e.init_seed == 0]
        if legacy_prev and legacy_curr:
            for cenemy in legacy_curr:
                best_match_hp = None
                for penemy in legacy_prev:
                    if (
                        penemy.type == cenemy.type
                        and penemy.variant == cenemy.variant
                    ):
                        if best_match_hp is None or penemy.hp > best_match_hp:
                            best_match_hp = penemy.hp
                if best_match_hp is not None and cenemy.hp < best_match_hp:
                    raw_damage += best_match_hp - cenemy.hp

        return raw_damage

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
        reward = 0.0
        if curr_hp < prev_hp:
            half_hearts_lost = prev_hp - curr_hp
            self._took_damage_this_room = True
            reward = half_hearts_lost * self.config.damage_taken

        # Red-heart damage is the deal-protection signal (floor-scoped).
        prev_red = self.prev_state.player.hp_red
        curr_red = state.player.hp_red
        if curr_red < prev_red:
            red_loss = prev_red - curr_red
            self._red_heart_damage_this_floor += red_loss
            if self.config.deal_protected_penalty != 0.0:
                penalty = red_loss * self.config.deal_protected_penalty
                reward += penalty
                self.reward_components["deal_protected"] = (
                    self.reward_components.get("deal_protected", 0.0) + penalty
                )

        return reward

    # -- pickups ---------------------------------------------------------

    def _compute_pickups(self, state: GameState) -> float:
        if self.config.pickup_collected == 0.0:
            return 0.0
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

    # -- wall collision --------------------------------------------------

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

    # Shot direction unit vectors (screen coords: +x right, +y down).
    # Matches ActionInjector SHOOT_MAP: 1=up 2=down 3=left 4=right,
    # 5=UL 6=UR 7=DL 8=DR.
    _SHOT_VECTORS = {
        1: (0.0, -1.0), 2: (0.0, 1.0), 3: (-1.0, 0.0), 4: (1.0, 0.0),
        5: (-1.0, -1.0), 6: (1.0, -1.0), 7: (-1.0, 1.0), 8: (1.0, 1.0),
    }

    def _compute_aim(self, state: GameState, action: Sequence[int] | None) -> float:
        if (
            self.config.aim_misalignment_penalty == 0.0
            and self.config.aim_shoot_no_enemy_penalty == 0.0
        ):
            return 0.0
        if action is None or len(action) < 2:
            return 0.0

        shoot = int(action[1])
        if shoot == 0:
            return 0.0

        if state.enemy_count == 0:
            return self.config.aim_shoot_no_enemy_penalty

        # Bearing to the nearest enemy (serializer deltas keep screen coords)
        ex, ey = state.player.nearest_enemy_dx, state.player.nearest_enemy_dy
        if abs(ex) < 1e-6 and abs(ey) < 1e-6:
            return 0.0  # on top of us; any direction is fine

        sx, sy = self._SHOT_VECTORS.get(shoot, (0.0, 0.0))
        dot = sx * ex + sy * ey
        norm_s = math.hypot(sx, sy)
        norm_e = math.hypot(ex, ey)
        cos_err = max(-1.0, min(1.0, dot / (norm_s * norm_e)))
        error_deg = math.degrees(math.acos(cos_err))

        tolerance = self.config.aim_tolerance_deg
        if error_deg <= tolerance:
            return 0.0

        # Linear ramp from 0 at the tolerance edge to full penalty at 180 deg
        frac = (error_deg - tolerance) / max(1e-6, 180.0 - tolerance)
        return self.config.aim_misalignment_penalty * min(1.0, frac)

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

    # -- run metrics -----------------------------------------------------

    def run_metrics(self) -> dict:
        """Run-level counters for dashboards / info dicts."""
        return {
            "rooms_visited": len(self._visited_rooms),
            "boss_kills": self._boss_kills,
            "floors_seen": self._floors_seen,
            "red_heart_damage_this_floor": self._red_heart_damage_this_floor,
        }
