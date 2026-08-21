"""Tests for the B4 generalist reward system and B2 lifecycle events."""

import unittest

from config import RewardConfig
from game_state import EnemyState, GameState, PlayerState, RunInfo
from reward import RewardShaper, ROOM_TYPE_BOSS


def make_enemy(seed: int, hp: float, max_hp: float, etype: int = 10) -> EnemyState:
    return EnemyState(
        type=etype,
        variant=0,
        hp=hp,
        position=(0.0, 0.0),
        init_seed=seed,
        max_hp=max_hp,
    )


def state(
    *,
    room_index: int = 1,
    stage: int = 1,
    stage_type: int = 0,
    room_type: int = 1,
    room_clear: bool = False,
    enemies: list | None = None,
    hp_red: float = 6.0,
    hp_soul: float = 0.0,
    player_dead: bool = False,
    terminal: bool = False,
    terminal_reason: str | None = None,
    events: list | None = None,
) -> GameState:
    return GameState(
        run=RunInfo(
            stage=stage,
            stage_type=stage_type,
            room_type=room_type,
            room_list_index=room_index,
            room_clear=room_clear,
        ),
        enemies=enemies or [],
        enemy_count=len(enemies) if enemies else 0,
        player=PlayerState(hp_red=hp_red, hp_soul=hp_soul),
        player_dead=player_dead,
        room_cleared=room_clear,
        terminal=terminal,
        terminal_reason=terminal_reason,
        events=events or [],
    )


class B4DamageProgressTests(unittest.TestCase):
    def make_shaper(self, **overrides) -> RewardShaper:
        return RewardShaper(
            RewardConfig(
                damage_dealt=1.0,
                damage_progress_cap=0.25,
                enemy_killed=0.0,
                room_cleared=2.0,
                no_hit_bonus=0.0,
                damage_taken=0.0,
                death=0.0,
                time_penalty=0.0,
                room_first_visit=0.0,
                boss_killed=0.0,
                floor_cleared=0.0,
                **overrides,
            )
        )

    def test_damage_is_fraction_of_room_hp_pool(self):
        shaper = self.make_shaper()
        shaper.compute(state(enemies=[make_enemy(1, 10.0, 10.0)]))
        # 2 of 10 HP -> 20% of the 0.25 cap = 0.05.
        reward = shaper.compute(state(enemies=[make_enemy(1, 8.0, 10.0)]))
        self.assertAlmostEqual(shaper.reward_components["damage_dealt"], 0.05, places=6)

    def test_final_blow_counts_when_enemy_disappears(self):
        shaper = self.make_shaper()
        shaper.compute(state(enemies=[make_enemy(1, 10.0, 10.0)]))
        # Killing blow removes the enemy's remaining 10 HP = 100% of the
        # room's HP pool, so the whole damage cap (0.25) is granted.
        reward = shaper.compute(state(enemies=[], room_clear=True))
        self.assertAlmostEqual(shaper.reward_components["damage_dealt"], 0.25, places=6)
        # Room clear is granted on the same step.
        self.assertAlmostEqual(shaper.reward_components["room_cleared"], 2.0)

    def test_budget_resets_per_room(self):
        shaper = self.make_shaper()
        # Room 1: hit the cap.
        shaper.compute(state(room_index=1, enemies=[make_enemy(1, 10.0, 10.0)]))
        shaper.compute(state(room_index=1, enemies=[]))
        # Room 2 (new index): full damage budget is available again.
        shaper.compute(state(room_index=2, enemies=[make_enemy(2, 10.0, 10.0)]))
        reward = shaper.compute(state(room_index=2, enemies=[make_enemy(2, 5.0, 10.0)]))
        self.assertAlmostEqual(shaper.reward_components["damage_dealt"], 0.125, places=5)

    def test_no_damage_reward_across_room_transition(self):
        shaper = self.make_shaper()
        shaper.compute(state(room_index=1, enemies=[make_enemy(1, 10.0, 10.0)]))
        # New room with no enemies must not count the old room's enemy as dead.
        reward = shaper.compute(state(room_index=2, enemies=[]))
        self.assertNotIn("damage_dealt", shaper.reward_components)


class B4MilestoneTests(unittest.TestCase):
    def make_shaper(self, **overrides) -> RewardShaper:
        return RewardShaper(
            RewardConfig(
                damage_dealt=0.0,
                enemy_killed=0.0,
                room_cleared=1.0,
                no_hit_bonus=0.25,
                damage_taken=0.0,
                death=0.0,
                time_penalty=0.0,
                room_first_visit=0.05,
                boss_killed=0.5,
                floor_cleared=2.0,
                run_won=10.0,
                deal_protected_penalty=-0.05,
                deal_protected_floor_bonus=0.25,
                **overrides,
            )
        )

    def test_clear_bonus_only_if_alive(self):
        shaper = self.make_shaper()
        shaper.compute(state(enemies=[make_enemy(1, 10.0, 10.0)]))
        # Simultaneous death + clear: the clear bonus is withheld.
        shaper.compute(
            state(enemies=[], room_clear=True, player_dead=True, terminal=True,
                 terminal_reason="death")
        )
        self.assertNotIn("room_cleared", shaper.reward_components)
        self.assertEqual(shaper.reward_components["death"], 0.0)  # death=0 in test cfg

    def test_boss_killed_reward_via_event(self):
        shaper = self.make_shaper()
        shaper.compute(state(room_index=1, enemies=[make_enemy(1, 1.0, 1.0)]))
        shaper.compute(
            state(room_index=1, room_clear=True, room_type=ROOM_TYPE_BOSS,
                  events=[{"name": "boss_killed"}, {"name": "room_cleared"}])
        )
        self.assertEqual(shaper.reward_components["boss_killed"], 0.5)
        self.assertEqual(shaper.reward_components["room_cleared"], 1.0)

    def test_floor_cleared_and_deal_protection_bonus(self):
        shaper = self.make_shaper()
        shaper.compute(state(stage=1, enemies=[make_enemy(1, 1.0, 1.0)]))
        # Clean floor transition -> floor_cleared + deal protection bonus.
        shaper.compute(state(stage=2, enemies=[], events=[{"name": "floor_cleared"}]))
        self.assertEqual(shaper.reward_components["floor_cleared"], 2.0)
        self.assertEqual(shaper.reward_components["deal_protected"], 0.25)

    def test_deal_protection_penalty_on_red_heart_damage(self):
        shaper = self.make_shaper()
        shaper.compute(state(hp_red=6.0, enemies=[make_enemy(1, 1.0, 1.0)]))
        shaper.compute(state(hp_red=4.0, enemies=[make_enemy(1, 1.0, 1.0)]))
        # 2 half-hearts lost -> -0.10, floor bonus suppressed by later clear.
        shaper.compute(
            state(stage=2, enemies=[], events=[{"name": "floor_cleared"}])
        )
        self.assertAlmostEqual(shaper.reward_components["damage_taken"], 0.0)
        # Sum over steps: penalty -0.10, no floor bonus because red hearts dropped.
        self.assertNotIn("deal_protected", shaper.reward_components)

    def test_run_won_reward(self):
        shaper = self.make_shaper()
        shaper.compute(state())
        shaper.compute(
            state(terminal=True, terminal_reason="run_won",
                  events=[{"name": "run_won"}])
        )
        self.assertEqual(shaper.reward_components["run_won"], 10.0)


class B4FirstVisitTests(unittest.TestCase):
    def make_shaper(self) -> RewardShaper:
        return RewardShaper(
            RewardConfig(
                damage_dealt=0.0,
                enemy_killed=0.0,
                room_cleared=0.0,
                damage_taken=0.0,
                death=0.0,
                time_penalty=0.0,
                room_first_visit=0.05,
            )
        )

    def test_first_visit_paid_once_and_revisit_free(self):
        shaper = self.make_shaper()
        # First tick in room 1 pays the one-shot exploration bonus.
        reward = shaper.compute(state(room_index=1))
        self.assertAlmostEqual(reward, 0.05, places=6)
        # Same room again: no second bonus (loop farming blocked).
        reward = shaper.compute(state(room_index=1))
        self.assertNotIn("room_first_visit", shaper.reward_components)
        reward = shaper.compute(state(room_index=1))
        self.assertNotIn("room_first_visit", shaper.reward_components)

    def test_two_distinct_rooms_pay_twice(self):
        shaper = self.make_shaper()
        shaper.compute(state(room_index=1))
        shaper.compute(state(room_index=1))
        reward = shaper.compute(state(room_index=2))
        self.assertAlmostEqual(reward, 0.05, places=6)


class B4DeathDominanceTests(unittest.TestCase):
    def test_death_stays_negative_after_max_gameplay_progress(self):
        # Even a perfect room clear + floor + boss cannot make death profitable.
        shaper = RewardShaper(
            RewardConfig(
                damage_dealt=1.0,
                damage_progress_cap=0.25,
                enemy_killed=0.0,
                room_cleared=1.0,
                no_hit_bonus=0.25,
                damage_taken=-0.5,
                death=-10.0,
                time_penalty=0.0,
                room_first_visit=0.05,
                boss_killed=0.5,
                floor_cleared=2.0,
                run_won=10.0,
            )
        )
        shaper.compute(state(enemies=[make_enemy(1, 10.0, 10.0)], hp_red=6.0))
        # Kill the enemy and clear the boss room but die taking 6 red hearts.
        reward = shaper.compute(
            state(room_clear=True, room_type=ROOM_TYPE_BOSS, player_dead=True,
                  terminal=True, terminal_reason="death", hp_red=0.0,
                  enemies=[], events=[{"name": "room_cleared"}, {"name": "boss_killed"}])
        )
        # The death term plus damage-taken term must outweigh all positives.
        self.assertLess(reward, 0.0)


if __name__ == "__main__":
    unittest.main()
