import os
import tempfile
import textwrap
import unittest
from pathlib import Path

from config import Config, load_config


class FullRunConfigTests(unittest.TestCase):
    def test_full_run_defaults_to_false(self):
        config = Config()
        self.assertFalse(config.env.full_run)
        self.assertFalse(config.to_game_settings()["full_run"])

    def test_full_run_is_wired_into_game_settings(self):
        config = Config()
        config.env.full_run = True
        self.assertTrue(config.to_game_settings()["full_run"])

    def test_load_config_reads_full_run(self):
        config_text = textwrap.dedent(
            """
            env:
              full_run: true
              enable_map_memory: true
            phase:
              enable_interactions: true
              spawn_enemies: false
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_text)
            path = handle.name

        try:
            config = load_config(path)
        finally:
            os.unlink(path)

        self.assertTrue(config.env.full_run)
        self.assertTrue(config.env.enable_map_memory)
        self.assertTrue(config.phase.enable_interactions)
        self.assertFalse(config.phase.spawn_enemies)
        self.assertTrue(config.to_game_settings()["full_run"])
        self.assertFalse(config.to_game_settings()["spawn_enemies"])


class SpeedMultiplierConfigTests(unittest.TestCase):
    def test_speed_multiplier_defaults_to_one(self):
        config = Config()
        self.assertEqual(config.env.speed_multiplier, 1)
        self.assertEqual(config.to_game_settings()["speed_multiplier"], 1)

    def test_speed_multiplier_is_wired_and_clamped(self):
        config = Config()
        config.env.speed_multiplier = 64
        self.assertEqual(config.to_game_settings()["speed_multiplier"], 64)
        config.env.speed_multiplier = 99
        self.assertEqual(config.to_game_settings()["speed_multiplier"], 64)
        config.env.speed_multiplier = 0
        self.assertEqual(config.to_game_settings()["speed_multiplier"], 1)

    def test_load_config_reads_speed_multiplier(self):
        config_text = textwrap.dedent(
            """
            env:
              speed_multiplier: 4
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_text)
            path = handle.name

        try:
            config = load_config(path)
        finally:
            os.unlink(path)

        self.assertEqual(config.env.speed_multiplier, 4)
        self.assertEqual(config.to_game_settings()["speed_multiplier"], 4)


class FullGameConfigFileTests(unittest.TestCase):
    def test_full_game_yaml_meets_stage_b_requirements(self):
        repo_root = Path(__file__).resolve().parent.parent
        config = load_config(str(repo_root / "configs" / "full_game.yaml"))

        # B3: full_run, natural spawns, map memory, all action branches, run-scale timeout.
        self.assertTrue(config.env.full_run)
        self.assertFalse(config.phase.spawn_enemies)
        self.assertTrue(config.env.enable_map_memory)
        self.assertTrue(config.phase.enable_interactions)
        self.assertFalse(config.phase.mask_shoot)
        # Global timeout expressed in run time, not a single-room duration.
        self.assertGreater(config.env.max_episode_steps, 10_000)
        # E8: multi-room consequences get a longer discount horizon.
        self.assertEqual(config.train.gamma, 0.995)
        self.assertEqual(config.reward.time_penalty, 0.0)
        self.assertGreater(config.reward.clear_speed_bonus, 0.0)
        self.assertLess(config.reward.death_early_penalty, 0.0)
        self.assertGreater(config.train.aim_aux_coef, 0.0)


class UnifiedCombatConfigTests(unittest.TestCase):
    def test_phase1a_uses_success_conditioned_speed_and_auxiliary_aim(self):
        repo_root = Path(__file__).resolve().parent.parent
        config = load_config(str(repo_root / "configs" / "phase1a.yaml"))

        self.assertEqual(config.reward.time_penalty, 0.0)
        self.assertEqual(config.reward.aim_misalignment_penalty, 0.0)
        self.assertEqual(config.reward.aim_shoot_no_enemy_penalty, 0.0)
        # 30s rooms: horizon matches max_episode_steps (300 decisions).
        self.assertEqual(config.reward.clear_speed_horizon, 300)
        # Safety-first hierarchy: speed only pays on a no-hit clear.
        self.assertTrue(config.reward.clear_speed_requires_no_hit)
        self.assertGreater(config.reward.no_hit_bonus, 1.0)
        self.assertLess(config.reward.death_early_penalty, 0.0)
        self.assertLess(config.reward.timeout_enemy_hp_penalty, 0.0)
        self.assertGreater(config.train.aim_aux_coef, 0.0)
        self.assertGreater(config.train.aim_aux_idle_weight, 0.0)
        # C5 representation heads active.
        self.assertGreater(config.train.aux_coef, 0.0)


if __name__ == "__main__":
    unittest.main()
