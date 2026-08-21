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


if __name__ == "__main__":
    unittest.main()
