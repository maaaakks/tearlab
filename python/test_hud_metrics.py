import unittest

import numpy as np

from config import Config
from game_state import GameState, PlayerState
from isaac_env import IsaacEnv


class HudMetricsTests(unittest.TestCase):
    def test_player_features_are_scaled_to_comparable_ranges(self):
        env = IsaacEnv(Config())
        state = GameState(
            grid=np.zeros((10, 7, 13), dtype=np.float32).tolist(),
            player=PlayerState(
                hp_red=12,
                speed=1.5,
                damage=10,
                range=500,
                num_coins=50,
                pos_x=0.75,
                nearest_projectile_dx=-0.5,
            ),
        )

        player = env._state_to_obs(state)["player"]

        self.assertEqual(player.dtype, np.float32)
        self.assertAlmostEqual(float(player[0]), 0.5)
        self.assertAlmostEqual(float(player[3]), 0.5)
        self.assertAlmostEqual(float(player[4]), 0.5)
        self.assertAlmostEqual(float(player[5]), 0.5)
        self.assertAlmostEqual(float(player[11]), 50 / 99)
        self.assertAlmostEqual(float(player[14]), 0.75)
        self.assertAlmostEqual(float(player[20]), -0.5)

    def test_debug_payload_merges_environment_and_training_metrics(self):
        env = IsaacEnv(Config())
        env._episode_num = 3
        env._ep_reward = 12.345
        env.step_count = 48
        env._ep_frames_dropped = 2
        env._hud_game_ticks_per_sec = 9.94
        env.reward_shaper.reward_components = {
            "time": -0.005,
            "damage_dealt": 1.0,
            "kills": 5.0,
        }
        env.update_training_metrics(
            global_steps=4096,
            total_timesteps=1_000_000,
            approx_kl=0.00812,
        )

        payload = env._build_debug_info()

        self.assertEqual(payload["episode"], 3)
        self.assertEqual(payload["episode_reward"], 12.35)
        self.assertEqual(payload["episode_step"], 48)
        self.assertEqual(payload["step_reward"], 5.995)
        self.assertEqual(payload["damage_dealt"], 1.0)
        self.assertEqual(payload["kills"], 5.0)
        self.assertEqual(payload["damage_taken"], 0)
        self.assertEqual(payload["time_reward"], -0.005)
        self.assertEqual(payload["game_tps"], 9.9)
        self.assertEqual(payload["frames_dropped"], 2)
        self.assertEqual(payload["global_steps"], 4096)
        self.assertEqual(payload["total_timesteps"], 1_000_000)
        self.assertEqual(payload["approx_kl"], 0.00812)


if __name__ == "__main__":
    unittest.main()
