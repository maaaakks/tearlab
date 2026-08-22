import unittest
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces

from config import Config
from live_trainer import (
    IsaacPolicy,
    LivePPOTrainer,
    next_state_targets,
)


def _make_observation_space():
    return spaces.Dict(
        {
            "grid": spaces.Box(0.0, 1.0, (10, 7, 13), np.float32),
            "player": spaces.Box(-2.0, 2.0, (22,), np.float32),
            "entities": spaces.Box(-1.0, 1.0, (32, 16), np.float32),
        }
    )


class NextStateTargetTests(unittest.TestCase):
    def test_target_summary_matches_next_obs(self):
        players = torch.zeros((3, 22), dtype=torch.float32)
        players[0, 14:16] = torch.tensor([0.5, -0.25])   # player pos
        players[0, 18:20] = torch.tensor([0.1, 0.2])     # nearest enemy delta
        players[1, 20:22] = torch.tensor([-0.4, 0.0])    # projectile delta

        entities = torch.zeros((3, 32, 16), dtype=torch.float32)
        # Batch 0: two enemies at known positions.
        entities[0, 0, 0] = 1.0   # kind = enemy
        entities[0, 0, 6:8] = torch.tensor([0.6, 0.8])
        entities[0, 1, 0] = 1.0
        entities[0, 1, 6:8] = torch.tensor([0.2, -0.4])
        # Batch 1: one pickup only (kind=2) -> no enemies.
        entities[1, 0, 0] = 2.0
        # Batch 2: no entities at all.

        targets = next_state_targets(players, entities)

        self.assertEqual(targets.shape, (3, 11))
        # Player position + deltas are copied through.
        self.assertAlmostEqual(targets[0, 0].item(), 0.5)
        self.assertAlmostEqual(targets[0, 1].item(), -0.25)
        self.assertAlmostEqual(targets[0, 5].item(), 0.2)
        self.assertAlmostEqual(targets[1, 6].item(), -0.4)
        # Enemy count normalized by 8: two enemies -> 0.25, none -> 0.
        self.assertAlmostEqual(targets[0, 8].item(), 0.25)
        self.assertAlmostEqual(targets[1, 8].item(), 0.0)
        # Mean enemy position of batch 0 = ((0.6+0.2)/2, (0.8-0.4)/2).
        self.assertAlmostEqual(targets[0, 9].item(), 0.4)
        self.assertAlmostEqual(targets[0, 10].item(), 0.2)

    def test_targets_without_entities_fill_zeros(self):
        players = torch.randn((4, 22))
        targets = next_state_targets(players, None)
        self.assertEqual(targets.shape, (4, 11))
        self.assertTrue(torch.equal(targets[:, 8:11], torch.zeros((4, 3))))

    def test_rejects_legacy_player_vectors(self):
        with self.assertRaisesRegex(ValueError, "22-value player vector"):
            next_state_targets(torch.zeros((2, 14)), None)


class NextStateHeadUpdateTests(unittest.TestCase):
    def _make_trainer(self):
        config = Config()
        config.train.n_epochs = 1
        config.train.batch_size = 4

        observation_space = _make_observation_space()
        action_nvec = [9, 9, 2, 2, 2]

        trainer = LivePPOTrainer.__new__(LivePPOTrainer)
        trainer.config = config
        trainer.device = torch.device("cpu")
        trainer.policy = IsaacPolicy(
            observation_space,
            action_nvec,
            aux_coef=0.05,
            next_state_coef=0.05,
        )
        self.assertTrue(trainer.policy.next_state_enabled)
        self.assertIn(
            "aux_next_state.weight",
            dict(trainer.policy.named_parameters()),
        )
        trainer.optimizer = torch.optim.Adam(
            trainer.policy.parameters(), lr=3e-4
        )
        trainer.gamma = 0.99
        trainer.gae_lambda = 0.95
        trainer.clip_range = 0.2
        trainer.ent_coef = 0.01
        trainer.ent_coef_final = 0.01
        trainer.entropy_decay_fraction = 1.0
        trainer.aim_aux_coef = 0.0
        trainer.aim_aux_final = 0.0
        trainer.aim_aux_decay_fraction = 0.0
        trainer.aim_aux_idle_weight = 0.25
        trainer.vf_coef = 0.5
        trainer.max_grad_norm = 0.5
        trainer.n_epochs = 1
        trainer.batch_size = 4
        trainer.recurrent = False
        trainer.recurrent_hidden_size = 128
        trainer.aux_coef = 0.05
        trainer.next_state_coef = 0.05
        trainer.normalize_returns = False
        trainer.return_norm_decay = 0.999
        trainer._return_mean = 0.0
        trainer._return_var = 1.0
        trainer._return_stats_seen = 0
        trainer.total_timesteps = 1_000
        trainer.collector = SimpleNamespace(steps=10)
        return trainer

    @staticmethod
    def _synthetic_batch(n: int = 8):
        rng = np.random.default_rng(0)
        batch = []
        for i in range(n):
            player = rng.normal(0, 0.3, size=22).astype(np.float32)
            obs = {
                "grid": rng.random((10, 7, 13)).astype(np.float32),
                "player": player,
                "entities": (
                    rng.random((32, 16)).astype(np.float32) * 2.0 - 1.0
                ),
            }
            next_player = player + rng.normal(0, 0.05, size=22).astype(
                np.float32
            )
            next_obs = {
                "grid": rng.random((10, 7, 13)).astype(np.float32),
                "player": next_player,
                "entities": (
                    rng.random((32, 16)).astype(np.float32) * 2.0 - 1.0
                ),
            }
            batch.append(
                {
                    "obs": obs,
                    "next_obs": next_obs,
                    "action": (0, 0, 0, 0, 0),
                    "mask": np.ones(sum([9, 9, 2, 2, 2]), dtype=bool),
                    "logprob": 0.0,
                    "value": 0.0,
                    "reward": 0.0,
                    "done": i == n - 1,
                    "next_value": 0.0,
                    "hidden_in": None,
                    "aux_targets": {
                        "room_type": 0,
                        "danger": 0.0,
                        "next_dest": 8,
                        "reachable": 0.0,
                    },
                }
            )
        return batch

    def test_next_state_loss_runs_inside_a_ppo_update(self):
        trainer = self._make_trainer()
        metrics = trainer._update(self._synthetic_batch())

        self.assertIsNotNone(metrics)
        self.assertIn("next_state_loss", metrics)
        self.assertIn("next_state_baseline_var", metrics)
        self.assertGreater(metrics["next_state_loss"], 0.0)
        self.assertTrue(np.isfinite(metrics["next_state_loss"]))
        self.assertGreater(metrics["next_state_baseline_var"], 0.0)
        self.assertGreater(metrics["next_state_ratio"], 0.0)

    def test_head_disabled_without_coefficient(self):
        observation_space = _make_observation_space()
        policy = IsaacPolicy(observation_space, [9, 9, 2, 2, 2])
        self.assertFalse(policy.next_state_enabled)
        self.assertNotIn("aux_next_state", dict(policy.named_parameters()))

        value, logits, _hidden, aux = policy(
            {
                key: torch.as_tensor(space.sample()).unsqueeze(0)
                for key, space in observation_space.spaces.items()
            }
        )
        self.assertIsNone(aux)

    def test_legacy_checkpoint_loads_with_fresh_head(self):
        legacy = IsaacPolicy(_make_observation_space(), [9, 9, 2, 2, 2])
        grown = IsaacPolicy(
            _make_observation_space(),
            [9, 9, 2, 2, 2],
            aux_coef=0.05,
            next_state_coef=0.05,
        )

        adapted = LivePPOTrainer._adapt_state_dict(
            {k: v.clone() for k, v in legacy.state_dict().items()},
            grown.state_dict(),
        )
        # strict=False semantics: shared trunk restored, all new aux heads
        # (C5 + next_state) keep their fresh init.
        missing = set(grown.state_dict()) - set(adapted)
        self.assertTrue(
            {"aux_next_state.weight", "aux_next_state.bias"}.issubset(missing)
        )
        grown.load_state_dict(adapted, strict=False)


if __name__ == "__main__":
    unittest.main()
