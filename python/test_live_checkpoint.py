import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from gymnasium import spaces

from config import Config
from live_checkpoint import LiveCheckpointManager
from live_trainer import LivePPOTrainer


class DummyEnv:
    def __init__(self):
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(0.0, 1.0, shape=(10, 7, 13), dtype=np.float32),
                "player": spaces.Box(
                    -np.inf, np.inf, shape=(22,), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.MultiDiscrete([9, 9])
        self.metrics = {}

    def update_training_metrics(self, **metrics):
        self.metrics.update(metrics)


class LiveCheckpointTests(unittest.TestCase):
    def test_round_trip_restores_policy_optimizer_and_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_base = Path(temp_dir) / "checkpoints"
            config = Config()
            config.wandb.run_name = "checkpoint-test"

            with patch(
                "live_checkpoint.get_checkpoint_dir",
                return_value=checkpoint_base,
            ):
                trainer = LivePPOTrainer(config, DummyEnv())
                trainer.collector.steps = 12_345
                trainer.update_count = 7

                loss = sum(
                    parameter.square().mean()
                    for parameter in trainer.policy.parameters()
                )
                trainer.optimizer.zero_grad()
                loss.backward()
                trainer.optimizer.step()

                expected_parameters = {
                    key: value.detach().clone()
                    for key, value in trainer.policy.state_dict().items()
                }
                saved_path = trainer.save("periodic")

                resumed = LivePPOTrainer(
                    config,
                    DummyEnv(),
                    resume=str(saved_path),
                )

                self.assertEqual(resumed.collector.steps, 12_345)
                self.assertEqual(resumed.update_count, 7)
                self.assertEqual(resumed.checkpoints.run_dir, saved_path.parent)
                self.assertTrue(resumed.optimizer.state_dict()["state"])
                for key, expected in expected_parameters.items():
                    self.assertTrue(
                        torch.equal(resumed.policy.state_dict()[key], expected),
                        key,
                    )

                latest = LiveCheckpointManager.resolve_checkpoint(
                    "latest",
                    checkpoint_base / "live",
                )
                self.assertEqual(latest, saved_path)

    def test_entropy_schedule_decays_to_final_value(self):
        config = Config()
        config.train.total_timesteps = 1000
        config.train.ent_coef = 0.02
        config.train.ent_coef_final = 0.005
        config.train.entropy_decay_fraction = 0.8

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "live_checkpoint.get_checkpoint_dir",
                return_value=Path(temp_dir),
            ):
                trainer = LivePPOTrainer(config, DummyEnv())

        self.assertAlmostEqual(trainer._current_ent_coef(0), 0.02)
        self.assertAlmostEqual(trainer._current_ent_coef(400), 0.0125)
        self.assertAlmostEqual(trainer._current_ent_coef(800), 0.005)
        self.assertAlmostEqual(trainer._current_ent_coef(1000), 0.005)

    def test_update_reports_policy_exploration_and_feature_health(self):
        config = Config()
        config.train.n_epochs = 1
        config.train.batch_size = 4
        config.train.target_kl = None

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "live_checkpoint.get_checkpoint_dir",
                return_value=Path(temp_dir),
            ):
                trainer = LivePPOTrainer(config, DummyEnv())

                observation = {
                    "grid": np.zeros((10, 7, 13), dtype=np.float32),
                    "player": np.zeros(22, dtype=np.float32),
                }
                tensors = {
                    key: torch.as_tensor(value).unsqueeze(0)
                    for key, value in observation.items()
                }
                with torch.no_grad():
                    value, move_logits, shoot_logits = trainer.policy(tensors)

                batch = []
                for index in range(8):
                    action = (index % 9, (index * 2) % 9)
                    move_dist = torch.distributions.Categorical(logits=move_logits)
                    shoot_dist = torch.distributions.Categorical(logits=shoot_logits)
                    logprob = (
                        move_dist.log_prob(torch.tensor([action[0]]))
                        + shoot_dist.log_prob(torch.tensor([action[1]]))
                    )
                    batch.append(
                        {
                            "obs": {key: val.copy() for key, val in observation.items()},
                            "action": action,
                            "logprob": float(logprob.item()),
                            "value": float(value.item()),
                            "reward": 1.0,
                            "done": index == 7,
                            "next_value": 0.0,
                        }
                    )

                metrics = trainer._update(batch)

        for key in (
            "clip_fraction",
            "explained_variance",
            "dead_feature_fraction",
            "move_action_entropy",
            "shoot_action_entropy",
            "grad_norm",
            "ent_coef",
        ):
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]), key)


if __name__ == "__main__":
    unittest.main()
