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
        self.action_space = spaces.MultiDiscrete([9, 9, 2, 2, 2])
        self.metrics = {}

    def update_training_metrics(self, **metrics):
        self.metrics.update(metrics)


class EntitiesEnv(DummyEnv):
    """Like DummyEnv but with the identity-bearing entity table observation."""

    def __init__(self):
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(0.0, 1.0, shape=(10, 7, 13), dtype=np.float32),
                "player": spaces.Box(
                    -np.inf, np.inf, shape=(22,), dtype=np.float32
                ),
                "entities": spaces.Box(-1.0, 1.0, shape=(32, 16), dtype=np.float32),
            }
        )
        self.action_space = spaces.MultiDiscrete([9, 9, 2, 2, 2])
        self.metrics = {}


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

    def test_resume_with_added_entities_observation(self):
        """A pre-entities checkpoint must load once the entities branch is
        added: overlapping fc.0 columns are copied, the new columns keep fresh
        init, and the optimizer state is reset (parameter set changed)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_base = Path(temp_dir) / "checkpoints"
            config = Config()
            config.wandb.run_name = "checkpoint-entities-add"

            with patch(
                "live_checkpoint.get_checkpoint_dir",
                return_value=checkpoint_base,
            ):
                trainer = LivePPOTrainer(config, DummyEnv())
                # Populate optimizer state on a real policy param so we can
                # verify it gets reset on resume.
                loss = trainer.policy.features.fc[0].weight.square().mean()
                trainer.optimizer.zero_grad()
                loss.backward()
                trainer.optimizer.step()
                self.assertTrue(trainer.optimizer.state_dict()["state"])
                saved_path = trainer.save("periodic")
                old_fc = trainer.policy.state_dict()["features.fc.0.weight"]
                old_fc_in = old_fc.shape[1]

            with patch(
                "live_checkpoint.get_checkpoint_dir",
                return_value=checkpoint_base,
            ):
                resumed = LivePPOTrainer(config, EntitiesEnv(), resume=str(saved_path))
                new_state = resumed.policy.state_dict()

                # Input dim grew by exactly the entity-encoder width (64).
                new_fc = new_state["features.fc.0.weight"]
                self.assertEqual(new_fc.shape[1], old_fc_in + 64)
                # Overlapping columns preserved, new columns finite (fresh init).
                self.assertTrue(torch.equal(new_fc[:, :old_fc_in], old_fc))
                self.assertTrue(torch.isfinite(new_fc).all())
                # Entity branch exists with freshly initialized weights.
                self.assertTrue(
                    any("entity_encoder" in key for key in new_state)
                )
                # Optimizer state was reset because the parameter set changed.
                self.assertFalse(resumed.optimizer.state_dict()["state"])

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
                    value, logits = trainer.policy(tensors)

                batch = []
                for index in range(8):
                    action = (index % 9, (index * 2) % 9, index % 2, 0, index % 2)
                    logprob = torch.zeros(())
                    for k, head_logits in enumerate(logits):
                        dist = torch.distributions.Categorical(logits=head_logits)
                        logprob = logprob + dist.log_prob(
                            torch.tensor([action[k]])
                        ).squeeze()
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
