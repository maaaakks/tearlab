"""Tests for C2/C3 entity-set encoder and observation fusion."""

import unittest

import numpy as np
import torch
from gymnasium import spaces

from entity_encoder import EntitySetEncoder
from network import IsaacFeatureExtractor


class EntitySetEncoderTests(unittest.TestCase):
    def test_encoder_is_permutation_invariant(self):
        # The same set in a different order must produce the same pooled
        # representation (up to floating-point noise).
        encoder = EntitySetEncoder(4, hidden_dim=16, num_heads=2)
        a = torch.randn(1, 8, 4)
        b = a[:, torch.tensor([3, 0, 7, 1, 6, 2, 5, 4]), :]
        with torch.no_grad():
            out_a = encoder(a)
            out_b = encoder(b)
        self.assertTrue(torch.allclose(out_a, out_b, atol=1e-5))

    def test_encoder_handles_variable_sets_in_fixed_table(self):
        encoder = EntitySetEncoder(4, hidden_dim=16, num_heads=2)
        full = torch.randn(2, 32, 4)
        with torch.no_grad():
            out_full = encoder(full)
        self.assertEqual(out_full.shape, (2, 16))


class FeatureExtractorEntityTests(unittest.TestCase):
    def test_extractor_fuses_entities(self):
        obs_space = spaces.Dict({
            "grid": spaces.Box(0.0, 1.0, (10, 7, 13), dtype=np.float32),
            "player": spaces.Box(-np.inf, np.inf, (22,), dtype=np.float32),
            "entities": spaces.Box(-1.0, 1.0, (32, 16), dtype=np.float32),
        })
        extractor = IsaacFeatureExtractor(obs_space, 256)
        obs = {
            "grid": torch.zeros(1, 10, 7, 13),
            "player": torch.zeros(1, 22),
            "entities": torch.zeros(1, 32, 16),
        }
        out = extractor(obs)
        self.assertEqual(out.shape, (1, 256))
        self.assertTrue(extractor.has_entities)

    def test_extractor_without_entities_keeps_old_keys(self):
        # Old observation space (no entities) must produce a state_dict with
        # exactly the pre-C keys — no entity_encoder.* parameters.
        obs_space = spaces.Dict({
            "grid": spaces.Box(0.0, 1.0, (10, 7, 13), dtype=np.float32),
            "player": spaces.Box(-np.inf, np.inf, (22,), dtype=np.float32),
        })
        extractor = IsaacFeatureExtractor(obs_space, 256)
        keys = set(extractor.state_dict().keys())
        self.assertFalse(any(k.startswith("entity_encoder") for k in keys))

    def test_extractor_fuses_goal_and_decision_when_present(self):
        obs_space = spaces.Dict({
            "grid": spaces.Box(0.0, 1.0, (10, 7, 13), dtype=np.float32),
            "player": spaces.Box(-np.inf, np.inf, (22,), dtype=np.float32),
            "goal": spaces.Box(-np.inf, np.inf, (12,), dtype=np.float32),
            "decision": spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32),
        })
        extractor = IsaacFeatureExtractor(obs_space, 256)
        self.assertTrue(extractor.has_goal)
        self.assertTrue(extractor.has_decision)
        obs = {
            "grid": torch.zeros(1, 10, 7, 13),
            "player": torch.zeros(1, 22),
            "goal": torch.zeros(1, 12),
            "decision": torch.zeros(1, 8),
        }
        out = extractor(obs)
        self.assertEqual(out.shape, (1, 256))

    def test_extractor_without_goal_keeps_old_keys(self):
        obs_space = spaces.Dict({
            "grid": spaces.Box(0.0, 1.0, (10, 7, 13), dtype=np.float32),
            "player": spaces.Box(-np.inf, np.inf, (22,), dtype=np.float32),
        })
        extractor = IsaacFeatureExtractor(obs_space, 256)
        keys = set(extractor.state_dict().keys())
        self.assertFalse(any(k.startswith("goal_fc") for k in keys))
        self.assertFalse(any(k.startswith("decision_fc") for k in keys))


if __name__ == "__main__":
    unittest.main()
