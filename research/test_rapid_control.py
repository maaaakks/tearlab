"""Boundary, accounting and learning checks for the bounded experiment."""
import copy
import unittest

import torch

from research.rapid_control import Adapter, SequenceReplay, returns_and_endpoint, update
from research.rapid_model import RapidQ


class RapidControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def replay(self):
        replay = SequenceReplay(6, 2, 'cpu')
        for t in range(3):
            obs = torch.full((2, 2, 140), float(t))
            done = torch.tensor([t == 1, False])
            episode = torch.tensor([2 if t == 2 else 0, 1])
            replay.append(obs, torch.tensor([0, 0]), torch.tensor([1, 2]), episode, obs + 1,
                          torch.tensor([1., 10.]), done, done, torch.zeros(2, 4),
                          torch.ones(2, 3, dtype=torch.bool))
        return replay

    def test_sequences_keep_terminal_tail_and_never_cross_lane_or_reset(self):
        replay = self.replay()
        indices, valid = replay.sequences(torch.tensor([0, 1, 2, 4]), 4)
        self.assertEqual(valid.tolist(), [[True, True, False, False], [True, True, True, False],
                                         [True, False, False, False], [True, False, False, False]])
        self.assertEqual(indices[0, :2].tolist(), [0, 2])
        total, discount, last = returns_and_endpoint(replay, torch.tensor([0, 1]), 4, .5)
        torch.testing.assert_close(total, torch.tensor([1.5, 17.5]))
        torch.testing.assert_close(discount, torch.tensor([.25, .125]))
        self.assertEqual(last.tolist(), [2, 5])
        self.assertEqual(replay.terminal[last].tolist(), [True, False])

    def test_replay_copies_transition_and_enforces_budget(self):
        replay = self.replay()
        stored = replay.obs.clone()
        with self.assertRaises(ValueError):
            replay.append(None, None, torch.zeros(2), *(None for _ in range(7)))
        torch.testing.assert_close(stored, replay.obs)

    def test_event_mixture_importance_weights_match_exact_probabilities(self):
        replay = self.replay()
        replay.events[:6] = torch.tensor([True, False, False, False, False, False])
        indices, weights = replay.sample(10000, torch.Generator().manual_seed(8))
        expected = 1 / (6 * (.5 / 6 + .5 * replay.events[indices].float()))
        torch.testing.assert_close(weights, expected)
        self.assertAlmostEqual(float(weights.mean()), 1., delta=.03)
        self.assertAlmostEqual(float((weights * (indices == 0)).mean()), 1/6, delta=.02)

    def test_timeout_keeps_final_observation_and_command_for_bootstrap(self):
        adapter = Adapter('micro', 5, 77, 'cpu')
        adapter.env.steps[:] = 99
        adapter.env.attack_timer.fill_(10000)
        adapter.env.targets[:] = torch.tensor([400., 200.])
        actions = torch.full((5,), 20)
        final, reward, terminal, done, measurements, info = adapter.step(actions)
        self.assertTrue(done.all())
        self.assertFalse(terminal.any())
        self.assertTrue(info['timeout'].all())
        self.assertEqual(adapter.previous.tolist(), [45] * 5)
        self.assertFalse(torch.equal(final, adapter.history))
        self.assertEqual(adapter.reset_count, 10)
        self.assertEqual(measurements.shape, (5, 4))

    def test_td_and_auxiliary_update_from_real_simulator_sequences(self):
        adapter = Adapter('micro', 5, 9, 'cpu')
        replay = SequenceReplay(50, 5, 'cpu')
        for step in range(10):
            obs, previous, episode = adapter.history.clone(), adapter.previous.clone(), adapter.episodes.clone()
            actions = torch.tensor([0, 1, 2, 3, 4])
            final, reward, terminal, done, measurement, _ = adapter.step(actions)
            replay.append(obs, previous, actions, episode, final, reward, terminal, done,
                          measurement, torch.ones(5, 3, dtype=torch.bool))
        model = RapidQ()
        target = copy.deepcopy(model).requires_grad_(False)
        before = model.q_heads[0].advantage.weight.detach().clone()
        td, auxiliary, reads = update(model, target, torch.optim.Adam(model.parameters()), replay,
                                      torch.Generator().manual_seed(3), 8)
        self.assertGreater(reads, 0)
        self.assertTrue(torch.isfinite(torch.tensor([td, auxiliary])).all())
        self.assertFalse(torch.equal(before, model.q_heads[0].advantage.weight))
        self.assertTrue(all(p.grad is None for p in target.parameters()))


if __name__ == '__main__':
    unittest.main()
