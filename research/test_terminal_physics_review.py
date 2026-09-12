"""Regression contracts found during the simulator review.

These tests cover the opt-in fix for NativeShooterEnv's terminal-boundary bug.
Run from the repository root with the project's Python interpreter:
    python -m unittest research.test_terminal_physics_review -v

No historical simulator or checkpoint is modified by this test module.
"""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from physics_v2 import NativeShooterEnv, NativePhysics
from research.rapid_physics import StrictNativeShooterEnv, StrictCombatSuiteEnv


class TerminalPhysicsReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def environment(self):
        env = StrictNativeShooterEnv(1, "cpu", 1)
        env.keep_terminal = True
        env.pos[:] = torch.tensor([260., 140.])
        env.targets[:] = torch.tensor([400., 140.])
        env.attack_timer.fill_(10000)
        env.hp.zero_()
        return env

    def test_clear_on_first_tick_preserves_terminal_player_state(self):
        env = self.environment()
        env.hp[0, 0] = 1
        # Existing tear overlaps the only enemy on the first native tick.
        env.bpos[0, 0] = torch.tensor([382., 140.])
        env.bvel[0, 0] = 0
        env.life[0, 0] = .5
        start = env.pos.clone()

        # Delayed action means native commands [idle, right, right]. The clear
        # happens on idle; the remaining commands must not mutate final state.
        _, _, done, info = env.step(torch.tensor([20]))

        self.assertTrue(done.item())
        self.assertTrue(info["success"].item())
        self.assertEqual(info["kills"].item(), 1)
        torch.testing.assert_close(env.pos, start, atol=1e-6, rtol=0)
        torch.testing.assert_close(env.native_velocity, torch.zeros_like(env.native_velocity))

    def test_contact_on_first_tick_prevents_unreported_post_terminal_kill(self):
        env = self.environment()
        env.hp[0, :2] = 1
        env.targets[0, 0] = env.pos[0]  # first tick ends episode by contact
        # This tear reaches enemy 1 only on tick 2. It cannot score a kill in
        # the episode after contact has already ended it on tick 1.
        env.bpos[0, 0] = torch.tensor([350., 140.])
        env.bvel[0, 0] = torch.tensor([600., 0.])
        env.life[0, 0] = .5

        _, _, done, info = env.step(torch.tensor([0]))

        self.assertTrue(done.item())
        self.assertTrue(info["contact"].item())
        self.assertEqual(info["damage"].item(), 0)
        self.assertEqual(info["kills"].item(), 0)
        self.assertEqual(env.hp[0, 1].item(), 1,
                         "Enemy killed after terminal while info reports zero kills")
        torch.testing.assert_close(env.bpos[0, 0], torch.tensor([370., 140.]))

    def test_terminal_lane_resets_without_resetting_survivor(self):
        env = StrictNativeShooterEnv(2, "cpu", 1)
        env.attack_timer.fill_(10000)
        env.targets[:] = torch.tensor([400., 140.])
        env.pos[:] = torch.tensor([260., 140.])
        env.hp[0].zero_()
        _, _, done, info = env.step(torch.tensor([20, 20]))
        self.assertEqual(done.tolist(), [True, False])
        self.assertEqual(info["success"].tolist(), [True, False])
        self.assertGreater(env.hp[0].sum().item(), 0)
        self.assertEqual(env.steps.tolist(), [0, 1])
        self.assertEqual(env.last_action.tolist(), [0, 20])
        torch.testing.assert_close(env.native_velocity[0], torch.zeros(2))
        self.assertGreater(env.native_velocity[1, 0].item(), 0)
        torch.testing.assert_close(env.pos[1], torch.tensor([262.92395, 140.]), atol=1e-4, rtol=0)

    def test_nonterminal_native_dynamics_are_unchanged(self):
        baseline = NativeShooterEnv(2, "cpu", 3)
        strict = StrictNativeShooterEnv(2, "cpu", 3)
        for env in (baseline, strict):
            env.keep_terminal = True
            env.attack_timer.fill_(10000)
            env.pos[:] = torch.tensor([260., 140.])
            env.targets[:] = torch.tensor([400., 240.])
        for actions in (torch.tensor([21, 17]), torch.tensor([32, 24]), torch.tensor([0, 7])):
            expected = baseline.step(actions)
            actual = strict.step(actions)
            self.assertFalse(actual[2].any().item())
            for got, want in zip(actual[:3], expected[:3]):
                torch.testing.assert_close(got, want)
            for name in actual[3]:
                torch.testing.assert_close(actual[3][name], expected[3][name])

    def test_combat_suite_keeps_roles_and_uses_strict_step(self):
        env = StrictCombatSuiteEnv([0, 6, 12], "cpu", 8, NativePhysics())
        self.assertIs(type(env).step, StrictNativeShooterEnv.step)
        self.assertEqual(env.shooters.tolist(), [0, 2, 4])
        self.assertEqual(env.is_chaser.sum(1).tolist(), [3, 2, 0])
        env.keep_terminal = True
        env.hp.zero_()
        start = env.pos.clone()
        env.step(torch.tensor([20, 20, 20]))
        torch.testing.assert_close(env.pos, start)


if __name__ == "__main__":
    unittest.main()
