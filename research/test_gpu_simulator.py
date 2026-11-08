"""Behavioral parity, ownership and reset regression checks; no training."""
import unittest
import torch
from research.rapid_physics import StrictNativeShooterEnv, StrictCombatSuiteEnv
from physics_v2 import NativePhysics
from research.gpu_simulator import (GraphReadyNativeEnv, GraphReadyCombatSuiteEnv,
                                    CudaGraphSimulator)


class ResetParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cpu_reset_and_step_sequence(self):
        for mixed in (False, True):
            args = ([i % 13 for i in range(9)], 'cpu', 17, NativePhysics()) if mixed else (9, 'cpu', 17)
            a = (StrictCombatSuiteEnv if mixed else StrictNativeShooterEnv)(*args)
            b = (GraphReadyCombatSuiteEnv if mixed else GraphReadyNativeEnv)(*args)
            for i in range(25):
                actions = (torch.arange(9) + 11 * i) % 45
                compare_outputs(self, a.step(actions), b.step(actions))
                compare_state(self, a, b)


def compare_outputs(test, a, b):
    for x, y in zip(a[:3], b[:3]):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    test.assertEqual(a[3].keys(), b[3].keys())
    for k in a[3]:
        torch.testing.assert_close(a[3][k], b[3][k], atol=0, rtol=0)


def compare_state(test, a, b):
    for k, v in vars(a).items():
        if isinstance(v, torch.Tensor):
            torch.testing.assert_close(v, getattr(b, k), atol=0, rtol=0,
                                       msg=lambda s: f'{k}: {s}')


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class GraphParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def pair(self, mixed=False, keep_terminal=False, n=64, diagnostic=False):
        args = ([i % 13 for i in range(n)], 'cuda', 29, NativePhysics()) if mixed else (n, 'cuda', 29)
        kw = {} if mixed else dict(enemies=4, diagnostic=diagnostic)
        a = (StrictCombatSuiteEnv if mixed else StrictNativeShooterEnv)(*args, **kw)
        b = (GraphReadyCombatSuiteEnv if mixed else GraphReadyNativeEnv)(*args, **kw)
        for e in (a, b):
            e.keep_terminal = keep_terminal
            e.p.max_steps = 7  # Force repeated asynchronous resets and timeouts.
        graph = CudaGraphSimulator(b)
        return a, b, graph

    def trajectory(self, mixed=False, keep_terminal=False, diagnostic=False):
        a, b, graph = self.pair(mixed, keep_terminal, diagnostic=diagnostic)
        for i in range(50):
            actions = (torch.arange(a.n, device='cuda') + i * 17) % 45
            compare_outputs(self, a.step(actions), graph.step(actions))
            compare_state(self, a, b)
        self.assertTrue(torch.equal(a.rng.get_state(), b.rng.get_state()))

    def test_shooters_autoreset(self):
        self.trajectory()

    def test_mixed_autoreset(self):
        self.trajectory(mixed=True)

    def test_shooters_terminal_mode(self):
        self.trajectory(keep_terminal=True)

    def test_mixed_terminal_mode(self):
        self.trajectory(mixed=True, keep_terminal=True)

    def test_diagnostic_damage_does_not_end_episode(self):
        self.trajectory(diagnostic=True)

    def test_first_tick_terminal_and_no_post_terminal_kill(self):
        a, b, graph = self.pair(keep_terminal=True, n=2)
        for e in (a, b):
            e.attack_timer.fill_(10000)
            e.pos[:] = torch.tensor([260., 140.], device='cuda')
            e.targets[:] = torch.tensor([400., 140.], device='cuda')
            e.hp.zero_()
            e.hp[0, 0] = 1
            e.bpos[0, 0] = torch.tensor([382., 140.], device='cuda')
            e.bvel.zero_()
            e.life[0, 0] = .5
            e.hp[1, :2] = 1
            e.targets[1, 0] = e.pos[1]
            e.bpos[1, 0] = torch.tensor([350., 140.], device='cuda')
            e.bvel[1, 0] = torch.tensor([600., 0.], device='cuda')
            e.life[1, 0] = .5
        actions = torch.tensor([20, 0], device='cuda')
        x = a.step(actions)
        y = graph.step(actions)
        compare_outputs(self, x, y)
        compare_state(self, a, b)
        self.assertEqual(y[3]['kills'].tolist(), [1., 0.])
        self.assertEqual(b.hp[1, 1].item(), 1)

    def test_outputs_remain_valid_after_next_step(self):
        _, _, g = self.pair()
        actions = torch.arange(g.n, device='cuda') % 45
        out = g.step(actions)
        saved = (out[0].clone(), out[1].clone(), out[2].clone(),
                 {k: v.clone() for k, v in out[3].items()})
        g.step((actions + 19) % 45)
        compare_outputs(self, out, saved)

    def test_explicit_masked_reset_preserves_rng_and_graph_inputs(self):
        a, b, g = self.pair(mixed=True, keep_terminal=True)
        actions = torch.arange(a.n, device='cuda') % 45
        for i in range(8):
            compare_outputs(self, a.step(actions), g.step(actions))
            mask = (actions + i) % 3 == 0
            a.keep_terminal = False
            a.reset(mask)
            a.keep_terminal = True
            g.reset(mask)
            compare_state(self, a, b)

    def test_rebound_state_and_changed_config_rejected(self):
        _, b, g = self.pair()
        actions = torch.zeros(g.n, device='cuda', dtype=torch.long)
        b.pos = b.pos.clone()
        with self.assertRaisesRegex(RuntimeError, 'rebound'):
            g.step(actions)
        b.pos = g._slots['pos']
        b.keep_terminal = True
        with self.assertRaisesRegex(RuntimeError, 'configuration'):
            g.step(actions)
        b.keep_terminal = False
        with self.assertRaises(ValueError):
            g.step(actions.float())


if __name__ == '__main__':
    unittest.main(verbosity=2)
