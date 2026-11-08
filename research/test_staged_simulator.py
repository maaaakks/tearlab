"""CPU qualification by default. GPU tests need RUN_ISAAC_GPU_TESTS=1 explicitly."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from research.staged_simulator import StagedNativeEnv, StagedCombatSuiteEnv, Optimizations, ALL, STATE
from research.gpu_ablation import VARIANTS, assert_pair, plan, qualify
from research.rapid_physics import StrictNativeShooterEnv, StrictCombatSuiteEnv
from research.native_runtime import build_native
from physics_v2 import NativePhysics

CACHE = Path(os.environ.get('ISAAC_NATIVE_CACHE', Path(__file__).parent/'native_build'))


class CPUQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.guard = patch('torch.cuda._lazy_init', side_effect=AssertionError('GPU forbidden during CPU qualification'))
        cls.guard.start()
        cls.metrics = []

    @classmethod
    def tearDownClass(cls):
        cls.guard.stop()
        destination = os.environ.get('ISAAC_CPU_REPORT')
        if destination:
            Path(destination).write_text(json.dumps(dict(gpu_executed=False, variants=cls.metrics), indent=2), encoding='utf8')

    def pair(self, options=ALL, mixed=False, n=16, seed=48, diagnostic=False, timeout=7, native=None):
        args = ([i%13 for i in range(n)], 'cpu', seed, native or NativePhysics()) if mixed else (n, 'cpu', seed)
        kwargs = {} if mixed else dict(enemies=4, diagnostic=diagnostic, native=native or NativePhysics())
        a = (StrictCombatSuiteEnv if mixed else StrictNativeShooterEnv)(*args, **kwargs)
        b = (StagedCombatSuiteEnv if mixed else StagedNativeEnv)(*args, **kwargs, optimizations=options, cache=CACHE)
        a.p.max_steps = b.p.max_steps = timeout
        b.prepare()
        return a, b

    def test_independent_add_remove_and_fusion_matrix(self):
        for mixed in (False, True):
            for name, options in VARIANTS.items():
                if options is None: continue
                with self.subTest(mixed=mixed, variant=name):
                    a, b = self.pair(options, mixed)
                    errors = {}
                    for t in range(32):
                        actions = (torch.arange(a.n)+t*17) % 45
                        x, y = a.step(actions), b.step(actions)
                        # Tensor-only control/masking must remain bit exact.
                        exact = name in ('staged_control', 'terminal_mask')
                        assert_pair(a, b, x, y, 0 if exact else 3e-4, 0 if exact else 1e-5, errors)
                    self.assertTrue(torch.equal(a.rng.get_state(), b.rng.get_state()))
                    self.metrics.append(dict(mixed=mixed, variant=name, lanes=a.n, steps=32,
                                             discrete_exact=True, rng_exact=True,
                                             max_abs_by_field=errors, max_abs=max(errors.values(), default=0)))

    def test_terminal_contacts_and_first_tick_clear(self):
        for options in (Optimizations(), Optimizations(terminal_mask=True), ALL):
            a, b = self.pair(options, n=2)
            for e in (a, b):
                e.keep_terminal = True
                e.attack_timer.fill_(10000)
                e.pos[:] = torch.tensor([260., 140.])
                e.targets[:] = torch.tensor([400., 140.])
                e.hp.zero_(); e.hp[0, 0] = 1
                e.bpos[0, 0] = torch.tensor([382., 140.]); e.bvel.zero_(); e.life[0, 0] = .5
                e.hp[1, :2] = 1; e.targets[1, 0] = e.pos[1]
                e.bpos[1, 0] = torch.tensor([350., 140.]);e.bvel[1, 0] = torch.tensor([600., 0.]);e.life[1, 0] = .5
            action = torch.tensor([20, 0])
            x, y = a.step(action), b.step(action)
            assert_pair(a, b, x, y, 3e-4, 1e-5, {})
            self.assertEqual(y[3]['kills'].tolist(), [1., 0.])
            self.assertEqual(b.hp[1, 1].item(), 1)

    def test_sparse_reset_none_partial_all_and_velocity_side_effect(self):
        a, b = self.pair(Optimizations(sparse_reset=True), mixed=True, n=9)
        a.attacks = a.attacks.long()
        for mode in ('none', 'partial', 'all', 'none'):
            mask = torch.zeros(9, dtype=torch.bool) if mode == 'none' else (torch.arange(9)%2 == 0 if mode == 'partial' else torch.ones(9, dtype=torch.bool))
            # Even an empty reset must refresh suite target velocity, not skip it.
            for e in (a, b): e.target_vel.fill_(123)
            a.reset(mask);b.reset(mask)
            for name in STATE:
                x, y = getattr(a, name), getattr(b, name)
                torch.testing.assert_close(x, y, atol=3e-4 if x.is_floating_point() else 0,
                                           rtol=1e-5 if x.is_floating_point() else 0)
            self.assertTrue(torch.equal(a.rng.get_state(), b.rng.get_state()))
            self.assertFalse((b.target_vel == 123).any())

    def test_wall_orders_and_command_delays(self):
        for order in (1, 2, 3, 4):
            for delay in (0, 1, 3):
                a, b = self.pair(ALL, n=2, native=NativePhysics(wall_update_order=order, command_delay_ticks=delay))
                for e in (a, b):
                    e.pos[:] = torch.tensor([[10.0002, 140.], [505.7039, 193.5712]])
                    e.native_velocity[:] = torch.tensor([[.00073421, 0.], [4.2459, .06814]])
                for t in range(6):
                    action = torch.tensor([15, 20]) if t<3 else torch.tensor([0, 0])
                    assert_pair(a, b, a.step(action), b.step(action), 3e-4, 1e-5, {})

    def test_overlapping_projectiles_ties_and_full_buffers(self):
        a, b = self.pair(Optimizations(projectiles=True), n=1)
        for e in (a, b):
            e.keep_terminal = True; e.attack_timer.fill_(10000)
            e.hp.fill_(20); e.targets[:] = torch.tensor([300., 140.])
            e.bpos[:] = torch.tensor([300., 140.]);e.bvel.zero_();e.life.fill_(.5)
            e.hostile_pos[:] = e.pos[:, None];e.hostile_vel.zero_();e.hostile_life.fill_(.5)
        action = torch.tensor([3])
        x, y = a.step(action), b.step(action)
        assert_pair(a, b, x, y, 3e-4, 1e-5, {})
        self.assertEqual(y[3]['damage'].item(), 8)

    def test_diagnostic_and_output_ownership(self):
        a, b = self.pair(ALL, diagnostic=True)
        action = torch.arange(a.n) % 45
        for e in (a, b): e.targets[:, 0] = e.pos
        x, y = a.step(action), b.step(action)
        assert_pair(a, b, x, y, 3e-4, 1e-5, {})
        frozen = (y[0].clone(), y[1].clone(), y[2].clone(), {k:v.clone() for k,v in y[3].items()})
        b.step(action)
        for i in range(3):torch.testing.assert_close(y[i], frozen[i], atol=0, rtol=0)
        for k in y[3]:torch.testing.assert_close(y[3][k], frozen[3][k], atol=0, rtol=0)

    def test_option_dependencies_and_configuration_guard(self):
        with self.assertRaises(ValueError):Optimizations(tick_fusion=True)
        with self.assertRaises(ValueError):Optimizations(decision_fusion=True)
        _, b = self.pair()
        b.p.max_steps += 1
        with self.assertRaisesRegex(RuntimeError, 'configuration'):
            b.step(torch.zeros(b.n, dtype=torch.long))
        with self.assertRaises(AttributeError):b.optimizations = Optimizations()

    def test_native_pointer_abi_rejects_invalid_inputs(self):
        _, b = self.pair()
        with self.assertRaises(ValueError):b.step(torch.zeros(b.n))
        with self.assertRaises(ValueError):b.step(torch.zeros(b.n+1, dtype=torch.long))
        with self.assertRaises(ValueError):b.reset(torch.zeros(b.n))
        c = StagedNativeEnv(2, 'cpu', 1, optimizations=ALL, cache=CACHE)
        c.bpos = c.bpos.double()
        with self.assertRaises(TypeError):c.prepare()

    def test_plan_does_not_touch_cuda(self):
        args = argparse.Namespace(only=None, scenarios=['shooters','mixed'], reset_modes=['natural','frequent'],
                                  lanes=[512,8192], seeds=[29], repeats=5, steps=64, warmup=8,
                                  verify_steps=64, atol=3e-4, rtol=1e-5)
        p = plan(args)
        self.assertFalse(p['gpu_executed'])
        self.assertEqual(p['status'], 'pending_gpu_validation')
        self.assertEqual(p['cases'], [])
        self.assertIn('fused_decision_warp', p['variants'])


@unittest.skipUnless(os.environ.get('RUN_ISAAC_GPU_TESTS') == '1', 'Explicit GPU authorization required')
class GPUQualification(unittest.TestCase):
    def test_every_variant_on_partial_warp_and_both_suites(self):
        args = argparse.Namespace(cache=CACHE, verify_steps=32, atol=3e-4, rtol=1e-5)
        for scenario in ('shooters', 'mixed'):
            for name in VARIANTS:
                with self.subTest(scenario=scenario, name=name):
                    result = qualify(name, scenario, 33, 481, 'frequent', args)
                    self.assertEqual(result['status'], 'passed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
