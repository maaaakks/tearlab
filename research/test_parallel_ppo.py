"""Explicit GPU regression tests for rollout capture and training graphs."""
import os
from pathlib import Path
import sys
import unittest

ROOT=(Path(__file__).resolve().parents[1] if Path(__file__).parent.name=='research'
      else Path(r'C:\Users\maxbr\Desktop\game\IsaacRL\binding-of-ai'))
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
import torch
from research import pilot_10m as p
try:
    from research.parallel_ppo import ParallelArm, Options
    from research.benchmark_parallel import differences
except ModuleNotFoundError:
    from parallel_ppo import ParallelArm, Options
    from benchmark_parallel import differences
from research.staged_simulator import STATE


@unittest.skipUnless(os.environ.get('RUN_ISAAC_GPU_TESTS')=='1', 'Explicit GPU opt-in required')
class ParallelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(113)
        torch.backends.cuda.matmul.allow_tf32=False
        cls.native=p.NativePhysics.load(ROOT/'data/physics_v2/calibration_r5.json')
        cls.initial=p.PilotPolicy().cuda()
        cls.cache=Path(os.environ.get('ISAAC_NATIVE_CACHE',Path(__file__).resolve().parent/'native_build'))

    def make(self, options=Options(True,True,False,False)):
        return ParallelArm(self.initial,p.balanced_training(128),23,self.native,
                           'cuda',self.cache,True,options=options)

    def test_capture_preserves_initial_state_and_rng_then_three_updates(self):
        reference=p.Arm(self.initial,p.balanced_training(128),23,self.native,
                        'cuda',self.cache,True)
        fast=self.make()
        # Scratch tensors intentionally have undefined initial contents.
        for k in STATE:
            torch.testing.assert_close(getattr(reference.env,k),getattr(fast.env,k),atol=0,rtol=0,msg=k)
        for key in ('action_rng','shuffle_rng'):
            self.assertTrue(torch.equal(getattr(reference,key).get_state(),getattr(fast,key).get_state()))
        self.assertTrue(torch.equal(reference.env.rng.get_state(),fast.env.rng.get_state()))
        self.assertEqual(float(fast.totals.abs().sum()),0.)
        self.assertEqual(int(fast.action_counts.sum()),0)
        for i in range(3):
            if i==1:
                # Force actual timeout observations, including PPO bootstrapping.
                reference.env.steps.fill_(398);fast.env.steps.fill_(398)
            left=reference.update();right=fast.update()
            gate=differences(reference,fast)
            self.assertEqual(gate['max_tensor_abs'],0.)
            self.assertEqual(gate['max_parameter_abs'],0.)
            self.assertEqual(left['optimizer_steps'],right['optimizer_steps'])

    def test_kl_stop_does_not_step_optimizer_and_next_update_works(self):
        fast=self.make()
        original=fast._train_policy
        def large_kl(obs,previous):
            logits,value=original(obs,previous)
            return logits+torch.linspace(-4,4,45,device='cuda'),value
        fast._train_policy=large_kl
        before=[x.detach().clone() for x in fast.policy.parameters()]
        stats=fast.update()
        self.assertEqual(stats['optimizer_steps'],0)
        self.assertGreater(stats['max_kl'],.02)
        for a,b in zip(before,fast.policy.parameters()):torch.testing.assert_close(a,b,atol=0,rtol=0)
        fast._train_policy=original
        self.assertGreater(fast.update()['optimizer_steps'],0)

    def test_configuration_and_minibatch_changes_rejected(self):
        fast=self.make()
        with self.assertRaisesRegex(ValueError,'minibatch'):fast.update(minibatch=128)
        parameter=fast.policy.actor.bias
        fast.policy.actor.bias=torch.nn.Parameter(parameter.detach().clone())
        with self.assertRaisesRegex(RuntimeError,'parameter storage'):fast.update()
        fast.policy.actor.bias=parameter
        fast.env.p.max_steps+=1
        with self.assertRaisesRegex(RuntimeError,'configuration'):fast.update()

    def test_fused_optimizer_contract_and_checkpoint_reload(self):
        fast=self.make(Options(True,True,True,False))
        stats=fast.update()
        self.assertGreater(stats['optimizer_steps'],0)
        state={k:v.detach().clone() for k,v in fast.policy.state_dict().items()}
        eager=p.PilotPolicy().cuda()
        eager.load_state_dict(state)
        x=fast.batch['obs'][0];previous=fast.batch['previous'][0]
        with torch.no_grad():
            for a,b in zip(fast.policy(x,previous),eager(x,previous)):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertEqual(set(state),set(self.initial.state_dict()))


if __name__=='__main__':unittest.main(verbosity=2)
