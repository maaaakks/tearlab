import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(r'C:\Users\maxbr\Desktop\game\IsaacRL\binding-of-ai')
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
import torch
try:
    from research.campaign_core import *
except ModuleNotFoundError:
    from campaign_core import *
from physics_v2 import NativePhysics


class TaskContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.guard=patch('torch.cuda._lazy_init',side_effect=AssertionError('CPU contract initialized CUDA'))
        cls.guard.start()

    @classmethod
    def tearDownClass(cls):cls.guard.stop()

    def env(self,budget=3):
        return TaskEnv([Scene()],7,NativePhysics(),'cpu',budget=budget,optimizations=Optimizations())

    def test_health_immunity_death_and_terminal_snapshot(self):
        env=self.env()
        env.pos.fill_(100);env.targets[:,0].fill_(100)
        env.attack_timer.fill_(1000000)
        action=torch.zeros(1,dtype=torch.long)
        _,_,done,info=env.step(action)
        self.assertFalse(done[0]);self.assertEqual(env.health.item(),2.)
        env.step(action)
        self.assertEqual(env.health.item(),2.)
        env.immunity.zero_();env.step(action)
        self.assertEqual(env.health.item(),1.)
        env.immunity.zero_();obs,_,done,info=env.step(action)
        self.assertTrue(done[0]);self.assertTrue(info['dead'][0]);self.assertFalse(info['success'][0])
        self.assertEqual(info['episode_hits'].item(),3.)
        self.assertEqual(info['terminal_observation'][0,236].item(),0.)
        self.assertEqual(obs[0,236].item(),1.)
        self.assertEqual(env.episode_hits.item(),0.)

    def test_nohit_budget_ends_at_first_contact(self):
        env=self.env(1);env.pos.fill_(100);env.targets[:,0].fill_(100)
        _,_,done,info=env.step(torch.zeros(1,dtype=torch.long))
        self.assertTrue(done[0]);self.assertTrue(info['dead'][0])
        self.assertEqual(info['episode_hits'].item(),1.)

    def test_reset_hp_practice_and_dynamic_scene_configuration(self):
        env=self.env();env.set_scenes([Scene(shooter_hp=2,attacks=False)])
        self.assertEqual(env.hp[0,0].item(),2)
        self.assertGreater(env.attack_timer.min().item(),10000)
        env.set_scenes([Scene(1,1,4,8)])
        self.assertEqual(env.hp.tolist(),[[8,4,0,0]])
        self.assertEqual(env.initial_hp.item(),12)
        self.assertEqual(env.obs().shape,(1,238))

    def test_win_after_damage_is_not_nohit_and_resets_potential(self):
        env=self.env();env.health.fill_(2);env.episode_hits.fill_(1);env.hp.zero_();env.potential.fill_(.5)
        _,_,done,info=env.step(torch.zeros(1,dtype=torch.long))
        self.assertTrue(done[0]);self.assertTrue(info['success'][0]);self.assertFalse(info['nohit'][0])
        self.assertEqual(env.potential.item(),0.)

    def test_finite_timeout_has_failure_cost_and_terminal_snapshot(self):
        env=self.env();env.steps.fill_(399)
        obs,reward,done,info=env.step(torch.zeros(1,dtype=torch.long))
        self.assertTrue(done[0]);self.assertTrue(info['timeout'][0]);self.assertLess(reward.item(),-1.99)
        self.assertEqual(env.steps.item(),0)
        self.assertEqual(info['terminal_observation'][0,7].item(),1.)

    def test_potential_discounted_telescoping_on_all_outcomes(self):
        gamma=.995
        for outcome in ('win','dead','timeout'):
            totals={}
            for mode in ('potential','sparse'):
                total=0.
                for t,(before,after) in enumerate([(0.,.2),(.2,.6),(.6,1. if outcome=='win' else .6)]):
                    flag=lambda wanted:torch.tensor([t==2 and outcome==wanted])
                    reward,_=task_math(flag('win'),flag('dead'),flag('timeout'),torch.tensor([False]),
                                      torch.tensor([before]),torch.tensor([after]),torch.tensor([after-before]),
                                      torch.ones(1),mode,gamma)
                    total+=gamma**t*reward.item()
                totals[mode]=total
            self.assertAlmostEqual(totals['potential'],totals['sparse'],places=5)
            if outcome!='win':self.assertLess(totals['potential'],0.)

    def test_entity_permutation_does_not_change_policy(self):
        torch.manual_seed(11);policy=Policy()
        x=torch.randn(5,238);previous=torch.tensor([0,1,2,3,45])
        y=x.clone()
        y[:,48:76]=x[:,48:76].reshape(5,4,7)[:,[2,0,3,1]].flatten(1)
        y[:,76:236]=x[:,76:236].reshape(5,32,5).flip(1).flatten(1)
        for a,b in zip(policy(x,previous),policy(y,previous)):
            torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5)

    def test_compositions_reserved_from_all_final_training_cells(self):
        self.assertFalse({(s.chasers,s.shooters) for s in old_id_cells()}&HOLDOUT)
        self.assertEqual({(s.chasers,s.shooters) for s in holdout_cells()},HOLDOUT)


@unittest.skipUnless(os.environ.get('RUN_ISAAC_GPU_TESTS')=='1','Explicit GPU opt-in required')
class GpuContracts(unittest.TestCase):
    def test_native_matches_tensor_task_for_both_budgets(self):
        torch.set_num_threads(1)
        cache=Path(os.environ['ISAAC_NATIVE_CACHE'])
        native=NativePhysics.load(ROOT/'data/physics_v2/calibration_r5.json')
        for budget in (1,3):
            scenes=balance([Scene(shooter_hp=2,attacks=False),Scene(1,1,4),Scene(0,2)],16)
            fast,fe,_=make_env(scenes,33,native,'cuda',cache,budget=budget)
            slow,se,_=make_env(scenes,33,native,'cuda',cache,True,budget=budget)
            for e in (fe,se):e.steps.fill_(398)
            for t in range(20):
                action=(torch.arange(16,device='cuda')+t*13)%45
                x=slow.step(action);y=fast.step(action)
                for a,b in zip(x[:3],y[:3]):
                    torch.testing.assert_close(a,b,atol=3e-4,rtol=1e-5)
                for k in x[3]:torch.testing.assert_close(x[3][k],y[3][k],atol=3e-4,rtol=1e-5,msg=k)
                for k in ('pos','hp','health','episode_hits','potential'):
                    torch.testing.assert_close(getattr(se,k),getattr(fe,k),atol=3e-4,rtol=1e-5,msg=k)
            self.assertTrue(torch.equal(fe.rng.get_state(),se.rng.get_state()))


if __name__=='__main__':unittest.main(verbosity=2)
