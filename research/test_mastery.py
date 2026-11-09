import gc
import os
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
import torch
try:from research.mastery_core import *
except ModuleNotFoundError:from mastery_core import *
from physics_v2 import NativePhysics


class ObjectiveContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_win_objective_and_shaping_rank_complete_trajectories(self):
        for result in ('success','dead','timeout'):
            totals={}
            for mode in ('win','potential'):
                total=0.
                for index,(before,after) in enumerate([(0.,.2),(.2,.5),(.5,.8)]):
                    info={name:torch.tensor([index==2 and name==result]) for name in ('success','dead','timeout')}
                    info['progress_fraction']=torch.tensor([after])
                    reward,_=objective(info,torch.tensor([before]),mode);total+=reward.item()
                totals[mode]=total
            expected=3 if result=='success' else -3
            self.assertAlmostEqual(totals['win'],expected,places=6)
            self.assertAlmostEqual(totals['potential'],expected,places=6)

    def test_legacy_reward_is_preserved(self):
        info=dict(success=torch.tensor([False]),dead=torch.tensor([True]),timeout=torch.tensor([False]),
                  reward_task=torch.tensor([-3.1]),reward_time=torch.tensor([-.002]),progress_fraction=torch.tensor([.8]))
        reward,parts=objective(info,torch.tensor([.5]),'legacy')
        self.assertAlmostEqual(reward.item(),-3.102,places=5);self.assertEqual(parts['progress'].item(),0.)

    def test_scenario_validity_and_closed_factors(self):
        scenes=[HardScene(0,k,pattern=p,placement=place) for k in (3,4) for p in ('surround','long','unequal') for place in ('free','edge','corner')]*16
        for width,height in GEOMETRIES+[(560,320),(720,400)]:
            env=HardEnv(scenes,123,NativePhysics(),'cpu',budget=1,width=width,height=height,optimizations=c.Optimizations())
            active=torch.arange(4)[None]<env.counts[:,None]
            self.assertTrue((((env.targets>=20)&(env.targets<=env.size-20)).all(-1)|~active).all())
            self.assertTrue((((env.targets-env.pos[:,None]).norm(dim=-1)>=79.99)|~active).all())
            for i in range(4):
                for j in range(i):self.assertTrue((((env.targets[:,i]-env.targets[:,j]).norm(dim=-1)>=34.99)|~(active[:,i]&active[:,j])).all())
        for hp in (4,8,12):
            cases=training_cases(hp)
            self.assertFalse({(a.scene.chasers,a.scene.shooters) for a in cases}&c.HOLDOUT)
            self.assertFalse(any(a.scene.pattern=='unequal' or (a.width,a.height) in [(560,320),(720,400)] for a in cases))

    def test_partial_group_reset_and_real_terminal(self):
        cases=[Case(HardScene(),440,240),Case(HardScene(),520,280)]
        env=MultiRoom(cases,41,NativePhysics(),None,'cpu',reference=True)
        saved=env.groups[1].pos.clone();env.reset(torch.tensor([True,False]))
        torch.testing.assert_close(env.groups[1].pos,saved,rtol=0,atol=0)
        for group in env.groups:group.steps.fill_(399)
        obs,reward,done,info=env.step(torch.zeros(2,dtype=torch.long))
        self.assertTrue(done.all());self.assertTrue(info['timeout'].all())
        self.assertTrue((info['terminal_observation'][:,7]==1).all());self.assertTrue((obs[:,7]==0).all())

    def test_contact_budget_changes_terminal_but_not_physics(self):
        for budget in (1,3):
            env=MultiRoom([Case(HardScene(1,0))],17,NativePhysics(),None,'cpu',reference=True,budget=budget)
            child=env.groups[0];child.targets[:,0].copy_(child.pos);child.target_vel.zero_()
            _,_,done,info=env.step(torch.zeros(1,dtype=torch.long))
            self.assertTrue(info['contact'].item())
            self.assertEqual(done.item(),budget==1)
            self.assertEqual(info['dead'].item(),budget==1)
            if budget==3:self.assertEqual(child.health.item(),2.)


@unittest.skipUnless(os.environ.get('RUN_ISAAC_GPU_TESTS')=='1','Explicit GPU test')
class GPUContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
        cls.cache=Path(os.environ['ISAAC_NATIVE_CACHE']);cls.native=NativePhysics.load(ROOT/'data/physics_v2/calibration_r5.json')

    def test_shared_generator_and_native_tensor_groups(self):
        cases=training_cases(12,32)
        a=MultiRoom(cases,13,self.native,self.cache);b=MultiRoom(cases,13,self.native,self.cache,reference=True)
        for env in (a,b):
            for child in env.groups:child.steps.fill_(395)
        for tick in range(16):
            action=(torch.arange(32,device='cuda')*13+tick*7)%45
            left=a.step(action);right=b.step(action);a._bind();b._bind()
            for x,y in zip(left[:3],right[:3]):torch.testing.assert_close(x,y,rtol=1e-5,atol=3e-4)
            for key in left[3]:torch.testing.assert_close(left[3][key],right[3][key],rtol=1e-5,atol=3e-4,msg=key)
            self.assertTrue(torch.equal(a.rng.get_state(),b.rng.get_state()))

    def test_captured_rollout_and_scene_change_for_all_rewards(self):
        for mode,budget in (('legacy',1),('win',1),('potential',1),('potential',3)):
            learner=Learner(x.CrossPolicy().cuda(),training_cases(4,128),23,self.native,self.cache,mode,budget)
            for _ in range(2):self.assertGreater(learner.update()['optimizer_steps'],0)
            learner.set_cases(training_cases(12,128));learner.update(epochs=0)
            self.assertTrue(all(torch.isfinite(v).all() for v in learner.policy.parameters()))
            self.assertEqual(learner.gamma,.995 if mode=='legacy' else 1.)
            del learner;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':unittest.main()
