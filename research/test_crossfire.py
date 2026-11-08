import os
from pathlib import Path
import sys
import unittest
ROOT=Path(r'C:\Users\maxbr\Desktop\game\IsaacRL\binding-of-ai')
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
import torch
try:from research.crossfire_core import *
except ModuleNotFoundError:from crossfire_core import *
from physics_v2 import NativePhysics


class CrossContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def env(self,scenes):
        env=CrossEnv(scenes,43,NativePhysics(),'cpu',budget=1,optimizations=c.Optimizations())
        env.keep_terminal=False
        return env

    def test_controlled_spawns_valid_and_independent_of_enemy_count(self):
        scenes=[CrossScene(0,4,placement=p,arrangement=a) for p in PLACEMENTS for a in ('ring','grouped')]*64
        env=self.env(scenes)
        mask=torch.ones(env.n,dtype=torch.bool)
        for seed in (17,61,97):
            env.rng.manual_seed(seed);env.reset(mask)
            self.assertTrue(((env.targets>=20)&(env.targets<=env.size-20)).all())
            self.assertTrue(((env.targets-env.pos[:,None]).norm(dim=-1)>=79.99).all())
            for i in range(4):
                for j in range(i):self.assertTrue(((env.targets[:,i]-env.targets[:,j]).norm(dim=-1)>=34.99).all())
        env.rng.manual_seed(191);env.reset(mask);pos=env.pos.clone();targets=env.targets.clone()
        env.set_scenes([CrossScene(0,2,placement=s.placement,arrangement=s.arrangement) for s in scenes])
        env.rng.manual_seed(191);env.reset(mask)
        torch.testing.assert_close(env.pos,pos,rtol=0,atol=0);torch.testing.assert_close(env.targets,targets,rtol=0,atol=0)

    def test_partial_reset_preserves_other_lane_and_empty_mask_advances_rng(self):
        env=self.env([CrossScene(),CrossScene(0,4)])
        names=['pos','targets','hp','vel','speed','health','attack_timer','hostile_life','steps']
        before={k:getattr(env,k).clone() for k in names}
        env.reset(torch.tensor([True,False]))
        for k in names:torch.testing.assert_close(getattr(env,k)[1],before[k][1],rtol=0,atol=0)
        before={k:getattr(env,k).clone() for k in names};rng=env.rng.get_state()
        env.reset(torch.zeros(2,dtype=torch.bool))
        for k in names:torch.testing.assert_close(getattr(env,k),before[k],rtol=0,atol=0)
        self.assertFalse(torch.equal(rng,env.rng.get_state()))

    def test_residual_variants_start_identically_and_ignore_padding(self):
        torch.manual_seed(91);base=c.Policy()
        env=self.env([CrossScene(0,4),CrossScene(0,2)])
        obs=env.obs();previous=torch.tensor([45,9])
        for variant in ('pool','risk','attention','capacity'):
            policy=CrossPolicy(variant);policy.load_state_dict(base.state_dict(),strict=False)
            a,v=policy(obs,previous);x,y=base(obs,previous)
            torch.testing.assert_close(a,x,rtol=0,atol=0);torch.testing.assert_close(v,y,rtol=0,atol=0)
            if variant!='pool':
                torch.nn.init.normal_(policy.residual.weight,std=.1)
                changed=obs.clone();bullets=changed[:,76:236].reshape(-1,32,5)
                bullets[:,:,:4]=torch.randn_like(bullets[:,:,:4])*100
                # Padded projectile tokens must not influence either representation.
                a,v=policy(obs,previous);x,y=policy(changed,previous)
                torch.testing.assert_close(a,x);torch.testing.assert_close(v,y)
                loss=a.square().mean()+v.square().mean();loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None))

    def test_closest_passage_geometry_and_galilean_velocity(self):
        obs=torch.zeros(1,238);bullets=obs[:,76:236].reshape(1,32,5)
        bullets[0,0]=torch.tensor([100/320,0,-100/400,0,1.])
        x=closest_passage(obs)[0,0]
        self.assertAlmostEqual(x[0].item(),.5);self.assertLess(x[1].item(),1e-6);self.assertEqual(x[2].item(),1.)
        bullets[0,0,2]+=50/400;obs[0,0]=50/100
        torch.testing.assert_close(closest_passage(obs)[0,0],x)

    def test_permutation_invariant_projectile_branch(self):
        env=self.env([CrossScene(0,3)]);obs=env.obs()
        obs[:,76:236]=torch.randn(1,160);obs[:,76:236].reshape(1,32,5)[:,:,4]=1
        perm=torch.randperm(32);changed=obs.clone();changed[:,76:236]=obs[:,76:236].reshape(1,32,5)[:,perm].flatten(1)
        for variant in ('risk','attention','capacity'):
            policy=CrossPolicy(variant);torch.nn.init.normal_(policy.residual.weight,std=.1)
            a,v=policy(obs,torch.tensor([45]));x,y=policy(changed,torch.tensor([45]))
            torch.testing.assert_close(a,x,atol=1e-5,rtol=1e-5);torch.testing.assert_close(v,y,atol=1e-5,rtol=1e-5)

    def test_schedules_keep_final_factors_closed(self):
        for count,hp in ((3,4),(3,12),(4,4),(4,12)):
            scenes=schedule(count,hp)
            self.assertFalse(any(s.synchronized or s.arrangement=='grouped' for s in scenes))
            self.assertFalse({(s.chasers,s.shooters) for s in scenes}&c.HOLDOUT)

    def test_synchronized_exam_repeats_joint_attacks(self):
        env=self.env([CrossScene(0,4,synchronized=True)]*8)
        env.prepare();env.keep_terminal=True
        emitted=0
        for _ in range(50):
            _,_,_,info=env.step(torch.zeros(8,dtype=torch.long))
            self.assertTrue(((info['attacks']%4)==0).all())
            emitted+=int(info['attacks'].sum())
        self.assertGreater(emitted,100)


@unittest.skipUnless(os.environ.get('RUN_ISAAC_GPU_TESTS')=='1','Explicit GPU test opt-in')
class CrossGPU(unittest.TestCase):
    def test_native_tensor_and_capture_terminals(self):
        torch.set_num_threads(1)
        cache=Path(os.environ['ISAAC_NATIVE_CACHE']);native=NativePhysics.load(ROOT/'data/physics_v2/calibration_r5.json')
        scenes=[CrossScene(0,k,placement=p,synchronized=s) for k in (2,3,4) for p in PLACEMENTS for s in (False,True)]
        sim,a,_=make_env(scenes,91,native,'cuda',cache)
        _,b,_=make_env(scenes,91,native,'cuda',cache,reference=True)
        for env in (a,b):
            env.rng.manual_seed(599);env.keep_terminal=False;env.reset(torch.ones(env.n,device='cuda',dtype=torch.bool));env.keep_terminal=True
            env.steps.fill_(390)
        sim._bind()
        generator=torch.Generator(device='cuda').manual_seed(61)
        for _ in range(25):
            action=torch.randint(45,(a.n,),device='cuda',generator=generator)
            x,r,d,info=sim.step(action);y,q,e,other=b.step(action)
            torch.testing.assert_close(x,y,rtol=1e-5,atol=2e-6);torch.testing.assert_close(r,q,rtol=1e-5,atol=2e-6)
            self.assertTrue(torch.equal(d,e));self.assertTrue(torch.equal(info['success'],other['success']))
            self.assertTrue(torch.equal(a.rng.get_state(),b.rng.get_state()))

    def test_each_representation_trains_and_checks_capture_state(self):
        torch.set_num_threads(1);cache=Path(os.environ['ISAAC_NATIVE_CACHE']);native=NativePhysics.load(ROOT/'data/physics_v2/calibration_r5.json')
        for variant in ('pool','risk','attention','capacity'):
            learner=CrossLearner(CrossPolicy(variant).cuda(),c.balance(schedule(3,4),128),71,native,cache)
            for _ in range(2):self.assertGreater(learner.update()['optimizer_steps'],0)
            self.assertTrue(all(torch.isfinite(p).all() for p in learner.policy.parameters()))
            learner.set_scenes(c.balance(schedule(4,12),128));learner.update(epochs=0)
            self.assertTrue((learner.env.shooters<=4).all())


if __name__=='__main__':unittest.main()
