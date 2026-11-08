import unittest
from unittest.mock import patch
import torch
from research.pilot_10m import cells,balanced_training,HOLDOUT,task_reward,gae,make_env,Scene,PilotPolicy
from physics_v2 import NativePhysics


class PilotContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.guard=patch('torch.cuda._lazy_init',side_effect=AssertionError('CPU test initialized CUDA'))
        cls.guard.start()

    @classmethod
    def tearDownClass(cls):cls.guard.stop()

    def test_disjoint_compositions_and_balanced_families(self):
        train=cells();held=cells(True)
        self.assertFalse({(s.enemies,s.shooters) for s in train}&HOLDOUT)
        self.assertEqual({(s.enemies,s.shooters) for s in held},HOLDOUT)
        self.assertEqual({s.enemies for s in train},{1,2,3,4})
        scenes=balanced_training(1024);counts={}
        for s in scenes:counts[s.family]=counts.get(s.family,0)+1
        self.assertLessEqual(max(counts.values())-min(counts.values()),1)
        self.assertTrue(all(s.chaser_hp==12 for s in cells(hard=True)))

    def test_reward_bounded_and_not_wall_or_raw_hp_dependent(self):
        info=dict(success=torch.tensor([False,False]),contact=torch.tensor([False,False]),damage=torch.tensor([2.,12.]))
        reward,parts=task_reward(info,torch.tensor([2.,12.]),True)
        torch.testing.assert_close(reward,torch.tensor([.099,.099]))
        self.assertLessEqual(float(parts['progress'].sum()/2),.100001)
        strict,_=task_reward(info,torch.tensor([2.,12.]),False)
        torch.testing.assert_close(strict,torch.tensor([-.001,-.001]))
        info['contact'].fill_(True)
        terminal,_=task_reward(info,torch.tensor([2.,12.]),True)
        self.assertTrue((terminal<0).all())
        info['success'].fill_(True)
        # A simultaneous clear/contact is not a success in the simulator; reward
        # consumes its authoritative success flag rather than inferring HP.

    def test_gae_bootstraps_timeout_not_reset_and_cuts_recursion(self):
        r=torch.tensor([[1.],[2.]])
        dones=torch.tensor([[True],[False]])
        timeout=dones.clone();values=torch.zeros(2,1)
        adv,target=gae(r,dones,timeout,values,torch.tensor([3.]),torch.tensor([[7.],[0.]]),gamma=.5,lam=1.)
        torch.testing.assert_close(target,torch.tensor([[4.5],[3.5]]))
        timeout.zero_()
        adv,target=gae(r,dones,timeout,values,torch.tensor([3.]),torch.tensor([[7.],[0.]]),gamma=.5,lam=1.)
        self.assertEqual(target[0].item(),1.)

    def test_previous_request_is_an_explicit_policy_input(self):
        policy=PilotPolicy()
        self.assertEqual(policy.body[0].in_features,222)
        x=torch.zeros(3,236)
        logits,value=policy(x,torch.tensor([0,20,45]))
        self.assertEqual(logits.shape,(3,45));self.assertEqual(value.shape,(3,))
        self.assertFalse(torch.equal(logits[0],logits[1]))

    def test_autoreset_preserves_terminal_observation_and_context(self):
        sim,e,hp=make_env([Scene(1,0,2),Scene(4,2,4)],7,NativePhysics(),'cpu',reference=True)
        e.hp[0].zero_();before=e.pos[0].clone()
        obs,_,done,info=sim.step(torch.tensor([20,0]))
        self.assertTrue(done[0]);self.assertEqual(e.steps[0].item(),0)
        # terminal snapshot is separate from the newly spawned observation.
        self.assertFalse(torch.equal(obs[0],info['terminal_observation'][0]))
        self.assertEqual(e.last_action[0].item(),0)
        self.assertEqual(hp.tolist(),[2.,32.])


if __name__=='__main__':unittest.main(verbosity=2)
