import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).parent))
from live_agent import LiveEncoder,Policy,ROOT
from research.campaign_core import TaskEnv,Scene,Optimizations
from physics_v2 import NativePhysics


def packet(env):
    origin=torch.tensor([40.,120.]);p=env.p
    enemies=[];projectiles=[]
    for i in range(4):
        if env.hp[0,i]>0:
            enemies.append(dict(init_seed=i+10,position=(env.targets[0,i]+origin).tolist(),velocity=(env.target_vel[0,i]/30).tolist(),
                                hp=float(env.hp[0,i])*3.5,attack_warning=int(env.attack_timer[0,i]<=env.config.warning_steps)))
    for i in range(8):
        if env.life[0,i]>0:
            projectiles.append(dict(init_seed=i+20,type=2,friendly=True,position=(env.bpos[0,i]+origin).tolist(),
                                    velocity=(env.bvel[0,i]/30).tolist(),age_ticks=(p.tear_lifetime-float(env.life[0,i]))*30))
    for i in range(32):
        if env.hostile_life[0,i]>0:
            projectiles.append(dict(init_seed=i+40,type=9,friendly=False,position=(env.hostile_pos[0,i]+origin).tolist(),
                                    velocity=(env.hostile_vel[0,i]/30).tolist()))
    return dict(player=dict(position=(env.pos[0]+origin).tolist(),velocity=(env.vel[0]/p.player_velocity_scale).tolist(),fire_delay=float(env.cooldown[0])*30,damage=3.5),
                room=dict(top_left=origin.tolist(),width=p.room_width,height=p.room_height),enemies=enemies,projectiles=projectiles,control_tick=int(env.steps[0])*3)


class LiveContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_projection_matches_qualified_simulator_observation(self):
        with patch('torch.cuda._lazy_init',side_effect=AssertionError('Live inference initialized CUDA')):
            env=TaskEnv([Scene(1,2,4)],41,NativePhysics(),'cpu',budget=1,optimizations=Optimizations())
            env.steps.fill_(37);env.vel.copy_(torch.tensor([[63.,-48.]]));env.cooldown.fill_(.2)
            env.life[0,:2]=torch.tensor([.5,.7]);env.bpos[0,:2]=torch.tensor([[70.,160.],[110.,180.]])
            env.bvel[0,:2]=torch.tensor([[200.,30.],[-80.,90.]])
            env.hostile_life[0,0]=1.;env.hostile_pos[0,0]=torch.tensor([60.,70.]);env.hostile_vel[0,0]=torch.tensor([10.,190.])
            actual=LiveEncoder().encode(packet(env))
            torch.testing.assert_close(actual,env.obs(),atol=2e-6,rtol=1e-5)

    def test_capacity_guards_and_tear_slot_stability(self):
        env=TaskEnv([Scene()],41,NativePhysics(),'cpu',budget=1,optimizations=Optimizations())
        state=packet(env);encoder=LiveEncoder()
        state['enemies']*=5
        with self.assertRaises(ValueError):encoder.encode(state)
        state=packet(env);state['projectiles']=[dict(init_seed=i,type=9,friendly=False,position=[0,0],velocity=[0,0]) for i in range(33)]
        with self.assertRaises(ValueError):encoder.encode(state)
        state=packet(env);state['projectiles']=[dict(init_seed=i,type=2,friendly=True,position=[50*i,100],velocity=[1,0],age_ticks=2) for i in (2,1)]
        a=encoder.encode(state);state['projectiles'].reverse();b=encoder.encode(state)
        torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_frozen_checkpoint_cpu_inference(self):
        root=Path(__file__).parent.parent/'isaac-crossfire/focus-seed129773/pool-best.pt'
        with patch('torch.cuda._lazy_init',side_effect=AssertionError('Live inference initialized CUDA')):
            model=Policy().eval();model.load_state_dict(torch.load(root,map_location='cpu',weights_only=True)['state_dict'])
            with torch.inference_mode():
                for previous in (45,0,44):
                    logits,value=model(torch.zeros(1,238),torch.tensor([previous]))
                    self.assertEqual(logits.shape,(1,45));self.assertTrue(torch.isfinite(logits).all())
                    self.assertTrue(torch.isfinite(value).all())


if __name__=='__main__':unittest.main()
