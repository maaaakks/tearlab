"""Explicit finite-horizon combat tasks on the qualified native simulator.

Life budget is experimental (contacts, not calibrated Isaac hearts). Native
motion/projectile code stays unchanged. Health is resolved once per decision;
native immunity prevents more than one damage event in that decision.
"""
from dataclasses import dataclass, asdict
import copy
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from research import pilot_10m as p
from research.parallel_ppo import ParallelArm, Options
from research.staged_simulator import StagedCombatSuiteEnv, ALL, Optimizations
from research.gpu_simulator import CudaGraphSimulator


@dataclass(frozen=True)
class Scene:
    chasers: int = 0
    shooters: int = 1
    chaser_hp: int = 4
    shooter_hp: int = 12
    attacks: bool = True

    @property
    def family(self):return f'{self.chasers}P_{self.shooters}T'

    @property
    def name(self):return f'{self.family}_p{self.chaser_hp}_t{self.shooter_hp}_a{int(self.attacks)}'


HOLDOUT={(2,1),(2,2),(1,3)}  # (chasers, shooters), never a training stage


def balance(scenes,n):
    return [scenes[i%len(scenes)] for i in range(n)]


def task_math(success,dead,timeout,contact,before,after,damage,initial_hp,
              mode='potential',gamma=.995,scale=1.5):
    terminal=success|dead|timeout
    task=3*success.float()-3*dead.float()-2*timeout.float()-.1*contact.float()
    time=torch.full_like(task,-.002)
    if mode=='potential':
        progress=scale*(gamma*torch.where(terminal,0.,after)-before)
    elif mode=='bounded':progress=scale*damage/initial_hp
    elif mode=='sparse':progress=torch.zeros_like(task)
    else:raise ValueError(mode)
    return task+time+progress,dict(task=task,time=time,progress=progress)


class TaskEnv(StagedCombatSuiteEnv):
    def __init__(self,scenes,seed,native,device='cuda',cache=None,budget=3,
                 reward_mode='potential',gamma=.995,width=520,height=280,
                 optimizations=ALL,autoreset=True):
        if budget not in (1,3):raise ValueError('Experimental budget must be 1 or 3 contacts')
        self._task_ready=False
        super().__init__([12]*len(scenes),device,seed,native,
                         optimizations=optimizations,cache=cache)
        self.budget=budget
        self.damage_ends_episode=budget==1
        self.reward_mode=reward_mode
        self.gamma=gamma
        self.autoreset=autoreset
        self.p.room_width=width;self.p.room_height=height;self.p.max_steps=400
        self.size.copy_(torch.tensor([width,height],device=self.device))
        n=self.n;d=self.device
        self.health=torch.full((n,),float(budget),device=d)
        self.episode_hits=torch.zeros(n,device=d)
        self.potential=torch.zeros(n,device=d)
        self.initial_hp=torch.ones(n,device=d)
        self.shooter_hp=torch.full((n,),12,device=d,dtype=torch.long)
        self.attacks_enabled=torch.ones(n,device=d,dtype=torch.bool)
        self._task_ready=True
        self.keep_terminal=False
        self.set_scenes(scenes)
        self.keep_terminal=True

    def set_scenes(self,scenes):
        if len(scenes)!=self.n:raise ValueError('Scene count must match persistent batch')
        self.counts.copy_(torch.tensor([s.chasers+s.shooters for s in scenes],device=self.device))
        self.shooters.copy_(torch.tensor([s.shooters for s in scenes],device=self.device))
        self.chaser_hits.copy_(torch.tensor([s.chaser_hp for s in scenes],device=self.device))
        self.shooter_hp.copy_(torch.tensor([s.shooter_hp for s in scenes],device=self.device))
        self.attacks_enabled.copy_(torch.tensor([s.attacks for s in scenes],device=self.device))
        slots=torch.arange(4,device=self.device)[None]
        self.is_chaser.copy_((slots>=self.shooters[:,None])&(slots<self.counts[:,None]))
        self.initial_hp.copy_(torch.tensor([s.chasers*s.chaser_hp+s.shooters*s.shooter_hp for s in scenes],device=self.device))
        terminal=self.keep_terminal
        self.keep_terminal=False
        self.reset(torch.ones(self.n,device=self.device,dtype=torch.bool))
        self.keep_terminal=terminal

    def reset(self,mask):
        if getattr(self,'keep_terminal',False):return
        super().reset(mask)
        if not self._task_ready:return
        active=torch.arange(4,device=self.device)[None]<self.counts[:,None]
        hp=torch.where(self.is_chaser,self.chaser_hits[:,None],self.shooter_hp[:,None])*active
        self.hp.copy_(torch.where(mask[:,None],hp,self.hp))
        self.health.copy_(torch.where(mask,float(self.budget),self.health))
        self.episode_hits.masked_fill_(mask,0)
        self.potential.masked_fill_(mask,0)
        self.attack_timer.copy_(torch.where(mask[:,None]&~self.attacks_enabled[:,None],1000000,self.attack_timer))

    def obs(self):
        native=super().obs()
        if not self._task_ready:return native
        return torch.cat((native,self.health[:,None]/3,self.immunity[:,None].float()/30),-1)

    @torch.no_grad()
    def step(self,actions):
        before=self.potential.clone()
        _,_,_,info=super().step(actions)
        contact=info['contact']
        self.health.sub_(contact.float())
        self.episode_hits.add_(contact.float())
        cleared=(self.hp<=0).all(-1)
        dead=self.health<=0
        success=cleared&~dead
        timeout=(self.steps>=self.p.max_steps)&~success&~dead
        done=success|dead|timeout
        after=(1-self.hp.sum(-1)/self.initial_hp).clamp(0,1)
        reward,parts=task_math(success,dead,timeout,contact,before,after,
                               info['damage'],self.initial_hp,self.reward_mode,self.gamma)
        self.potential.copy_(after)
        info.update(success=success,dead=dead,timeout=timeout,
                    nohit=success&(self.episode_hits==0),episode_hits=self.episode_hits.clone(),
                    progress_fraction=after,reward_task=parts['task'],
                    reward_progress=parts['progress'],reward_time=parts['time'])
        info['terminal_observation']=self.obs()
        if self.autoreset:
            self.keep_terminal=False
            try:self.reset(done)
            finally:self.keep_terminal=True
        return self.obs(),reward,done,info


def make_env(scenes,seed,native,device,cache,reference=False,*,budget=3,
             reward_mode='potential',gamma=.995,width=520,height=280):
    env=TaskEnv(scenes,seed,native,device,cache,budget,reward_mode,gamma,width,height,
                optimizations=Optimizations() if reference else ALL)
    env.prepare()
    sim=env if device=='cpu' else CudaGraphSimulator(env)
    return sim,env,env.initial_hp


class Policy(p.PilotPolicy):
    def __init__(self):
        super().__init__()
        self.body[0]=nn.Linear(224,128)

    def forward(self,obs,previous):
        enemies=obs[:,48:76].reshape(-1,4,7)
        projectiles=obs[:,76:236].reshape(-1,32,5)
        features=torch.cat((obs[:,:48],self.pool(self.enemy,enemies,enemies[:,:,5]>0),
            self.pool(self.projectile,projectiles,projectiles[:,:,4]>0),
            F.one_hot(previous,46).to(obs.dtype),obs[:,236:]),1)
        state=self.body(features)
        return self.actor(state),self.value(state).squeeze(-1)


def reward_from_info(info,initial_hp,dense):
    parts=dict(task=info['reward_task'],progress=info['reward_progress'],time=info['reward_time'])
    return parts['task']+parts['progress']+parts['time'],parts


class Learner(ParallelArm):
    def __init__(self,initial,scenes,seed,native,cache,budget=3,reward_mode='potential',
                 gamma=.995,lr=3e-4,rollout=32):
        def factory(scenes,seed,native,device,cache,reference):
            return make_env(scenes,seed,native,device,cache,reference,budget=budget,
                            reward_mode=reward_mode,gamma=gamma)
        super().__init__(initial,scenes,seed,native,'cuda',cache,True,rollout=rollout,
                         env_factory=factory,options=Options(True,True,True,False),
                         reward_fn=reward_from_info,bootstrap_timeouts=False,gamma=gamma)
        self.name=f'life{budget}-{reward_mode}'
        for group in self.optimizer.param_groups:group['lr']=lr

    def set_scenes(self,scenes):
        self.env.set_scenes(scenes)
        self.sim._bind()
        self.obs.copy_(self.env.obs())
        self.previous.fill_(45)


class Examiner:
    """Greedy actions; fixed scenes/seeds; reusable 32-decision evaluation graph."""
    def __init__(self,policy,scenes,per_cell,seed,native,cache,budget=3,width=520,height=280):
        self.policy=policy
        self.scenes=scenes
        self.per_cell=per_cell
        self.seed=seed
        expanded=[s for s in scenes for _ in range(per_cell)]
        self.sim,self.env,_=make_env(expanded,seed,native,'cuda',cache,budget=budget,width=width,height=height)
        n=self.env.n;d=self.env.device
        self.obs=self.env.obs().clone()
        self.previous=torch.full((n,),45,device=d,dtype=torch.long)
        self.active=torch.ones(n,device=d,dtype=torch.bool)
        self.stats=torch.zeros(n,7,device=d)
        self.action_rng=torch.Generator(device=d).manual_seed(seed+999)
        # Stats: clear,nohit,contacts,timeouts,hp_fraction,steps,shots.
        current=torch.cuda.current_stream();stream=torch.cuda.Stream();stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(2):self._chunk()
        current.wait_stream(stream);current.synchronize()
        self.graph=torch.cuda.CUDAGraph();self.graph.register_generator_state(self.env.rng)
        self.graph.register_generator_state(self.action_rng)
        with torch.cuda.graph(self.graph,stream=stream):self._chunk()
        current.wait_stream(stream)

    @torch.no_grad()
    def _chunk(self):
        for _ in range(32):
            action=(self.policy(self.obs,self.previous)[0].argmax(-1) if self.policy is not None else
                    torch.randint(45,(self.env.n,),device=self.env.device,generator=self.action_rng))
            obs,_,done,info=self.env.step(action)
            self.sim._bind()
            self.stats[:,0].add_(self.active*info['success'])
            self.stats[:,1].add_(self.active*info['nohit'])
            self.stats[:,2].add_(self.active*info['contact'])
            self.stats[:,3].add_(self.active*info['timeout'])
            self.stats[:,4].add_(self.active*done*info['progress_fraction'])
            self.stats[:,5].add_(self.active)
            self.stats[:,6].add_(self.active*info['fired'])
            self.active.logical_and_(~done)
            self.obs.copy_(obs)
            self.previous.copy_(torch.where(done,45,action))

    @torch.no_grad()
    def run(self):
        self.env.rng.manual_seed(self.seed)
        self.action_rng.manual_seed(self.seed+999)
        self.sim.reset(torch.ones(self.env.n,device='cuda',dtype=torch.bool))
        self.obs.copy_(self.env.obs());self.previous.fill_(45)
        self.active.fill_(True);self.stats.zero_()
        for _ in range(math.ceil(self.env.p.max_steps/32)):self.graph.replay()
        if self.active.any():raise RuntimeError('Unfinished audit episodes')
        values=self.stats.cpu().reshape(len(self.scenes),self.per_cell,7)
        keys=['clear_rate','nohit_rate','damage_events','timeout_rate','hp_fraction','decisions','shots']
        rows=[dict(scene=asdict(s),name=s.name,**dict(zip(keys,values[i].mean(0).tolist()))) for i,s in enumerate(self.scenes)]
        families={}
        for row,s in zip(rows,self.scenes):families.setdefault(s.family,[]).append(row)
        family_metrics={f:{k:sum(r[k] for r in items)/len(items) for k in keys} for f,items in families.items()}
        return dict(seed=self.seed,budget=self.env.budget,episodes=self.env.n,cells=rows,
                    families=family_metrics,macro={k:sum(r[k] for r in family_metrics.values())/len(family_metrics) for k in keys})


def old_id_cells():
    return [Scene(s.enemies-s.shooters,s.shooters,s.chaser_hp) for s in p.cells()]


def holdout_cells():
    return [Scene(c,s,hp) for c,s in sorted(HOLDOUT) for hp in [2,4,8]]
