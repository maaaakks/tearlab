"""Controlled crossfire tasks and independently ablatable residual representations."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from research import campaign_core as c


@dataclass(frozen=True)
class CrossScene(c.Scene):
    placement: str = 'free'
    arrangement: str = 'ring'
    synchronized: bool = False

    @property
    def name(self):
        return super().name+'_'+self.placement+'_'+self.arrangement+('_sync' if self.synchronized else '')


PLACEMENTS={'center':0,'free':1,'edge':2,'corner':3}


def closest_passage(obs):
    """Constant-velocity geometry, NOT a collision oracle or future simulation."""
    bullets=obs[:,76:236].reshape(-1,32,5)
    r=bullets[:,:,:2]*320
    v=bullets[:,:,2:4]*400-obs[:,:2,None].transpose(1,2)*100
    dot=(r*v).sum(-1)
    t=(-dot/v.square().sum(-1).clamp_min(1e-6)).clamp(0,2)
    distance=(r+t[:,:,None]*v).square().sum(-1).clamp_min(1e-12).sqrt()
    return torch.stack((t/2,distance/320,(dot<0).to(obs.dtype)),dim=-1)


class CrossPolicy(c.Policy):
    def __init__(self,variant='pool'):
        super().__init__();self.variant=variant
        if variant=='risk':
            self.extra=nn.Sequential(nn.Linear(8,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
            self.residual=nn.Linear(64,128)
        elif variant=='attention':
            self.query=nn.Linear(54,32)
            self.key=nn.Linear(5,32);self.val=nn.Linear(5,32)
            self.residual=nn.Linear(32,128)
        elif variant!='pool':raise ValueError(variant)
        if variant!='pool':
            nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)

    def forward(self,obs,previous):
        enemies=obs[:,48:76].reshape(-1,4,7)
        bullets=obs[:,76:236].reshape(-1,32,5)
        prev=F.one_hot(previous,46).to(obs.dtype)
        features=torch.cat((obs[:,:48],self.pool(self.enemy,enemies,enemies[:,:,5]>0),
                            self.pool(self.projectile,bullets,bullets[:,:,4]>0),prev,obs[:,236:]),1)
        state=self.body(features)
        if self.variant=='risk':
            extra=torch.cat((bullets,closest_passage(obs)),dim=-1)
            state=state+self.residual(self.pool(self.extra,extra,bullets[:,:,4]>0))
        elif self.variant=='attention':
            q=self.query(torch.cat((obs[:,:8],prev),-1))
            score=(self.key(bullets)*q[:,None]).sum(-1)/math.sqrt(32)
            mask=bullets[:,:,4]>0
            weight=score.masked_fill(~mask,-1e4).softmax(-1)*mask
            weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-8)
            state=state+self.residual((self.val(bullets)*weight[:,:,None]).sum(1))
        return self.actor(state),self.value(state).squeeze(-1)


class CrossEnv(c.TaskEnv):
    """New reset distribution; native motion, collision and projectiles unchanged.

    Generate four separated positions regardless of active count. A deterministic
    grid fallback preserves validity if rejection sampling runs out; report its use.
    All RNG calls have fixed shapes and occur even on an empty reset mask.
    """
    def __init__(self,scenes,*args,**kwargs):
        self._cross_ready=False
        super().__init__(scenes,*args,**kwargs)
        n,d=self.n,self.device
        self.layout=torch.zeros(n,device=d,dtype=torch.long)
        self.grouped=torch.zeros(n,device=d,dtype=torch.bool)
        self.synchronized=torch.zeros(n,device=d,dtype=torch.bool)
        self.reset_fallbacks=torch.zeros(n,device=d,dtype=torch.long)
        self._grid=torch.tensor([(x,y) for y in (.1,.5,.9) for x in (.1,.5,.9)],device=d)*self.size
        self._cross_ready=True
        self.set_scenes(scenes)

    def set_scenes(self,scenes):
        if self._cross_ready:
            self.layout.copy_(torch.tensor([PLACEMENTS[s.placement] for s in scenes],device=self.device))
            self.grouped.copy_(torch.tensor([s.arrangement=='grouped' for s in scenes],device=self.device))
            self.synchronized.copy_(torch.tensor([s.synchronized for s in scenes],device=self.device))
        super().set_scenes(scenes)

    def _draw_attacks(self,tick=0):
        super()._draw_attacks(tick)
        if self._cross_ready:
            self._random[tick].copy_(torch.where(self.synchronized[:,None],self._random[tick,:,:1],self._random[tick]))

    def reset(self,mask):
        if not self._cross_ready:return super().reset(mask)
        if self.keep_terminal:return
        n,d=self.n,self.device
        r=torch.rand(n,4,generator=self.rng,device=d)
        center=self.size/2+(r[:,:2]-.5)*20
        free=30+r[:,:2]*(self.size-60)
        side=(r[:,2]*4).long()
        edge=free.clone()
        edge[:,0]=torch.where(side<2,torch.where(side==0,30.,self.size[0]-30),free[:,0])
        edge[:,1]=torch.where(side>=2,torch.where(side==2,30.,self.size[1]-30),free[:,1])
        corner=torch.where(r[:,:2]<.5,30.,self.size-30)
        pos=torch.where((self.layout==0)[:,None],center,free)
        pos=torch.where((self.layout==2)[:,None],edge,pos)
        pos=torch.where((self.layout==3)[:,None],corner,pos)
        a=torch.rand(n,4,64,generator=self.rng,device=d)
        radius=torch.rand(n,4,64,generator=self.rng,device=d)
        toward=torch.atan2(self.size[1]/2-pos[:,1],self.size[0]/2-pos[:,0])
        angles=torch.where(self.grouped[:,None,None],toward[:,None,None]+(a-.5)*1.8,a*(2*math.pi))
        radii=torch.where(self.grouped[:,None,None],90+radius*80,90+radius*20)
        candidates=pos[:,None,None]+torch.stack((angles.cos(),angles.sin()),-1)*radii[:,:,:,None]
        chosen=[];fallbacks=torch.zeros(n,device=d,dtype=torch.long)
        for slot in range(4):
            points=candidates[:,slot]
            valid=((points>=20)&(points<=self.size-20)).all(-1)
            grid=self._grid[None].expand(n,-1,-1)
            grid_ok=(grid-pos[:,None]).square().sum(-1)>=80**2
            for old in chosen:
                valid &= (points-old[:,None]).square().sum(-1)>=35**2
                grid_ok &= (grid-old[:,None]).square().sum(-1)>=35**2
            selected=points.gather(1,valid.long().argmax(-1)[:,None,None].expand(-1,1,2)).squeeze(1)
            fallback=grid.gather(1,grid_ok.long().argmax(-1)[:,None,None].expand(-1,1,2)).squeeze(1)
            exists=valid.any(-1)
            chosen.append(torch.where(exists[:,None],selected,fallback))
            fallbacks+=(~exists).long()
        targets=torch.stack(chosen,1)
        self.pos.copy_(torch.where(mask[:,None],pos,self.pos))
        self.targets.copy_(torch.where(mask[:,None,None],targets,self.targets))
        speed=(60+40*torch.rand(n,4,generator=self.rng,device=d))*self.is_chaser
        self.speed.copy_(torch.where(mask[:,None],speed,self.speed))
        active=torch.arange(4,device=d)[None]<self.counts[:,None]
        hp=torch.where(self.is_chaser,self.chaser_hits[:,None],self.shooter_hp[:,None])*active
        self.hp.copy_(torch.where(mask[:,None],hp,self.hp))
        delta=self.pos[:,None]-self.targets
        velocity=delta/delta.norm(dim=-1,keepdim=True).clamp_min(1e-6)*self.speed[:,:,None]*active[:,:,None]
        self.target_vel.copy_(torch.where(mask[:,None,None],velocity,self.target_vel))
        cfg=self.config
        timer=torch.randint(cfg.first_attack_min_steps,cfg.first_attack_max_steps+1,(n,4),generator=self.rng,device=d)
        timer=torch.where(self.synchronized[:,None],timer[:,:1],timer)
        timer=torch.where(self.is_chaser|~active|~self.attacks_enabled[:,None],1000000,timer)
        self.attack_timer.copy_(torch.where(mask[:,None],timer,self.attack_timer))
        projectile_speed=cfg.projectile_speed_min+torch.rand(n,4,generator=self.rng,device=d)*(cfg.projectile_speed_max-cfg.projectile_speed_min)
        self.attack_speed.copy_(torch.where(mask[:,None],projectile_speed,self.attack_speed))
        for name in ['vel','native_velocity','life','hostile_life','cooldown','steps','immunity','last_action','fire_clock','episode_hits','potential']:
            state=getattr(self,name);state.masked_fill_(mask.reshape((n,)+(1,)*(state.ndim-1)),0)
        self.health.masked_fill_(mask,float(self.budget));self.eye.masked_fill_(mask,1)
        self.reset_fallbacks.add_(mask*fallbacks)


def make_env(scenes,seed,native,device,cache,reference=False,*,budget=1,width=520,height=280):
    env=CrossEnv(scenes,seed,native,device,cache,budget=budget,reward_mode='sparse',width=width,height=height,
                 optimizations=c.Optimizations() if reference else c.ALL)
    env.prepare()
    return (env if device=='cpu' else c.CudaGraphSimulator(env)),env,env.initial_hp


class CrossLearner(c.ParallelArm):
    def __init__(self,initial,scenes,seed,native,cache,options=None):
        super().__init__(initial,scenes,seed,native,'cuda',cache,True,rollout=32,
                         env_factory=make_env,options=options or c.Options(True,True,True,False),
                         reward_fn=c.reward_from_info,bootstrap_timeouts=False,gamma=.995)

    def set_scenes(self,scenes):
        self.env.set_scenes(scenes);self.sim._bind();self.obs.copy_(self.env.obs());self.previous.fill_(45)


RETENTION=[CrossScene(0,1),CrossScene(0,2),CrossScene(4,0,4),CrossScene(1,1,4),CrossScene(1,2,4),CrossScene(3,1,4)]
TARGETS=[CrossScene(0,k,shooter_hp=hp,placement=place) for k in (3,4) for hp in (4,12) for place in ('free','edge','corner')]


def schedule(count,hp):
    target=[CrossScene(0,count,shooter_hp=hp,placement=place) for place in ('center','free','edge','corner')]
    return target*3+RETENTION


class CrossExaminer(c.Examiner):
    def __init__(self,policy,scenes,per_cell,seed,native,cache,budget=1,width=520,height=280):
        # Reuse the proven first-episode-only graph/aggregation, substituting factory
        # through an explicit constructor rather than monkey-patching shared module state.
        self.policy=policy;self.scenes=scenes;self.per_cell=per_cell;self.seed=seed
        expanded=[s for s in scenes for _ in range(per_cell)]
        self.sim,self.env,_=make_env(expanded,seed,native,'cuda',cache,budget=budget,width=width,height=height)
        n=self.env.n;d=self.env.device
        self.obs=self.env.obs().clone();self.previous=torch.full((n,),45,device=d,dtype=torch.long)
        self.active=torch.ones(n,device=d,dtype=torch.bool);self.stats=torch.zeros(n,7,device=d)
        self.action_rng=torch.Generator(device=d).manual_seed(seed+999)
        current=torch.cuda.current_stream();stream=torch.cuda.Stream();stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(2):self._chunk()
        current.wait_stream(stream);current.synchronize()
        self.graph=torch.cuda.CUDAGraph();self.graph.register_generator_state(self.env.rng)
        self.graph.register_generator_state(self.action_rng)
        with torch.cuda.graph(self.graph,stream=stream):self._chunk()
        current.wait_stream(stream)
