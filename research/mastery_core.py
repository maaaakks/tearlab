"""Hard topology, simultaneous geometry groups and finite-horizon win objectives."""
from dataclasses import dataclass,asdict
import copy
import math
import torch
from research import crossfire_core as x
c=x.c


@dataclass(frozen=True)
class HardScene(x.CrossScene):
    pattern: str='random'

    @property
    def name(self):return super().name+'_'+self.pattern


@dataclass(frozen=True)
class Case:
    scene: HardScene
    width: int=520
    height: int=280

    @property
    def name(self):return f'{self.width}x{self.height}_'+self.scene.name


class HardEnv(x.CrossEnv):
    def __init__(self,scenes,*args,**kwargs):
        self._hard_ready=False
        super().__init__(scenes,*args,**kwargs)
        self.pattern=torch.zeros(self.n,device=self.device,dtype=torch.long)
        self._hard_ready=True;self.set_scenes(scenes)

    def set_scenes(self,scenes):
        if self._hard_ready:
            names={'random':0,'surround':1,'long':2,'unequal':3}
            self.pattern.copy_(torch.tensor([names[s.pattern] for s in scenes],device=self.device))
        super().set_scenes(scenes)

    def reset(self,mask):
        if not self._hard_ready:return super().reset(mask)
        if self.keep_terminal:return
        super().reset(mask)
        n,d=self.n,self.device
        random=torch.rand(n,4,4,generator=self.rng,device=d)
        player_random=torch.rand(n,64,2,generator=self.rng,device=d)
        count=self.counts.clamp_min(1)
        angle=random[:,0,0:1]*(2*math.pi)+torch.arange(4,device=d)[None]*(2*math.pi)/count[:,None]
        angle=angle+(random[:,:,1]-.5)*.44
        max_radius=min(115.,min(self.p.room_width,self.p.room_height)/2-26)
        radius=85+random[:,:,2]*(max_radius-85)
        center=self.size/2+(random[:,0,:2]-.5)*10
        regular=center[:,None]+torch.stack((angle.cos(),angle.sin()),-1)*radius[:,:,None]
        # Reserved unequal ring: alternating near/far radii, not used in training.
        far=min(150.,min(self.p.room_width,self.p.room_height)/2-26)
        unequal_radius=torch.where((torch.arange(4,device=d)%2)[None]==0,85.,far)
        unequal=center[:,None]+torch.stack((angle.cos(),angle.sin()),-1)*unequal_radius[:,:,None]
        ellipse=self.size[None,None]/2+torch.stack((angle.cos()*self.size[0]*.32,angle.sin()*self.size[1]*.30),-1)
        candidates=30+player_random*(self.size-60)
        # Keep corner/border intent while avoiding unsafe initial overlaps.
        side=(random[:,0,3]*4).long()
        candidates[:,:,0]=torch.where((self.layout==2)[:,None]&(side<2)[:,None],torch.where((side==0)[:,None],30.,self.size[0]-30),candidates[:,:,0])
        candidates[:,:,1]=torch.where((self.layout==2)[:,None]&(side>=2)[:,None],torch.where((side==2)[:,None],30.,self.size[1]-30),candidates[:,:,1])
        corners=torch.where(player_random<.5,30.,self.size-30)
        candidates=torch.where((self.layout==3)[:,None,None],corners,candidates)
        active=torch.arange(4,device=d)[None]<self.counts[:,None]
        distances=(candidates[:,:,None]-ellipse[:,None]).norm(dim=-1).masked_fill(~active[:,None],1e4).amin(-1)
        valid=distances>=80
        index=torch.where(valid.any(-1),valid.long().argmax(-1),distances.argmax(-1))
        long_pos=candidates.gather(1,index[:,None,None].expand(-1,1,2)).squeeze(1)
        selected_pos=torch.where((self.pattern==2)[:,None],long_pos,center)
        selected_targets=torch.where((self.pattern==2)[:,None,None],ellipse,regular)
        selected_targets=torch.where((self.pattern==3)[:,None,None],unequal,selected_targets)
        change=mask&(self.pattern>0)
        self.pos.copy_(torch.where(change[:,None],selected_pos,self.pos))
        self.targets.copy_(torch.where(change[:,None,None],selected_targets,self.targets))
        delta=self.pos[:,None]-self.targets
        velocity=delta/delta.norm(dim=-1,keepdim=True).clamp_min(1e-6)*self.speed[:,:,None]*active[:,:,None]
        self.target_vel.copy_(torch.where(change[:,None,None],velocity,self.target_vel))


def objective(info,before,mode):
    done=info['success']|info['dead']|info['timeout']
    if mode=='legacy':
        return info['reward_task']+info['reward_time'],dict(task=info['reward_task'],time=info['reward_time'],progress=torch.zeros_like(before))
    task=3*info['success'].float()-3*(info['dead']|info['timeout']).float()
    progress=1.5*(torch.where(done,0.,info['progress_fraction'])-before) if mode=='potential' else torch.zeros_like(before)
    if mode not in ('win','potential'):raise ValueError(mode)
    return task+progress,dict(task=task,time=torch.zeros_like(before),progress=progress)


class MultiRoom:
    """One policy/rollout over independent native groups; no physics ABI changes.

    Fixed-order cases must be contiguous by geometry. A single registered CUDA
    generator drives all groups, with stable buffers for whole-rollout capture.
    """
    def __init__(self,cases,seed,native,cache,device='cuda',mode='legacy',reference=False,budget=1):
        self.cases=list(cases);self.n=len(cases);self.device=torch.device(device)
        if budget not in (1,3):raise ValueError('Life budget must be 1 or 3 contacts')
        self.mode=mode;self.budget=budget;self.rng=torch.Generator(device=device).manual_seed(seed)
        self.groups=[];self.slices=[];seen=set();start=0
        while start<len(cases):
            geometry=(cases[start].width,cases[start].height)
            if geometry in seen:raise ValueError('Geometry groups must be contiguous')
            seen.add(geometry);end=start+1
            while end<len(cases) and (cases[end].width,cases[end].height)==geometry:end+=1
            env=HardEnv([s.scene for s in cases[start:end]],seed,native,device,cache,budget=budget,reward_mode='sparse',
                        width=geometry[0],height=geometry[1],optimizations=c.Optimizations() if reference else c.ALL)
            env.prepare();env.rng=self.rng
            self.groups.append(env);self.slices.append(slice(start,end));start=end
        self.p=self.groups[0].p
        self.actions=torch.zeros(self.n,device=device,dtype=torch.long)
        self.device=self.actions.device
        self._stream=torch.cuda.current_stream() if device=='cuda' else None
        self._slots=[{k:v for k,v in vars(env).items() if isinstance(v,torch.Tensor)} for env in self.groups]
        self._signatures=[c.CudaGraphSimulator._config_signature(type('Proxy',(),{'env':e})()) for e in self.groups]
        self.initial_hp=torch.cat([e.initial_hp for e in self.groups])
        self.reset(torch.ones(self.n,device=device,dtype=torch.bool))

    def _snapshot(self):return [{k:v.clone() for k,v in slots.items()} for slots in self._slots]

    def _restore(self,saved):
        for env,slots,values in zip(self.groups,self._slots,saved):
            for k,v in values.items():slots[k].copy_(v);setattr(env,k,slots[k])

    def _bind(self):
        for env,slots in zip(self.groups,self._slots):
            for name,slot in slots.items():
                current=getattr(env,name)
                if current is not slot:slot.copy_(current);setattr(env,name,slot)

    def _check(self,actions):
        if actions.dtype!=torch.long or actions.shape!=(self.n,) or actions.device!=self.device:raise ValueError('Actions ABI')
        if self._stream is not None and torch.cuda.current_stream()!=self._stream:raise RuntimeError('Capture stream changed')
        for env,slots,signature in zip(self.groups,self._slots,self._signatures):
            if c.CudaGraphSimulator._config_signature(type('Proxy',(),{'env':env})())!=signature:raise RuntimeError('Configuration changed')
            if any(getattr(env,k) is not v for k,v in slots.items()):raise RuntimeError('Tensor rebound')

    def reset(self,mask):
        for env,part in zip(self.groups,self.slices):
            env.keep_terminal=False
            try:env.reset(mask[part])
            finally:env.keep_terminal=True
        self._bind()

    def set_cases(self,cases):
        if len(cases)!=self.n:raise ValueError('Case count')
        for env,part in zip(self.groups,self.slices):
            if any((s.width,s.height)!=(env.p.room_width,env.p.room_height) for s in cases[part]):raise ValueError('Geometry storage changed')
            env.set_scenes([s.scene for s in cases[part]])
        self.cases=list(cases);self.initial_hp.copy_(torch.cat([e.initial_hp for e in self.groups]));self._bind()

    def obs(self):return torch.cat([e.obs() for e in self.groups])

    def step(self,actions):
        before=torch.cat([e.potential for e in self.groups])
        results=[e.step(actions[part]) for e,part in zip(self.groups,self.slices)]
        obs=torch.cat([r[0] for r in results]);done=torch.cat([r[2] for r in results])
        info={k:torch.cat([r[3][k] for r in results]) for k in results[0][3]}
        reward,parts=objective(info,before,self.mode)
        info.update(reward_task=parts['task'],reward_time=parts['time'],reward_progress=parts['progress'])
        return obs,reward,done,info


class Learner(c.ParallelArm):
    def __init__(self,initial,cases,seed,native,cache,mode='legacy',budget=1):
        def factory(scenes,seed,native,device,cache,reference):
            env=MultiRoom(scenes,seed,native,cache,device,mode,reference,budget)
            return env,env,env.initial_hp
        super().__init__(initial,cases,seed,native,'cuda',cache,True,env_factory=factory,rollout=32,
                         options=c.Options(True,True,True,False),reward_fn=c.reward_from_info,
                         bootstrap_timeouts=False,gamma=.995 if mode=='legacy' else 1.)

    def set_cases(self,cases):
        self.env.set_cases(cases);self.obs.copy_(self.env.obs());self.previous.fill_(45)


GEOMETRIES=[(440,240),(520,280),(640,360)]
RETENTION=[HardScene(0,1),HardScene(0,2),HardScene(4,0,4),HardScene(1,1,4),HardScene(1,2,4),HardScene(3,1,4)]


def training_cases(hp,n=1024):
    targets=[HardScene(0,k,shooter_hp=hp,pattern='surround') for k in (3,4)]
    targets += [HardScene(0,k,shooter_hp=hp,pattern='long',placement=p) for k in (3,4) for p in ('free','edge','corner')]
    targets += [HardScene(0,4,shooter_hp=hp,placement=p) for p in ('free','edge','corner')]
    scenes=targets*2+RETENTION*2
    cases=[]
    sizes=[n//4,n//2,n-n//4-n//2]
    for (width,height),size in zip(GEOMETRIES,sizes):cases += [Case(scenes[i%len(scenes)],width,height) for i in range(size)]
    return cases


def dev_cells():
    scenes=[HardScene(0,k,pattern='surround') for k in (3,4)]
    scenes += [HardScene(0,4,pattern='long',placement=p) for p in ('free','edge','corner')]
    return [Case(s,w,h) for w,h in GEOMETRIES for s in scenes+RETENTION]


class Examiner:
    def __init__(self,policy,cells,per_cell,seed,native,cache,controller=None,budget=1):
        self.policy=policy;self.controller=controller;self.cells=cells;self.per_cell=per_cell;self.seed=seed
        self.env=MultiRoom([cell for cell in cells for _ in range(per_cell)],seed,native,cache,budget=budget)
        self.obs=self.env.obs().clone();self.previous=torch.full((self.env.n,),45,device='cuda',dtype=torch.long)
        self.active=torch.ones(self.env.n,device='cuda',dtype=torch.bool);self.stats=torch.zeros(self.env.n,7,device='cuda')
        self.sim=self.env
        current=torch.cuda.current_stream();stream=torch.cuda.Stream();stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(2):self._chunk()
        current.wait_stream(stream);current.synchronize()
        self.graph=torch.cuda.CUDAGraph();self.graph.register_generator_state(self.env.rng)
        with torch.cuda.graph(self.graph,stream=stream):self._chunk()
        current.wait_stream(stream)

    @torch.no_grad()
    def _chunk(self):
        for _ in range(32):
            action=self.policy(self.obs,self.previous)[0].argmax(-1) if self.controller is None else self.controller(self.obs,self.previous)
            obs,_,done,info=self.env.step(action);self.env._bind()
            for column,value in enumerate((info['success'],info['nohit'],info['contact'],info['timeout'],done*info['progress_fraction'],torch.ones_like(done),info['fired'])):
                self.stats[:,column].add_(self.active*value)
            self.active.logical_and_(~done);self.obs.copy_(obs);self.previous.copy_(torch.where(done,45,action))

    @torch.no_grad()
    def run(self):
        self.env.rng.manual_seed(self.seed);self.env.reset(torch.ones(self.env.n,device='cuda',dtype=torch.bool))
        self.obs.copy_(self.env.obs());self.previous.fill_(45);self.active.fill_(True);self.stats.zero_()
        for _ in range(13):self.graph.replay()
        if self.active.any():raise RuntimeError('Unfinished exam')
        values=self.stats.cpu().reshape(len(self.cells),self.per_cell,7)
        keys=['clear_rate','nohit_rate','damage','timeout_rate','hp_removed','decisions','shots']
        rows=[dict(name=s.name,case=asdict(s),**dict(zip(keys,values[i].mean(0).tolist()))) for i,s in enumerate(self.cells)]
        return dict(seed=self.seed,budget=self.env.budget,per_cell=self.per_cell,episodes=self.env.n,cells=rows)


def quality(results):
    averages=[sum(r['cells'][i]['clear_rate'] for r in results)/len(results) for i in range(len(results[0]['cells']))]
    hard=[];retention={};ret_cells=[]
    for cell,value in zip(results[0]['cells'],averages):
        if cell['case']['scene']['pattern']!='random':hard.append(value)
        else:
            key=cell['case']['scene']['chasers'],cell['case']['scene']['shooters']
            retention.setdefault(key,[]).append(value);ret_cells.append(value)
    ret=min(sum(v)/len(v) for v in retention.values())
    return (min(1.,ret/.8,min(ret_cells)/.65),min(hard),sum(hard)/len(hard))


class AnalyticController:
    """Independent feasibility reference; no training labels or reward shaping."""
    def __init__(self,native):
        self.native=native
        self.moves=torch.tensor([[0,0],[0,-1],[0,1],[-1,0],[1,0],[-1,-1],[1,-1],[-1,1],[1,1]],device='cuda',dtype=torch.float32)
        self.moves/=self.moves.norm(dim=-1,keepdim=True).clamp_min(1)

    def __call__(self,obs,previous):
        from physics_v2 import player_tick
        n=len(obs);enemy=obs[:,48:76].reshape(n,4,7);bullet=obs[:,76:236].reshape(n,32,5)
        pos=obs[:,2:4]*640;size=(obs[:,2:4]+obs[:,4:6])*640
        velocity=obs[:,:2]*100
        # Derive native velocity from the existing observed velocity scale.
        from multi_combat_lab import physics as base_physics
        scale=base_physics().player_velocity_scale
        p=pos[:,None].expand(-1,9,-1).reshape(-1,2)
        v=(velocity/scale)[:,None].expand(-1,9,-1).reshape(-1,2)
        extent=size[:,None].expand(-1,9,-1).reshape(-1,2)
        candidate=self.moves[None].expand(n,-1,-1).reshape(-1,2)
        old=self.moves[torch.where(previous==45,0,previous//5)][:,None].expand(-1,9,-1).reshape(-1,2)
        for tick in range(6):p,v,_=player_tick(p,v,old if tick<self.native.command_delay_ticks else candidate,extent,self.native)
        future=p.reshape(n,9,2);delta=future-pos[:,None]
        r=bullet[:,:,:2]*320;bv=bullet[:,:,2:4]*400
        rv=bv[:,None]-delta[:,:,None]/.2
        t=(-(r[:,None]*rv).sum(-1)/rv.square().sum(-1).clamp_min(1e-6)).clamp(0,.3)
        clearance=(r[:,None]+t[:,:,:,None]*rv).norm(dim=-1).masked_fill(bullet[:,None,:,4]<=0,1e4).amin(-1)
        target=pos[:,None]+enemy[:,:,:2]*320
        rel=target[:,None]-future[:,:,None]
        alive=enemy[:,:,5]>0
        axis=rel.abs().amin(-1).masked_fill(~alive[:,None],1e4).amin(-1)
        distance=rel.norm(dim=-1).masked_fill(~alive[:,None],1e4).amin(-1)
        wall=torch.minimum(future,size[:,None]-future).amin(-1)
        centroid=(target*alive[:,:,None]).sum(1)/alive.sum(1,keepdim=True).clamp_min(1)
        radial=pos-centroid;tangent=torch.stack((-radial[:,1],radial[:,0]),-1)
        tangent=tangent/tangent.norm(dim=-1,keepdim=True).clamp_min(1)
        motion=(delta*tangent[:,None]).sum(-1)/30
        score=-8*torch.exp(-clearance.clamp_min(0)/12)-5*torch.exp(-distance/20)-.008*axis+.12*motion
        score-=.15*(25-wall).clamp_min(0)
        move=score.argmax(-1)
        relative=enemy[:,:,:2]*320
        # Cardinal axis nearest a living target; no proxy reward fed to PPO.
        dx,dy=relative.unbind(-1)
        vertical=dx.abs().masked_fill(~alive,1e4);horizontal=dy.abs().masked_fill(~alive,1e4)
        costs=torch.stack((vertical.masked_fill(dy>=0,1e4).amin(-1),vertical.masked_fill(dy<=0,1e4).amin(-1),
                           horizontal.masked_fill(dx>=0,1e4).amin(-1),horizontal.masked_fill(dx<=0,1e4).amin(-1)),-1)
        shoot=costs.argmin(-1)+1
        return move*5+shoot
