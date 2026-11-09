"""Hard combat compositions with per-cell metrics and balanced family aggregation."""
from dataclasses import dataclass
import torch
from physics_v2 import NativeShooterEnv

CELLS=[dict(name=f'{p}P_{s}T_hp{h}',family=f'{p}P_{s}T',chasers=p,shooters=s,hits=h) for p,s in [(3,0),(4,0),(2,2),(3,1)] for h in [4,8,12]]+[dict(name='0P_4T',family='0P_4T',chasers=0,shooters=4,hits=12)]

class CombatSuiteEnv(NativeShooterEnv):
    def __init__(self,ids,device,seed,native):
        self.cell_ids=torch.tensor(ids,device=device,dtype=torch.long)
        self.shooters=torch.tensor([CELLS[i]['shooters'] for i in ids],device=device)
        self.counts=torch.tensor([CELLS[i]['shooters']+CELLS[i]['chasers'] for i in ids],device=device)
        self.chaser_hits=torch.tensor([CELLS[i]['hits'] for i in ids],device=device)
        slots=torch.arange(4,device=device)[None]
        self.is_chaser=(slots>=self.shooters[:,None])&(slots<self.counts[:,None])
        super().__init__(len(ids),device,seed,enemies=4,native=native)

    def reset(self,mask):
        if self.keep_terminal:return
        super().reset(mask)
        varied=mask&(self.shooters<4)
        positions=20+torch.rand(self.n,2,generator=self.rng,device=self.device)*(self.size-40)
        # Keep at least 90 units to each enemy and 35 between enemies. Candidate
        # rejection samples valid geometry; never clamp an enemy onto the player.
        targets=[];valid_room=torch.ones(self.n,dtype=torch.bool,device=self.device)
        for slot in range(4):
            angles=torch.rand(self.n,64,generator=self.rng,device=self.device)*6.283185307
            radius=90+20*torch.rand(self.n,64,generator=self.rng,device=self.device)
            candidates=positions[:,None]+torch.stack((angles.cos(),angles.sin()),-1)*radius[:,:,None]
            valid=((candidates>=20)&(candidates<=self.size-20)).all(-1)
            for previous in targets: valid&=(candidates-previous[:,None]).norm(dim=-1)>=35
            valid_room&=valid.any(-1)
            index=valid.long().argmax(-1)
            targets.append(candidates[torch.arange(self.n,device=self.device),index])
        varied&=valid_room
        self.pos=torch.where(varied[:,None],positions,self.pos)
        self.targets=torch.where(varied[:,None,None],torch.stack(targets,1),self.targets)
        active=torch.arange(4,device=self.device)[None]<self.counts[:,None]
        hp=torch.where(self.is_chaser,self.chaser_hits[:,None],12)*active
        self.hp=torch.where(mask[:,None],hp,self.hp)
        speed=(60+40*torch.rand(self.n,4,generator=self.rng,device=self.device))*self.is_chaser
        self.speed=torch.where(mask[:,None],speed,self.speed)
        self.attack_timer=torch.where(self.is_chaser|~active,1000000,self.attack_timer)
        self.update_velocity()

    def update_velocity(self):
        delta=self.pos[:,None]-self.targets
        self.target_vel=delta/delta.norm(dim=-1,keepdim=True).clamp_min(1e-6)*self.speed[:,:,None]*(self.hp>0)[:,:,None]

    def before_step(self):
        self.attack_timer=torch.where(self.is_chaser,1000000,self.attack_timer)
        super().before_step();self.update_velocity()
        proposed=self.targets+self.target_vel/30
        low=proposed<20;high=proposed>self.size-20
        self.targets=torch.where(low,40-proposed,torch.where(high,2*(self.size-20)-proposed,proposed))
        self.target_vel=torch.where(low|high,-self.target_vel,self.target_vel)


def aggregate(cells):
    families={}
    for row in cells:
        families.setdefault(row['family'],[]).append(row)
    keys=['no_hit_clear_rate','damage_rate','timeout_rate','mean_kills','mean_enemy_hits','near_wall_fraction','blocked_fraction','left_fraction']
    family_metrics={name:{key:sum(r[key] for r in rows)/len(rows) for key in keys} for name,rows in families.items()}
    return dict(cells=cells,families=family_metrics,global_metrics={key:sum(r[key] for r in family_metrics.values())/len(family_metrics) for key in keys},worst_family_rate=min(r['no_hit_clear_rate'] for r in family_metrics.values()))

@torch.no_grad()
def evaluate(policy,device,native,seed,per_cell=256):
    ids=[i for i in range(len(CELLS)) for _ in range(per_cell)]
    env=CombatSuiteEnv(ids,device,seed,native);env.keep_terminal=True
    active=torch.ones(env.n,device=device,dtype=torch.bool);wins=torch.zeros_like(active)
    keys=['steps','contact','timeout','kills','damage','near_wall','blocked','left']
    totals={k:torch.zeros(env.n,device=device) for k in keys}
    for _ in range(env.p.max_steps):
        pos=env.pos.clone();near=torch.minimum(pos,env.size-pos).amin(-1)<50
        actions=policy(env.obs())[0].argmax(-1);_,_,done,info=env.step(actions)
        extras=dict(steps=torch.ones_like(pos[:,0]),near_wall=near,left=pos[:,0]<260,blocked=(actions//5!=0)&((env.pos-pos).norm(dim=-1)<.5))
        for k in keys:totals[k]+=active*(extras[k] if k in extras else info[k])
        wins|=active&done&info['success'];active&=~done
        if not active.any():break
    rows=[]
    for i,cell in enumerate(CELLS):
        m=env.cell_ids==i;times=totals['steps'][m&wins]*.1;steps=totals['steps'][m].sum().clamp_min(1)
        rows.append(dict(**cell,episodes=per_cell,seed=seed,no_hit_clear_rate=float(wins[m].float().mean()),damage_rate=float(totals['contact'][m].mean()),timeout_rate=float(totals['timeout'][m].mean()),mean_kills=float(totals['kills'][m].mean()),mean_enemy_hits=float(totals['damage'][m].mean()),clear_seconds_median=float(times.median()) if len(times) else None,clear_seconds_p90=float(torch.quantile(times,.9)) if len(times) else None,near_wall_fraction=float(totals['near_wall'][m].sum()/steps),blocked_fraction=float(totals['blocked'][m].sum()/steps),left_fraction=float(totals['left'][m].sum()/steps)))
    return aggregate(rows)
