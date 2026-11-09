"""Feed-forward GPU shooter scenarios, reusable independently of the trainer."""
from dataclasses import dataclass,asdict
import torch
from torch import nn
from alignment_lab import Physics,swept_hit
from multi_combat_lab import CombatEnv,physics as combat_physics


@dataclass
class ShooterConfig:
    projectile_speed_min:float=180.
    projectile_speed_max:float=240.
    interval_min_steps:int=8
    interval_max_steps:int=13
    warning_steps:int=3
    first_attack_min_steps:int=4
    first_attack_max_steps:int=7
    projectile_lifetime:float=4.
    projectile_hit_radius:float=18.
    diagnostic_immunity_steps:int=10


def physics():
    p=combat_physics();p.target_speed_min=0.;p.target_speed_max=0.
    p.target_hits=12;p.max_steps=400
    return p


class ShooterPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.enemy=nn.Sequential(nn.Linear(7,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
        self.projectile=nn.Sequential(nn.Linear(5,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
        self.body=nn.Sequential(nn.Linear(176,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh())
        self.actor=nn.Linear(128,45);self.value=nn.Linear(128,1)

    @staticmethod
    def pool(encoder,features,mask):
        x=encoder(features);mask=mask[:,:,None]
        return torch.cat((torch.where(mask,x,0).sum(1),
                          torch.where(mask.any(1),torch.where(mask,x,-1e9).amax(1),0)),1)

    def forward(self,obs):
        enemies=obs[:,48:76].reshape(-1,4,7)
        projectiles=obs[:,76:].reshape(-1,32,5)
        x=torch.cat((obs[:,:48],self.pool(self.enemy,enemies,enemies[:,:,5]>0),
                     self.pool(self.projectile,projectiles,projectiles[:,:,4]>0)),1)
        x=self.body(x)
        return self.actor(x),self.value(x).squeeze(-1)


class ShooterEnv(CombatEnv):
    def __init__(self,count,device,seed,enemies=1,p=None,config=None,diagnostic=False):
        self.config=config or ShooterConfig();self.damage_ends_episode=not diagnostic
        self.hostile_pos=torch.zeros(count,32,2,device=device);self.hostile_vel=torch.zeros_like(self.hostile_pos)
        self.hostile_life=torch.zeros(count,32,device=device)
        self.attack_timer=torch.zeros(count,4,device=device,dtype=torch.long)
        self.attack_speed=torch.zeros(count,4,device=device)
        self.immunity=torch.zeros(count,device=device,dtype=torch.long)
        self.attacks=torch.zeros(count,device=device)
        super().__init__(count,device,seed,enemies,p or physics())

    def reset(self,mask):
        super().reset(mask)
        c=self.config
        timers=torch.randint(c.first_attack_min_steps,c.first_attack_max_steps+1,(self.n,4),generator=self.rng,device=self.device)
        speeds=c.projectile_speed_min+torch.rand(self.n,4,generator=self.rng,device=self.device)*(c.projectile_speed_max-c.projectile_speed_min)
        self.attack_timer=torch.where(mask[:,None],timers,self.attack_timer)
        self.attack_speed=torch.where(mask[:,None],speeds,self.attack_speed)
        self.hostile_life[mask]=0;self.immunity[mask]=0

    def obs(self):
        base=super().obs()
        enemies=base[:,48:].reshape(-1,4,6)
        warning=((self.attack_timer<=self.config.warning_steps)&(self.hp>0)).float()
        enemies=torch.cat((enemies,warning[:,:,None]),-1).flatten(1)
        active=self.hostile_life>0
        projectiles=torch.cat(((self.hostile_pos-self.pos[:,None])/320,self.hostile_vel/400,active[:,:,None]),-1)
        projectiles=torch.where(active[:,:,None],projectiles,0).flatten(1)
        return torch.cat((base[:,:48],enemies,projectiles),-1)

    def before_step(self):
        self.attack_timer-=1;self.immunity=(self.immunity-1).clamp_min(0)
        fire=(self.hp>0)&(self.attack_timer<=0)
        self.attacks=fire.sum(-1)
        rows=torch.arange(self.n,device=self.device)
        for enemy in range(4):
            slot=self.hostile_life.argmin(-1);emit=fire[:,enemy]&(self.hostile_life[rows,slot]<=0)
            delta=self.pos-self.targets[:,enemy]
            velocity=delta/delta.norm(dim=-1,keepdim=True).clamp_min(1e-6)*self.attack_speed[:,enemy,None]
            self.hostile_pos[rows,slot]=torch.where(emit[:,None],self.targets[:,enemy],self.hostile_pos[rows,slot])
            self.hostile_vel[rows,slot]=torch.where(emit[:,None],velocity,self.hostile_vel[rows,slot])
            self.hostile_life[rows,slot]=torch.where(emit,self.config.projectile_lifetime,self.hostile_life[rows,slot])
        timer=torch.randint(self.config.interval_min_steps,self.config.interval_max_steps+1,(self.n,4),device=self.device,generator=self.rng)
        self.attack_timer=torch.where(fire,timer,self.attack_timer)

    def damage_events(self,contact,old_pos):
        active=self.hostile_life>0
        end=self.hostile_pos+self.hostile_vel*self.hostile_life.clamp(0,self.p.dt)[:,:,None]
        hit=active&swept_hit(self.hostile_pos-old_pos[:,None],end-self.pos[:,None],torch.zeros_like(end),self.config.projectile_hit_radius)
        projectile=hit.any(-1)&(self.immunity<=0)
        touch=contact&(self.immunity<=0)
        damage=projectile|touch
        self.immunity=torch.where(damage,self.config.diagnostic_immunity_steps,self.immunity)
        self.hostile_pos=end;self.hostile_life=(self.hostile_life-self.p.dt).clamp_min(0)
        outside=((end<0)|(end>self.size)).any(-1)
        self.hostile_life[hit|outside]=0
        return damage,dict(projectile_damage=projectile,contact_damage=touch&~projectile,attacks=self.attacks)


@torch.no_grad()
def evaluate(policy,device,enemies,seed,episodes=1024,p=None,config=None,diagnostic=False):
    env=ShooterEnv(episodes,device,seed,enemies,p,config,diagnostic)
    active=torch.ones(episodes,device=device,dtype=torch.bool)
    total={k:torch.zeros(episodes,device=device) for k in ['contact','projectile_damage','contact_damage','attacks','kills','fired','steps','timeout']}
    cleared=torch.zeros_like(active)
    for _ in range(env.p.max_steps):
        _,_,done,info=env.step(policy(env.obs())[0].argmax(-1))
        for k in total: total[k]+=active*(1 if k=='steps' else info[k])
        # Completion can occur alongside a hit in diagnostic mode; kills counts remain authoritative.
        cleared |= active & done & (total['kills']>=enemies)
        active&=~done
    nohit=cleared&(total['contact']==0)
    times=total['steps']*env.p.dt;winning=times[nohit]
    return dict(enemies=enemies,episodes=episodes,seed=seed,diagnostic=diagnostic,
                no_hit_clear_rate=float(nohit.float().mean()),clear_rate=float(cleared.float().mean()),
                damage_per_episode=float(total['contact'].mean()),projectile_damage=float(total['projectile_damage'].mean()),
                contact_damage=float(total['contact_damage'].mean()),timeout_rate=float(total['timeout'].mean()),
                clear_seconds_median=float(winning.median()) if len(winning) else None,
                clear_seconds_p90=float(torch.quantile(winning,.9)) if len(winning) else None,
                mean_attacks=float(total['attacks'].mean()),min_attacks_in_nohit_clears=int(total['attacks'][nohit].min()) if nohit.any() else None,
                mean_shots=float(total['fired'].mean()))
