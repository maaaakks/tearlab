"""GPU curriculum: 2, 3, 4 controlled chasers; gated no-contact clears."""
import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from alignment_lab import Physics, swept_hit, tear_velocity

ROOT=Path(__file__).resolve().parents[1]


def physics():
    return Physics(player_speed=237,velocity_retention=.6967**3,movement_inheritance=.6,
                   inheritance_parallel=0,fire_interval=11/30,room_width=520,room_height=280,
                   native_coordinates=True,player_velocity_scale=59.59,moving_target=True,
                   pursuit=True,target_hits=4,target_speed_min=60,target_speed_max=120,
                   target_turn_fraction=0,max_steps=300,wall_cost=.02)


def first_contact(start,end,radius):
    """First time in [0,1] a relative segment enters a disk; inf for no hit."""
    delta=end-start
    a=delta.square().sum(-1)
    b=(start*delta).sum(-1)
    c=start.square().sum(-1)-radius**2
    disc=b.square()-a*c
    t=(-b-disc.clamp_min(0).sqrt())/a.clamp_min(1e-12)
    valid=(disc>=0)&(a>1e-12)&(t>=0)&(t<=1)
    return torch.where(c<=0,0.,torch.where(valid,t,float('inf')))


class CombatPolicy(nn.Module):
    """Shared entity encoder, invariant to enemy ordering; no tactical action heads."""
    def __init__(self):
        super().__init__()
        self.enemy=nn.Sequential(nn.Linear(6,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
        self.body=nn.Sequential(nn.Linear(112,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh())
        self.actor=nn.Linear(128,45);self.value=nn.Linear(128,1)

    def forward(self,obs):
        enemies=obs[:,48:].reshape(-1,4,6)
        mask=enemies[:,:,-1:]>0
        encoded=self.enemy(enemies)
        total=torch.where(mask,encoded,0).sum(1)
        maximum=torch.where(mask,encoded,-1e9).amax(1)
        maximum=torch.where(mask.any(1),maximum,0)
        x=self.body(torch.cat((obs[:,:48],total,maximum),-1))
        return self.actor(x),self.value(x).squeeze(-1)


class CombatEnv:
    slots=8
    damage_ends_episode=True

    def before_step(self):
        pass

    def damage_events(self,contact,old_pos):
        return contact,{}
    def __init__(self,count,device,seed,enemies=2,p=None):
        if not 1<=enemies<=4: raise ValueError('1..4 enemies required')
        self.n=count;self.device=torch.device(device);self.p=p or physics();self.enemies=enemies
        self.rng=torch.Generator(device=self.device).manual_seed(seed)
        self.size=torch.tensor([self.p.room_width,self.p.room_height],device=self.device)
        self.move=torch.tensor([[0,0],[0,-1],[0,1],[-1,0],[1,0],[-1,-1],[1,-1],[-1,1],[1,1]],device=self.device,dtype=torch.float32)
        self.move/=self.move.norm(dim=-1,keepdim=True).clamp_min(1)
        self.pos=torch.zeros(count,2,device=self.device);self.vel=torch.zeros_like(self.pos)
        self.targets=torch.zeros(count,4,2,device=self.device);self.target_vel=torch.zeros_like(self.targets)
        self.hp=torch.zeros(count,4,device=self.device,dtype=torch.long)
        self.speed=torch.zeros(count,4,device=self.device)
        self.bpos=torch.zeros(count,8,2,device=self.device);self.bvel=torch.zeros_like(self.bpos)
        self.life=torch.zeros(count,8,device=self.device);self.cooldown=torch.zeros(count,device=self.device)
        self.steps=torch.zeros(count,device=self.device,dtype=torch.long)
        self.reset(torch.ones(count,device=self.device,dtype=torch.bool))

    def reset(self,mask):
        r=torch.rand(self.n,15,generator=self.rng,device=self.device)
        pos=self.size/2+(r[:,:2]-.5)*torch.tensor([40.,20.],device=self.device)
        angles=r[:,2:3]*6.283185307+torch.arange(4,device=self.device)[None]*6.283185307/self.enemies+(r[:,3:7]-.5)*.6
        radius=90+20*r[:,7:11]
        targets=pos[:,None]+torch.stack((angles.cos(),angles.sin()),-1)*radius[:,:,None]
        self.pos=torch.where(mask[:,None],pos,self.pos)
        self.targets=torch.where(mask[:,None,None],targets,self.targets)
        self.speed=torch.where(mask[:,None],self.p.target_speed_min+r[:,11:15]*(self.p.target_speed_max-self.p.target_speed_min),self.speed)
        hp=(torch.arange(4,device=self.device)<self.enemies).long()*self.p.target_hits
        self.hp=torch.where(mask[:,None],hp,self.hp)
        delta=self.pos[:,None]-self.targets
        velocity=delta/delta.norm(dim=-1,keepdim=True).clamp_min(1e-6)*self.speed[:,:,None]
        self.target_vel=torch.where(mask[:,None,None],velocity,self.target_vel)
        self.vel[mask]=0;self.life[mask]=0;self.cooldown[mask]=0;self.steps[mask]=0

    def obs(self):
        bullets=torch.cat(((self.bpos-self.pos[:,None])/320,self.bvel/400,self.life[:,:,None]/self.p.tear_lifetime),-1)
        bullets=torch.where((self.life>0)[:,:,None],bullets,0).flatten(1)
        active=self.hp>0
        enemies=torch.cat(((self.targets-self.pos[:,None])/320,self.target_vel/100,
                           (self.hp/self.p.target_hits)[:,:,None],active[:,:,None]),-1)
        enemies=torch.where(active[:,:,None],enemies,0).flatten(1)
        return torch.cat((self.vel/100,self.pos/640,(self.size-self.pos)/640,
                          self.cooldown[:,None]/self.p.fire_interval,self.steps[:,None]/self.p.max_steps,
                          bullets,enemies),-1)

    @torch.no_grad()
    def step(self,actions):
        self.before_step()
        alive=self.hp>0;old_targets=self.targets.clone();old_pos=self.pos.clone()
        delta=self.pos[:,None]-self.targets
        self.target_vel=delta/delta.norm(dim=-1,keepdim=True).clamp_min(1e-6)*self.speed[:,:,None]*alive[:,:,None]
        self.targets+=self.target_vel*self.p.dt
        move,shoot=actions//5,actions%5
        self.vel=self.p.velocity_retention*self.vel+(1-self.p.velocity_retention)*self.p.player_speed*self.move[move]
        proposed=self.pos+self.vel*self.p.dt
        self.pos=proposed.clamp(torch.full_like(self.size,20),self.size-20)
        self.vel=torch.where(self.pos!=proposed,0,self.vel)
        self.cooldown=(self.cooldown-self.p.dt).clamp_min(0)
        rows=torch.arange(self.n,device=self.device);slot=self.life.argmin(1)
        fire=(shoot!=0)&(self.cooldown<=1e-6)&(self.life[rows,slot]<=0)
        self.bpos[rows,slot]=torch.where(fire[:,None],self.pos,self.bpos[rows,slot])
        self.bvel[rows,slot]=torch.where(fire[:,None],tear_velocity(self.move[shoot],self.vel,self.p),self.bvel[rows,slot])
        self.life[rows,slot]=torch.where(fire,self.p.tear_lifetime,self.life[rows,slot])
        self.cooldown=torch.where(fire,self.p.fire_interval,self.cooldown)
        end=self.bpos+self.bvel*self.life.clamp(0,self.p.dt)[:,:,None]
        target_end=old_targets[:,None]+(self.targets-old_targets)[:,None]*(self.life.clamp(0,self.p.dt)/self.p.dt)[:,:,None,None]
        times=first_contact(self.bpos[:,:,None]-old_targets[:,None],end[:,:,None]-target_end,self.p.hit_radius)
        times=torch.where(alive[:,None]&(self.life>0)[:,:,None],times,float('inf'))
        closest,index=times.min(-1);hits=closest.isfinite()
        counts=torch.zeros_like(self.hp).scatter_add(1,index,hits.long())
        damage=counts.clamp_max(self.hp);self.hp-=damage
        self.bpos=end;self.life=(self.life-self.p.dt).clamp_min(0)
        outside=((end<0)|(end>self.size)).any(-1)
        self.life[hits|outside]=0
        contact=(alive&swept_hit(old_pos[:,None]-old_targets,self.pos[:,None]-self.targets,
                                torch.zeros_like(self.targets),self.p.contact_radius)).any(-1)
        contact,extra=self.damage_events(contact,old_pos)
        self.steps+=1
        cleared=(self.hp<=0).all(-1);success=cleared&~contact
        timeout=(self.steps>=self.p.max_steps)&~cleared&~(contact & self.damage_ends_episode)
        done=cleared|(contact & self.damage_ends_episode)|timeout
        kills=(alive&(self.hp<=0)).sum(-1)
        wall=(move!=0)&((self.pos-old_pos).norm(dim=-1)<.5)
        reward=success.float()-contact.float()-.002-self.p.wall_cost*wall+.05*damage.sum(-1)*(~contact)
        info=dict(success=success,contact=contact,timeout=timeout,kills=kills,damage=damage.sum(-1),fired=fire,wall=wall)
        info.update(extra)
        self.reset(done)
        return self.obs(),reward,done,info


@torch.no_grad()
def evaluate(policy,device,enemies,seed,episodes=1024,p=None):
    env=CombatEnv(episodes,device,seed,enemies,p)
    active=torch.ones(episodes,device=device,dtype=torch.bool)
    totals={key:torch.zeros(episodes,device=device) for key in ['success','contact','timeout','kills','damage','fired','wall','steps']}
    for _ in range(env.p.max_steps):
        _,_,done,info=env.step(policy(env.obs())[0].argmax(-1))
        for key in totals: totals[key]+=active*(1 if key=='steps' else info[key])
        active&=~done
    result={key:float(value.mean()) for key,value in totals.items()}
    result.update(episodes=episodes,seed=seed,enemies=enemies,seconds=result.pop('steps')*env.p.dt)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda');parser.add_argument('--envs',type=int,default=512)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--max-updates-per-stage',type=int,default=3000)
    parser.add_argument('--check-every',type=int,default=500)
    parser.add_argument('--threshold',type=float,default=.90)
    args=parser.parse_args()
    if args.envs<1 or args.max_updates_per_stage<1 or args.check_every<1 or not 0<args.threshold<1: parser.error('Invalid budget or threshold')
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    p=physics();policy=CombatPolicy().to(args.device);optimizer=torch.optim.Adam(policy.parameters(),lr=.001)
    folder=ROOT/'data/multi_combat'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');folder.mkdir(parents=True)
    report=dict(config=vars(args),physics=asdict(p),algorithm='A2C',from_scratch=True,live_validated=False,stages=[],complete=False)
    print(f'Output: {folder}',flush=True);started=time.monotonic()
    for enemies in [2,3,4]:
        env=CombatEnv(args.envs,args.device,args.seed+enemies*100,enemies,p);obs=env.obs()
        stage=dict(enemies=enemies,evaluations=[],passed=False)
        report['stages'].append(stage)
        for update in range(1,args.max_updates_per_stage+1):
            values=[];probs=[];rewards=[];dones=[];entropies=[]
            for _ in range(32):
                logits,value=policy(obs);dist=torch.distributions.Categorical(logits=logits);action=dist.sample()
                obs,reward,done,_=env.step(action)
                values.append(value);probs.append(dist.log_prob(action));rewards.append(reward);dones.append(done);entropies.append(dist.entropy())
            with torch.no_grad():
                returns=policy(obs)[1];targets=[]
                for r,d in zip(reversed(rewards),reversed(dones)):
                    returns=r+.99*(~d)*returns;targets.append(returns)
                targets=torch.stack(list(reversed(targets)))
            advantage=targets-torch.stack(values)
            loss=-(torch.stack(probs)*advantage.detach()).mean()+.5*advantage.square().mean()-.01*torch.stack(entropies).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step()
            if update%100==0: print(dict(enemies=enemies,update=update,loss=loss.item()),flush=True)
            if update%args.check_every==0 or update==args.max_updates_per_stage:
                # A fresh promotion lot each time; none of its transitions train the network.
                result=evaluate(policy,args.device,enemies,1_000_000+enemies*10000+update)
                result['update']=update;stage['evaluations'].append(result);print(result,flush=True)
                stage['passed']=result['success']>args.threshold
                stage['updates']=update
                checkpoint=folder/f'policy_{enemies}v1.pt'
                torch.save(dict(schema_version=2,state_dict=policy.state_dict(),physics=asdict(p),enemies=enemies,
                                passed=stage['passed'],algorithm='A2C'),checkpoint)
                (folder/'report.json').write_text(json.dumps(report,indent=2))
                if stage['passed']: break
        # Separate final audit, not used for promotion or further training.
        stage['audit']=evaluate(policy,args.device,enemies,9_000_000+enemies)
        stage['regression']={str(n):evaluate(policy,args.device,n,9_100_000+n) for n in range(2,enemies)}
        if not stage['passed']: break
    report['complete']=len(report['stages'])==3 and all(s['passed'] for s in report['stages'])
    report['elapsed_seconds']=time.monotonic()-started
    report['transitions']=sum(s['updates'] for s in report['stages'])*32*args.envs
    (folder/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
