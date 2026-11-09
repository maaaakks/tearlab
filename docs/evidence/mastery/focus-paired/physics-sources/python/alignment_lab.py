"""Batched temporal alignment experiment. Physics is provisional, not calibrated Isaac."""
import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn


@dataclass
class Physics:
    dt: float = 0.1
    player_speed: float = 100.
    velocity_retention: float = 0.5
    tear_speed: float = 300.
    movement_inheritance: float = 1.0  # hypothesis, NOT an engine measurement
    tear_lifetime: float = 0.8
    fire_interval: float = 0.3
    hit_radius: float = 18.
    contact_radius: float = 25.
    max_steps: int = 100
    room_width: float = 640.
    room_height: float = 400.
    inheritance_parallel: float = -1.  # negative = legacy full-vector inheritance
    shot_budget: int = 0  # zero = legacy unlimited
    shot_cost: float = 0.
    miss_cost: float = 0.
    wall_cost: float = 0.
    native_coordinates: bool = False
    player_velocity_scale: float = 30.
    moving_target: bool = False
    target_speed_min: float = 40.
    target_speed_max: float = 100.
    target_turn_fraction: float = 0.35
    target_turn_min_steps: int = 6
    target_turn_max_steps: int = 18
    pursuit: bool = False
    target_hits: int = 1


def advance_target(position, velocity, dt, size):
    """Reflect at a 20-unit margin; preserves speed (one crossing/axis per step)."""
    proposed=position+velocity*dt
    low=torch.full_like(proposed,20.)
    high=size-20
    crossed=(proposed<low)|(proposed>high)
    end=torch.where(proposed<low,2*low-proposed,torch.where(proposed>high,2*high-proposed,proposed))
    return end,torch.where(crossed,-velocity,velocity)


def tear_velocity(direction, player_velocity, physics):
    # Inherited velocity is captured at emission; it is NOT renormalized or
    # changed when the player subsequently turns or stops.
    if physics.inheritance_parallel >= 0:
        parallel=(player_velocity*direction).sum(-1,keepdim=True)*direction
        return (physics.tear_speed*direction + physics.inheritance_parallel*parallel
                + physics.movement_inheritance*(player_velocity-parallel))
    return physics.tear_speed * direction + physics.movement_inheritance * player_velocity


def swept_hit(start, end, target, radius):
    segment = end - start
    fraction = ((target - start) * segment).sum(-1) / segment.square().sum(-1).clamp_min(1e-8)
    closest = start + fraction.clamp(0, 1)[..., None] * segment
    return (target - closest).square().sum(-1) <= radius**2


class AlignmentEnv:
    slots = 8
    obs_dim = 10 + slots * 5

    def __init__(self, count, device, seed, physics=None):
        self.n, self.device = count, torch.device(device)
        self.p = physics or Physics()
        self.obs_dim = (51 if self.p.shot_budget else 50)+(2 if self.p.moving_target else 0)+(1 if self.p.pursuit else 0)
        if self.p.tear_lifetime / self.p.fire_interval >= self.slots:
            raise ValueError('Projectile buffer too small for lifetime/cadence')
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.move = torch.tensor([[0,0],[0,-1],[0,1],[-1,0],[1,0],[-1,-1],[1,-1],[-1,1],[1,1]],
                                 device=self.device, dtype=torch.float32)
        self.move /= self.move.norm(dim=-1, keepdim=True).clamp_min(1)
        self.pos = torch.zeros(count, 2, device=self.device)
        self.vel = torch.zeros_like(self.pos)
        self.target = torch.zeros_like(self.pos)
        self.target_vel = torch.zeros_like(self.pos)
        self.target_health = torch.full((count,),self.p.target_hits,device=self.device,dtype=torch.long)
        self.target_speed = torch.zeros(count,device=self.device)
        self.target_turns = torch.zeros(count,device=self.device,dtype=torch.bool)
        self.target_timer = torch.zeros(count,device=self.device,dtype=torch.long)
        self.bpos = torch.zeros(count, self.slots, 2, device=self.device)
        self.bvel = torch.zeros_like(self.bpos)
        self.life = torch.zeros(count, self.slots, device=self.device)
        self.cooldown = torch.zeros(count, device=self.device)
        self.steps = torch.zeros(count, device=self.device, dtype=torch.long)
        self.shots = torch.zeros(count, device=self.device, dtype=torch.long)
        self.reset(torch.ones(count, device=self.device, dtype=torch.bool))

    def reset(self, mask):
        # Generate full batches to avoid CPU/GPU synchronization on reset counts.
        random = torch.rand(self.n, 4, generator=self.rng, device=self.device)
        size=torch.tensor([self.p.room_width,self.p.room_height],device=self.device)
        pos = (0.25+random[:,:2]*0.5)*size
        sign = torch.where(pos < size/2, 1., -1.)
        target = pos + sign * (45 + random[:, 2:] * (size*0.35-45).clamp(0,100))
        if not self.p.native_coordinates:
            pos=random[:,:2]*torch.tensor([400.,160.],device=self.device)+120
            sign=torch.where(pos<torch.tensor([320.,200.],device=self.device),1.,-1.)
            target=pos+sign*(45+random[:,2:]*100)
        self.pos = torch.where(mask[:, None], pos, self.pos)
        self.target = torch.where(mask[:, None], target, self.target)
        self.vel[mask] = 0
        self.life[mask] = 0
        self.cooldown[mask] = 0
        self.steps[mask] = 0
        self.shots[mask] = 0
        self.target_health[mask] = self.p.target_hits
        if self.p.moving_target:
            heading=torch.rand(self.n,device=self.device,generator=self.rng)*6.283185307
            speed=self.p.target_speed_min+(self.p.target_speed_max-self.p.target_speed_min)*torch.rand(self.n,device=self.device,generator=self.rng)
            velocity=torch.stack((heading.cos(),heading.sin()),dim=-1)*speed[:,None]
            self.target_vel=torch.where(mask[:,None],velocity,self.target_vel)
            turns=torch.rand(self.n,device=self.device,generator=self.rng)<self.p.target_turn_fraction
            self.target_turns=torch.where(mask,turns,self.target_turns)
            timer=torch.randint(self.p.target_turn_min_steps,self.p.target_turn_max_steps+1,(self.n,),device=self.device,generator=self.rng)
            self.target_timer=torch.where(mask,timer,self.target_timer)
            self.target_speed=torch.where(mask,speed,self.target_speed)
            if self.p.pursuit:
                delta=self.pos-self.target
                velocity=delta/delta.norm(dim=-1,keepdim=True).clamp_min(1e-6)*self.target_speed[:,None]
                self.target_vel=torch.where(mask[:,None],velocity,self.target_vel)

    def obs(self):
        active = self.life > 0
        bullets = torch.cat(((self.bpos-self.pos[:,None])/320, self.bvel/400,
                             self.life[...,None]/self.p.tear_lifetime), dim=-1)
        bullets = torch.where(active[...,None], bullets, 0).flatten(1)
        result = torch.cat(((self.target-self.pos)/320, self.vel/100, self.pos/640,
                          (torch.tensor([self.p.room_width,self.p.room_height], device=self.device)-self.pos)/640,
                          self.cooldown[:,None]/self.p.fire_interval,
                          self.steps[:,None]/self.p.max_steps, bullets), dim=-1)
        if self.p.shot_budget:
            result=torch.cat((result,((self.p.shot_budget-self.shots)/self.p.shot_budget)[:,None]),dim=-1)
        if self.p.moving_target:
            result=torch.cat((result,self.target_vel/100),dim=-1)
        if self.p.pursuit:
            result=torch.cat((result,(self.target_health/self.p.target_hits)[:,None]),dim=-1)
        return result

    @torch.no_grad()
    def step(self, actions):
        old_target=self.target.clone()
        if self.p.pursuit:
            delta=self.pos-self.target
            self.target_vel=delta/delta.norm(dim=-1,keepdim=True).clamp_min(1e-6)*self.target_speed[:,None]
            self.target=self.target+self.target_vel*self.p.dt
        elif self.p.moving_target:
            self.target_timer-=1
            change=self.target_turns & (self.target_timer<=0)
            heading=torch.rand(self.n,device=self.device,generator=self.rng)*6.283185307
            speed=self.target_vel.norm(dim=-1)
            velocity=torch.stack((heading.cos(),heading.sin()),dim=-1)*speed[:,None]
            self.target_vel=torch.where(change[:,None],velocity,self.target_vel)
            timer=torch.randint(self.p.target_turn_min_steps,self.p.target_turn_max_steps+1,(self.n,),device=self.device,generator=self.rng)
            self.target_timer=torch.where(change,timer,self.target_timer)
            self.target,self.target_vel=advance_target(self.target,self.target_vel,self.p.dt,
                torch.tensor([self.p.room_width,self.p.room_height],device=self.device))
        move, shoot = actions // 5, actions % 5
        self.vel = self.p.velocity_retention*self.vel + (1-self.p.velocity_retention)*self.p.player_speed*self.move[move]
        old_pos = self.pos.clone()
        proposed = self.pos + self.vel*self.p.dt
        self.pos = proposed.clamp(torch.tensor([20.,20.],device=self.device),
                                  torch.tensor([self.p.room_width-20,self.p.room_height-20],device=self.device))
        self.vel = torch.where(self.pos != proposed, 0, self.vel)
        self.cooldown = (self.cooldown-self.p.dt).clamp_min(0)
        fire = (shoot != 0) & (self.cooldown <= 1e-6)
        if self.p.shot_budget:
            fire &= self.shots < self.p.shot_budget
        slot = self.life.argmin(dim=1)
        rows = torch.arange(self.n, device=self.device)
        fire &= self.life[rows,slot] <= 0
        self.shots += fire
        self.bpos[rows,slot] = torch.where(fire[:,None], self.pos, self.bpos[rows,slot])
        velocity = tear_velocity(self.move[shoot], self.vel, self.p)
        self.bvel[rows,slot] = torch.where(fire[:,None], velocity, self.bvel[rows,slot])
        self.life[rows,slot] = torch.where(fire, self.p.tear_lifetime, self.life[rows,slot])
        self.cooldown = torch.where(fire, self.p.fire_interval, self.cooldown)
        end = self.bpos + self.bvel * self.life.clamp(0,self.p.dt)[...,None]
        # Relative swept trajectories catch a target crossing a tear between observations.
        target_at_end=old_target[:,None]+(self.target-old_target)[:,None]*(self.life.clamp(0,self.p.dt)/self.p.dt)[...,None]
        bullet_hit = (self.life > 0) & swept_hit(self.bpos-old_target[:,None],end-target_at_end,
                                                torch.zeros_like(end),self.p.hit_radius)
        hit = bullet_hit.any(-1)
        damage=bullet_hit.sum(-1).clamp_max(self.target_health)
        self.target_health-=damage
        was_active=self.life>0
        self.bpos = end
        self.life = (self.life-self.p.dt).clamp_min(0)
        outside = ((end < 0) | (end > torch.tensor([self.p.room_width,self.p.room_height],device=self.device))).any(-1)
        self.life[outside] = 0
        misses=(was_active & (self.life<=0) & ~bullet_hit).sum(-1)
        self.life[bullet_hit]=0  # A normal tear disappears on its first impact.
        contact = swept_hit(old_pos-old_target,self.pos-self.target,torch.zeros_like(self.pos),self.p.contact_radius)
        self.steps += 1
        killed=self.target_health<=0
        success = killed & ~contact
        done = killed | contact | (self.steps >= self.p.max_steps)
        exhausted=(self.shots>=self.p.shot_budget)&(self.life<=0).all(-1) if self.p.shot_budget else torch.zeros_like(done)
        done |= exhausted
        wall=(move!=0)&((self.pos-old_pos).norm(dim=-1)<0.5)
        reward = (success.float() - contact.float() - 0.002 - self.p.shot_cost*fire
                  - self.p.miss_cost*misses - self.p.wall_cost*wall)
        if self.p.pursuit:
            reward+=0.2*damage/self.p.target_hits*(~contact)
        timeout=(self.steps>=self.p.max_steps)&~killed&~contact
        info = {'success': success, 'contact': contact, 'done': done,'fired':fire,'misses':misses,'wall':wall,'damage':damage,'timeout':timeout}
        self.reset(done)
        return self.obs(), reward, done, info


class Policy(nn.Module):
    def __init__(self,obs_dim=50):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        self.actor = nn.Linear(64,45)
        self.value = nn.Linear(64,1)
    def forward(self,obs):
        x=self.body(obs)
        return self.actor(x),self.value(x).squeeze(-1)


@torch.no_grad()
def evaluate(policy, device, physics, random=False, seed=90001):
    env=AlignmentEnv(512,device,seed,physics)
    alive=torch.ones(env.n,dtype=torch.bool,device=env.device)
    successes=torch.zeros_like(alive)
    contacts=torch.zeros_like(alive)
    steps=torch.zeros(env.n,device=env.device)
    shots=torch.zeros_like(steps);misses=torch.zeros_like(steps);wall=torch.zeros_like(steps)
    damage=torch.zeros_like(steps)
    timeouts=torch.zeros_like(alive)
    for _ in range(physics.max_steps):
        if random:
            action=torch.randint(45,(env.n,),device=env.device,generator=env.rng)
        else:
            action=policy(env.obs())[0].argmax(-1)
        _,_,done,info=env.step(action)
        successes |= alive & info['success']
        contacts |= alive & info['contact']
        steps += alive
        shots += alive*info['fired'];misses += alive*info['misses'];wall += alive*info['wall']
        damage += alive*info['damage']
        timeouts |= alive & info['timeout']
        alive &= ~done
    return dict(episodes=env.n,hit_rate=successes.float().mean().item(),
                contact_rate=contacts.float().mean().item(),mean_steps=steps.mean().item(),
                mean_shots=shots.mean().item(),expired_misses=misses.sum().item(),
                hits_per_emitted_shot=(damage.sum()/shots.sum().clamp_min(1)).item(),
                wall_step_rate=(wall.sum()/steps.sum().clamp_min(1)).item(),
                mean_impacts=damage.mean().item(),timeout_rate=timeouts.float().mean().item(),
                success_definition='kill_without_contact' if physics.pursuit else 'first_hit_without_contact')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--updates',type=int,default=200)
    parser.add_argument('--envs',type=int,default=512)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--precision',action='store_true',help='Three-shot diagnostic with trace-informed physics; not fully calibrated')
    parser.add_argument('--moving',action='store_true',help='Implies precision: random moving target, current velocity observed')
    parser.add_argument('--pursuit',action='store_true',help='Fixed empty room; four-hit chaser, unlimited shots, no-contact kills')
    parser.add_argument('--inheritance',type=float,default=None,
                        help='Provisional movement inheritance coefficient (training default: 1)')
    parser.add_argument('--evaluate',type=Path,help='Evaluate a saved frozen simulation checkpoint; no training')
    args=parser.parse_args()
    if args.updates<1 or args.envs<1 or (args.inheritance is not None and not 0<=args.inheritance<=3):
        parser.error('Invalid budget or inheritance')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    physics=Physics(movement_inheritance=1. if args.inheritance is None else args.inheritance)
    if args.precision or args.moving or args.pursuit:
        physics=Physics(player_speed=237.,velocity_retention=0.6967**3,tear_speed=300.,
                        movement_inheritance=0.60,inheritance_parallel=0.,fire_interval=11/30,
                        room_width=520.,room_height=280.,shot_budget=3,shot_cost=.03,miss_cost=.05,
                        wall_cost=.02,native_coordinates=True,player_velocity_scale=59.59)
        physics.moving_target=args.moving or args.pursuit
        if args.pursuit:
            physics.pursuit=True
            physics.target_hits=4
            physics.target_speed_min=60.
            physics.target_speed_max=120.
            physics.target_turn_fraction=0.
            physics.shot_budget=0
            physics.shot_cost=0.
            physics.miss_cost=0.
            physics.max_steps=150
    if args.evaluate:
        saved=torch.load(args.evaluate,map_location=args.device,weights_only=True)
        if saved.get('schema_version')!=1:
            raise ValueError('Unsupported alignment checkpoint')
        physics=Physics(**saved['physics'])
        if args.inheritance is not None:
            physics.movement_inheritance=args.inheritance
        policy=Policy(saved['state_dict']['body.0.weight'].shape[1]).to(args.device).eval()
        policy.load_state_dict(saved['state_dict'])
        print(json.dumps(dict(physics=asdict(physics),trained=False,live_validated=False,
                              final=evaluate(policy,args.device,physics)),indent=2))
        return
    env=AlignmentEnv(args.envs,args.device,args.seed+100,physics)
    policy=Policy(env.obs_dim).to(env.device)
    optimizer=torch.optim.Adam(policy.parameters(),lr=0.001)
    output=Path(__file__).resolve().parents[1]/'data/alignment_lab'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True)
    baseline=evaluate(policy,args.device,physics,random=True)
    initial=evaluate(policy,args.device,physics)
    if args.device=='cuda': torch.cuda.synchronize()
    start=time.perf_counter()
    with (output/'metrics.jsonl').open('w') as log:
        obs=env.obs()
        for update in range(args.updates):
            values=[];logprobs=[];rewards=[];dones=[];entropies=[]
            for _ in range(32):
                logits,value=policy(obs)
                dist=torch.distributions.Categorical(logits=logits)
                action=dist.sample()
                obs,reward,done,_=env.step(action)
                values.append(value);logprobs.append(dist.log_prob(action))
                rewards.append(reward);dones.append(done);entropies.append(dist.entropy())
            with torch.no_grad():
                returns=policy(obs)[1]
                targets=[]
                for r,d in zip(reversed(rewards),reversed(dones)):
                    returns=r+0.99*(~d)*returns
                    targets.append(returns)
                targets=torch.stack(list(reversed(targets)))
            value=torch.stack(values)
            advantage=targets-value
            loss=-(torch.stack(logprobs)*advantage.detach()).mean()+0.5*advantage.square().mean()-0.01*torch.stack(entropies).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(),0.5)
            optimizer.step()
            if (update+1)%20==0 or update==0:
                row=dict(update=update+1,loss=loss.item(),transitions=(update+1)*32*args.envs)
                log.write(json.dumps(row)+'\n');log.flush();print(row,flush=True)
    if args.device=='cuda': torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    final=evaluate(policy,args.device,physics)
    report=dict(config=vars(args),physics=asdict(physics),physics_calibrated=False,
                algorithm='A2C, 32-step returns',from_scratch=True,live_validated=False,
                random=baseline,initial=initial,final=final,training_seconds=elapsed,
                transitions=args.updates*32*args.envs,checkpoint_selection='final fixed budget')
    torch.save(dict(state_dict=policy.cpu().state_dict(),physics=asdict(physics),schema_version=1,
                    precision=bool(args.precision or args.moving or args.pursuit)),output/'policy.pt')
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));print(f'Output: {output}')


if __name__=='__main__': main()
