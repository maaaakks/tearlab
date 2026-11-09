"""Versioned native-tick physics candidate. V1 simulators and checkpoints stay intact.

Native velocity is the engine's serialized velocity, not units/second.
One game tick integrates two half-steps; a policy decision spans three ticks.
This is an empirical candidate for base Isaac in the empty rectangular room.
"""
from dataclasses import dataclass,replace
import json
from pathlib import Path
import torch
from alignment_lab import swept_hit
from multi_combat_lab import first_contact
from shooter_lab import ShooterEnv,ShooterConfig,physics


@dataclass(frozen=True)
class NativePhysics:
    schema:int=1
    idle_retention:float=.775
    parallel_retention:float=.775
    reverse_retention:float=.71424
    transverse_retention:float=.62
    acceleration_half:float=.528204
    wall_margin:float=10.
    tear_inheritance:float=1.2
    fire_ticks:int=11
    command_delay_ticks:int=1
    eye_offset:float=4.
    wall_update_order:int=4
    wall_rebound:float=.1
    wall_bias:float=.1
    wall_correction_max:float=2.

    @classmethod
    def load(cls,path):
        values=json.loads(Path(path).read_text(encoding='utf-8'))['parameters']
        values.setdefault('wall_update_order',1)
        return cls(**values)


def player_tick(position,velocity,direction,size,cfg):
    """Return position, native velocity, and mid-tick velocity used by tears."""
    moving=direction.square().sum(-1,keepdim=True)>0
    def friction(v):
        projection=(v*direction).sum(-1,keepdim=True)
        parallel=projection*direction
        parallel_drag=torch.where(projection>=0,cfg.parallel_retention**.5,cfg.reverse_retention**.5)
        driven=parallel*parallel_drag+(v-parallel)*cfg.transverse_retention**.5
        return torch.where(moving,driven,v*cfg.idle_retention**.5)
    for half in range(2):
        proposed=position+velocity
        position=proposed.clamp(torch.full_like(size,cfg.wall_margin),size-cfg.wall_margin)
        collided=position!=proposed
        if cfg.wall_update_order in (3,4):
            incoming=velocity
            velocity=friction(incoming)+cfg.acceleration_half*direction
            inward=torch.where(proposed<cfg.wall_margin,1.,-1.)
            # Empirical native wall response AFTER drag/input, in each half-step.
            # Its small signed remainder preserves first/second-half contact phase.
            rebound=cfg.wall_rebound*velocity+inward*cfg.wall_bias
            if cfg.wall_update_order==4:
                position=proposed+(position-proposed).clamp(-cfg.wall_correction_max,cfg.wall_correction_max)
                collided=collided & (velocity*inward<0)
            velocity=torch.where(collided,rebound,velocity)
            if half==0:mid=velocity.clone()
            continue
        # The observed tick includes a collision-resolution phase between the
        # two velocity updates. Erasing momentum before the first update loses
        # tangential motion while sliding along a wall.
        if half==1 or cfg.wall_update_order==1:
            velocity=torch.where(collided,0.,velocity)
        velocity=friction(velocity)+cfg.acceleration_half*direction
        if half==0:mid=velocity.clone()
        if half==1 and cfg.wall_update_order==2:
            # Native serialized horizontal velocity is post-wall resolution.
            velocity[...,0]=torch.where(collided[...,0],0.,velocity[...,0])
    return position,velocity,mid


def native_tear_velocity(direction,mid_velocity,cfg):
    projection=(mid_velocity*direction).sum(-1,keepdim=True)
    transverse=mid_velocity-projection*direction
    # Native positive parallel inheritance accelerates a tear; opposite movement
    # does not slow its base forward speed. Symmetric under horizontal reflection.
    return 10*direction+cfg.tear_inheritance*(transverse+projection.clamp_min(0)*direction)


class NativeShooterEnv(ShooterEnv):
    """GPU-batched candidate: integer cooldown, native motion, three ticks/decision.

Attack timers and immunity are native ticks. Observations keep the existing
236-feature contract; the checkpoint must identify this distinct environment.
No reward for centering or handcrafted aiming/dodging is introduced.
"""
    def __init__(self,count,device,seed,enemies=1,native=None,diagnostic=False):
        self.native=native or NativePhysics()
        self.native_velocity=torch.zeros(count,2,device=device)
        self.last_action=torch.zeros(count,dtype=torch.long,device=device)
        self.fire_clock=torch.zeros(count,device=device,dtype=torch.long)
        self.eye=torch.ones(count,device=device)
        self.keep_terminal=False
        c=ShooterConfig(interval_min_steps=24,interval_max_steps=39,warning_steps=9,
            first_attack_min_steps=12,first_attack_max_steps=21,diagnostic_immunity_steps=30)
        p=replace(physics(),player_speed=60*self.native.acceleration_half/(1-self.native.parallel_retention**.5))
        super().__init__(count,device,seed,enemies,p,c,diagnostic)

    def reset(self,mask):
        if self.keep_terminal:return
        super().reset(mask)
        self.native_velocity[mask]=0;self.last_action[mask]=0;self.fire_clock[mask]=0
        self.eye[mask]=1

    @torch.no_grad()
    def step(self,actions):
        active=torch.ones(self.n,dtype=torch.bool,device=self.device)
        totals={k:torch.zeros(self.n,device=self.device) for k in ('damage','kills','attacks','fired')}
        contact_any=torch.zeros_like(active);projectile_any=contact_any.clone();touch_any=contact_any.clone()
        previous_position=self.pos.clone();rows=torch.arange(self.n,device=self.device)
        for sub in range(3):
            command=self.last_action if sub<self.native.command_delay_ticks else actions
            move,shoot=command//5,command%5
            old_pos=self.pos.clone();alive=self.hp>0
            # before_step uses native tick timers and real-valued projectile speeds.
            old_targets=self.targets.clone()
            self.before_step();totals['attacks']+=active*self.attacks
            self.pos,self.native_velocity,mid=player_tick(self.pos,self.native_velocity,self.move[move],self.size,self.native)
            self.vel=self.native_velocity*self.p.player_velocity_scale
            self.fire_clock-=1
            slot=self.life.argmin(-1)
            fire=active&(shoot!=0)&(self.fire_clock<0)&(self.life[rows,slot]<=0)
            direction=self.move[shoot]
            velocity=native_tear_velocity(direction,mid,self.native)*30
            side=torch.stack((-direction[:,1],direction[:,0]),-1)*self.eye[:,None]*self.native.eye_offset
            self.bpos[rows,slot]=torch.where(fire[:,None],self.pos+side,self.bpos[rows,slot])
            self.bvel[rows,slot]=torch.where(fire[:,None],velocity,self.bvel[rows,slot])
            self.life[rows,slot]=torch.where(fire,self.p.tear_lifetime,self.life[rows,slot])
            self.fire_clock=torch.where(fire,self.native.fire_ticks-1,self.fire_clock)
            self.cooldown=self.fire_clock.clamp_min(0)/30
            self.eye=torch.where(fire,-self.eye,self.eye)
            end=self.bpos+self.bvel*self.life.clamp(0,1/30)[:,:,None]
            target_end=old_targets[:,None]+(self.targets-old_targets)[:,None]*(self.life.clamp(0,1/30)*30)[:,:,None,None]
            times=first_contact(self.bpos[:,:,None]-old_targets[:,None],end[:,:,None]-target_end,self.p.hit_radius)
            times=torch.where(alive[:,None]&(self.life>0)[:,:,None],times,float('inf'))
            closest,index=times.min(-1);hits=closest.isfinite()
            counts=torch.zeros_like(self.hp).scatter_add(1,index,hits.long())
            damage=counts.clamp_max(self.hp);self.hp-=damage
            self.bpos=end;self.life=(self.life-1/30).clamp_min(0)
            self.life[hits|((end<0)|(end>self.size)).any(-1)]=0
            # Keep hostile propagation at the same native tick resolution.
            hostile_active=self.hostile_life>0
            hostile_end=self.hostile_pos+self.hostile_vel*self.hostile_life.clamp(0,1/30)[:,:,None]
            hit=hostile_active&swept_hit(self.hostile_pos-old_pos[:,None],hostile_end-self.pos[:,None],torch.zeros_like(hostile_end),self.config.projectile_hit_radius)
            projectile=hit.any(-1)&(self.immunity<=0)
            touch=(alive&swept_hit(old_pos[:,None]-old_targets,self.pos[:,None]-self.targets,torch.zeros_like(self.targets),self.p.contact_radius)).any(-1)&(self.immunity<=0)
            contact=projectile|touch
            self.immunity=torch.where(contact,self.config.diagnostic_immunity_steps,self.immunity)
            self.hostile_pos=hostile_end;self.hostile_life=(self.hostile_life-1/30).clamp_min(0)
            self.hostile_life[hit|((hostile_end<0)|(hostile_end>self.size)).any(-1)]=0
            totals['damage']+=active*damage.sum(-1);totals['fired']+=active*fire
            totals['kills']+=active*(alive&(self.hp<=0)).sum(-1)
            contact_any|=active&contact;projectile_any|=active&projectile;touch_any|=active&touch
            active&=~((self.hp<=0).all(-1)|(contact & self.damage_ends_episode))
        self.last_action=actions.clone();self.steps+=1
        cleared=(self.hp<=0).all(-1);success=cleared&~contact_any
        timeout=(self.steps>=self.p.max_steps)&~cleared&~(contact_any & self.damage_ends_episode)
        done=cleared|(contact_any & self.damage_ends_episode)|timeout
        wall=(actions//5!=0)&((self.pos-previous_position).norm(dim=-1)<.5)
        reward=success.float()-contact_any.float()-.002-self.p.wall_cost*wall+.05*totals['damage']*(~contact_any)
        info=dict(success=success,contact=contact_any,timeout=timeout,wall=wall,
                  projectile_damage=projectile_any,contact_damage=touch_any&~projectile_any,**totals)
        self.reset(done)
        return self.obs(),reward,done,info
