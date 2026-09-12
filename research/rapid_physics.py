"""Opt-in native physics with immutable terminal states inside a decision.

The historical native equations, timers, rewards and reset behavior are retained.
Only the handling of lanes that finish before the third native tick changes:
their physical state stays at the terminal tick. The original module and its
checkpoints are untouched. Fixed three-tick batch calls still count as three
issued simulator ticks, even when a terminal lane stops advancing earlier.
"""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from alignment_lab import swept_hit
from multi_combat_lab import first_contact
from physics_v2 import NativeShooterEnv, native_tear_velocity, player_tick
from combat_suite import CombatSuiteEnv

SIMULATOR_VERSION = "native-v2-terminal-freeze-v1"
BASELINE_SOURCE_SHA256 = "099df91816bdca3eaa7352763ae9d0552b3ad223fee6f87b023fb5cb841e2b18"


class StrictNativeShooterEnv(NativeShooterEnv):
    """NativeShooterEnv.step, versioned to freeze completed vector lanes."""

    def _lane_snapshot(self):
        # Room size and action directions are shared geometry, not lane state;
        # their first dimensions can accidentally equal n (2 or 9).
        return {name: value.clone() for name, value in vars(self).items()
                if name not in {"size", "move"} and isinstance(value, torch.Tensor)
                and value.ndim and value.shape[0] == self.n}

    def _restore_inactive(self, snapshot, active):
        for name, previous in snapshot.items():
            current = getattr(self, name)
            mask = active.reshape((self.n,) + (1,) * (current.ndim - 1))
            setattr(self, name, torch.where(mask, current, previous))

    @torch.no_grad()
    def step(self, actions):
        active=torch.ones(self.n,dtype=torch.bool,device=self.device)
        totals={k:torch.zeros(self.n,device=self.device) for k in ('damage','kills','attacks','fired')}
        contact_any=torch.zeros_like(active);projectile_any=contact_any.clone();touch_any=contact_any.clone()
        previous_position=self.pos.clone();rows=torch.arange(self.n,device=self.device)
        for sub in range(3):
            # Every lane is active on tick zero. Subsequent ticks may contain
            # completed lanes: retain all their per-lane state, not just scores.
            frozen = self._lane_snapshot() if sub else None
            command=self.last_action if sub<self.native.command_delay_ticks else actions
            move,shoot=command//5,command%5
            old_pos=self.pos.clone();alive=self.hp>0
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
            if frozen is not None:
                self._restore_inactive(frozen, active)
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


class StrictCombatSuiteEnv(CombatSuiteEnv, StrictNativeShooterEnv):
    """CombatSuite roles and spawn distribution with terminal-safe stepping."""

