"""Independently selectable native simulator stages; GPU validation pending.

The all-off staged path is a control for orchestration overhead. Each stage has
a tensor fallback. CPU-native execution exercises the exact shared C++ helpers.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from contextlib import nullcontext
import torch
from research.gpu_simulator import GraphReadyNativeEnv, GraphReadyCombatSuiteEnv
from physics_v2 import player_tick, native_tear_velocity
from alignment_lab import swept_hit
from multi_combat_lab import first_contact
from research.native_runtime import NativeRuntime, config_values


@dataclass(frozen=True)
class Optimizations:
    movement: bool = False
    attacks: bool = False
    projectiles: bool = False
    terminal_mask: bool = False
    sparse_reset: bool = False
    direct_observation: bool = False
    tick_fusion: bool = False
    decision_fusion: bool = False
    mapping: str = 'thread'

    def __post_init__(self):
        if self.mapping not in ('thread', 'warp'):
            raise ValueError('mapping must be thread or warp')
        if self.tick_fusion and not (self.movement and self.attacks and self.projectiles):
            raise ValueError('tick_fusion requires movement, attacks and projectiles')
        if self.decision_fusion and not (self.tick_fusion and self.terminal_mask):
            raise ValueError('decision_fusion requires tick_fusion and terminal_mask')
        if self.mapping == 'warp' and self.tick_fusion and not self.decision_fusion:
            raise ValueError('warp tick fusion requires decision_fusion; staged warp projectiles are supported')


ALL = Optimizations(True, True, True, True, True, True, True, True)
STATE = ('pos vel targets target_vel speed bpos bvel life cooldown hostile_pos '
         'hostile_vel hostile_life attack_speed native_velocity eye hp steps '
         'attack_timer immunity last_action fire_clock attacks').split()


class StagedMixin:
    def __init__(self, *args, optimizations=Optimizations(), cache=None, native_arch='89', **kwargs):
        self._optimizations = optimizations
        self.native_cache = cache
        self.native_arch = native_arch
        self._prepared = False
        self._ops = None
        self.profile_stages = False
        super().__init__(*args, **kwargs)

    @property
    def optimizations(self):
        return self._optimizations

    def prepare(self):
        """Allocate/compile before capture, after setting physics and timeouts."""
        if self._prepared:
            if tuple(config_values(self)) != self._numeric_signature:
                raise RuntimeError('Native configuration changed after prepare; construct a new environment')
            return self
        n, d = self.n, self.device
        if self.pos.dtype != torch.float32 or self.hp.dtype != torch.int64:
            raise TypeError('Native ABI requires float32 state and int64 counters')
        self._mixed = hasattr(self, 'is_chaser')
        self._roles = self.is_chaser if self._mixed else torch.zeros(n, 4, dtype=torch.bool, device=d)
        self._counts = self.counts if self._mixed else torch.full((n,), self.enemies, dtype=torch.long, device=d)
        self._shooters = self.shooters if self._mixed else self._counts
        self._chaser_hits = self.chaser_hits if self._mixed else self._counts
        self.attacks = self.attacks.long()
        self._old_pos = torch.empty_like(self.pos)
        self._old_targets = torch.empty_like(self.targets)
        self._prev_pos = torch.empty_like(self.pos)
        self._alive = torch.empty_like(self.hp, dtype=torch.bool)
        self._active = torch.ones(n, dtype=torch.bool, device=d)
        self._totals = torch.empty(n, 4, device=d)
        self._flags = torch.empty(n, 3, dtype=torch.bool, device=d)
        self._mid = torch.empty_like(self.pos)
        self._random = torch.empty(3, n, 4, dtype=torch.long, device=d)
        self._obs = torch.empty(n, 236, device=d)
        self._mask = torch.empty(n, dtype=torch.bool, device=d)
        self._rbase = torch.empty(n, 15, device=d)
        self._rtimer = torch.empty(n, 4, dtype=torch.long, device=d)
        self._rspeed = torch.empty(n, 4, device=d)
        self._rpos = torch.empty(n, 2, device=d)
        self._rangle = torch.empty(4, n, 64, device=d)
        self._rradius = torch.empty_like(self._rangle)
        self._rch_speed = torch.empty(n, 4, device=d)
        self._rows = torch.arange(n, device=d)
        o = self.optimizations
        if o.movement or o.attacks or o.projectiles or o.sparse_reset or o.direct_observation:
            self._ops = NativeRuntime(self, cache=self.native_cache, arch=self.native_arch)
        self._numeric_signature = tuple(config_values(self))
        self._prepared = True
        return self

    def _range(self, name):
        return torch.profiler.record_function('isaac::' + name) if self.profile_stages else nullcontext()

    def _write(self, name, value):
        old = getattr(self, name)
        if self.optimizations.terminal_mask:
            mask = self._active.reshape((self.n,) + (1,) * (old.ndim - 1))
            value = torch.where(mask, value, old)
        old.copy_(value)

    def _draw_attacks(self, tick=0):
        torch.randint(self.config.interval_min_steps, self.config.interval_max_steps + 1,
                      (self.n, 4), generator=self.rng, device=self.device, out=self._random[tick])

    def _torch_attacks(self):
        timer = torch.where(self._roles, 1000000, self.attack_timer) if self._mixed else self.attack_timer
        timer = timer - 1
        self._write('immunity', (self.immunity - 1).clamp_min(0))
        fire = (self.hp > 0) & (timer <= 0)
        self._write('attacks', fire.sum(-1))
        self._totals[:, 2].add_(self._active * self.attacks)
        for enemy in range(4):
            slot = self.hostile_life.argmin(-1)
            emit = fire[:, enemy] & (self.hostile_life[self._rows, slot] <= 0)
            delta = self.pos - self.targets[:, enemy]
            velocity = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6) * self.attack_speed[:, enemy, None]
            if self.optimizations.terminal_mask:
                emit = emit & self._active
            self.hostile_pos[self._rows, slot] = torch.where(emit[:, None], self.targets[:, enemy], self.hostile_pos[self._rows, slot])
            self.hostile_vel[self._rows, slot] = torch.where(emit[:, None], velocity, self.hostile_vel[self._rows, slot])
            self.hostile_life[self._rows, slot] = torch.where(emit, self.config.projectile_lifetime, self.hostile_life[self._rows, slot])
        self._write('attack_timer', torch.where(fire, self._random[0], timer))
        if self._mixed:
            delta = self.pos[:, None] - self.targets
            velocity = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6) * self.speed[:, :, None] * (self.hp > 0)[:, :, None]
            proposed = self.targets + velocity / 30
            low, high = proposed < 20, proposed > self.size - 20
            self._write('targets', torch.where(low, 40-proposed, torch.where(high, 2*(self.size-20)-proposed, proposed)))
            self._write('target_vel', torch.where(low | high, -velocity, velocity))

    def _torch_motion(self, command):
        pos, vel, mid = player_tick(self.pos, self.native_velocity, self.move[command // 5], self.size, self.native)
        self._write('pos', pos)
        self._write('native_velocity', vel)
        self._write('vel', vel * self.p.player_velocity_scale)
        self._mid.copy_(mid)

    def _torch_projectiles(self, command):
        shoot = command % 5
        clock = self.fire_clock - 1
        slot = self.life.argmin(-1)
        fire = self._active & (shoot != 0) & (clock < 0) & (self.life[self._rows, slot] <= 0)
        direction = self.move[shoot]
        velocity = native_tear_velocity(direction, self._mid, self.native) * 30
        side = torch.stack((-direction[:, 1], direction[:, 0]), -1) * self.eye[:, None] * self.native.eye_offset
        self.bpos[self._rows, slot] = torch.where(fire[:, None], self.pos+side, self.bpos[self._rows, slot])
        self.bvel[self._rows, slot] = torch.where(fire[:, None], velocity, self.bvel[self._rows, slot])
        self.life[self._rows, slot] = torch.where(fire, self.p.tear_lifetime, self.life[self._rows, slot])
        clock = torch.where(fire, self.native.fire_ticks-1, clock)
        self._write('fire_clock', clock)
        self._write('cooldown', clock.clamp_min(0) / 30)
        self._write('eye', torch.where(fire, -self.eye, self.eye))
        end = self.bpos + self.bvel * self.life.clamp(0, 1/30)[:, :, None]
        target_end = self._old_targets[:, None] + (self.targets-self._old_targets)[:, None] * (self.life.clamp(0, 1/30)*30)[:, :, None, None]
        times = first_contact(self.bpos[:, :, None]-self._old_targets[:, None], end[:, :, None]-target_end, self.p.hit_radius)
        times = torch.where(self._alive[:, None] & (self.life > 0)[:, :, None], times, float('inf'))
        closest, index = times.min(-1)
        hits = closest.isfinite()
        counts = torch.zeros_like(self.hp).scatter_add(1, index, hits.long())
        damage = counts.clamp_max(self.hp)
        self._write('hp', self.hp - damage)
        self._write('bpos', end)
        dead = hits | ((end < 0) | (end > self.size)).any(-1)
        self._write('life', torch.where(dead, 0, (self.life-1/30).clamp_min(0)))
        hostile_active = self.hostile_life > 0
        hostile_end = self.hostile_pos + self.hostile_vel * self.hostile_life.clamp(0, 1/30)[:, :, None]
        hit = hostile_active & swept_hit(self.hostile_pos-self._old_pos[:, None], hostile_end-self.pos[:, None], torch.zeros_like(hostile_end), self.config.projectile_hit_radius)
        projectile = hit.any(-1) & (self.immunity <= 0)
        touch = (self._alive & swept_hit(self._old_pos[:, None]-self._old_targets, self.pos[:, None]-self.targets, torch.zeros_like(self.targets), self.p.contact_radius)).any(-1) & (self.immunity <= 0)
        contact = projectile | touch
        self._write('immunity', torch.where(contact, self.config.diagnostic_immunity_steps, self.immunity))
        self._write('hostile_pos', hostile_end)
        dead = hit | ((hostile_end < 0) | (hostile_end > self.size)).any(-1)
        self._write('hostile_life', torch.where(dead, 0, (self.hostile_life-1/30).clamp_min(0)))
        self._totals[:, 0].add_(self._active * damage.sum(-1))
        self._totals[:, 1].add_(self._active * (self._alive & (self.hp <= 0)).sum(-1))
        self._totals[:, 3].add_(self._active * fire)
        self._flags[:, 0].logical_or_(self._active & contact)
        self._flags[:, 1].logical_or_(self._active & projectile)
        self._flags[:, 2].logical_or_(self._active & touch)
        self._active.logical_and_(~((self.hp <= 0).all(-1) | (contact & self.damage_ends_episode)))

    @torch.no_grad()
    def step(self, actions):
        if actions.dtype != torch.int64 or actions.shape != (self.n,) or actions.device != self.pos.device:
            raise ValueError('Actions must be int64 on the state device with shape [n]')
        actions = actions.contiguous()
        self.prepare()
        o = self.optimizations
        self._active.fill_(True)
        self._totals.zero_()
        self._flags.zero_()
        self._prev_pos.copy_(self.pos)
        if o.decision_fusion:
            for tick in range(3):
                self._draw_attacks(tick)
            with self._range('decision'):
                name = 'decision_warp' if o.mapping == 'warp' else 'decision'
                self._ops.launch(name, action=actions, damage_ends=self.damage_ends_episode)
        else:
            for sub in range(3):
                with self._range('terminal_snapshot'):
                    frozen = {name: getattr(self, name).clone() for name in STATE} if sub and not o.terminal_mask else None
                    was_active = self._active.clone() if frozen else None
                self._old_pos.copy_(self.pos)
                self._old_targets.copy_(self.targets)
                self._alive.copy_(self.hp > 0)
                command = self.last_action if sub < self.native.command_delay_ticks else actions
                self._draw_attacks()
                if o.tick_fusion:
                    with self._range('tick'):
                        self._ops.launch('tick', action=command, masked=o.terminal_mask, damage_ends=self.damage_ends_episode)
                else:
                    with self._range('attacks'):
                        self._ops.launch('attacks', masked=o.terminal_mask) if o.attacks else self._torch_attacks()
                    with self._range('movement'):
                        self._ops.launch('motion', action=command, masked=o.terminal_mask) if o.movement else self._torch_motion(command)
                    with self._range('projectiles'):
                        if o.projectiles:
                            name = 'projectiles_warp' if o.mapping == 'warp' else 'projectiles'
                            self._ops.launch(name, action=command, masked=o.terminal_mask, damage_ends=self.damage_ends_episode)
                        else:
                            self._torch_projectiles(command)
                if frozen:
                    with self._range('terminal_restore'):
                        for name, previous in frozen.items():
                            current = getattr(self, name)
                            mask = was_active.reshape((self.n,) + (1,) * (current.ndim-1))
                            current.copy_(torch.where(mask, current, previous))
        self.last_action.copy_(actions)
        self.steps.add_(1)
        cleared = (self.hp <= 0).all(-1)
        contact, projectile, touch = self._flags.unbind(-1)
        success = cleared & ~contact
        timeout = (self.steps >= self.p.max_steps) & ~cleared & ~(contact & self.damage_ends_episode)
        done = cleared | (contact & self.damage_ends_episode) | timeout
        wall = (actions // 5 != 0) & ((self.pos-self._prev_pos).norm(dim=-1) < .5)
        damage, kills, attacks, fired = self._totals.unbind(-1)
        reward = success.float()-contact.float()-.002-self.p.wall_cost*wall+.05*damage*(~contact)
        # Own output tensors: scratch buffers are reused on the next call.
        info = dict(success=success, contact=contact.clone(), timeout=timeout, wall=wall,
                    projectile_damage=projectile.clone(), contact_damage=touch & ~projectile,
                    damage=damage.clone(), kills=kills.clone(), attacks=attacks.clone(), fired=fired.clone())
        with self._range('reset'):
            self.reset(done)
        with self._range('observations'):
            return self.obs(), reward, done, info

    def reset(self, mask):
        if mask.dtype != torch.bool or mask.shape != (self.n,) or mask.device != self.pos.device:
            raise ValueError('Reset mask must be bool on the state device with shape [n]')
        if not self._prepared:
            return super().reset(mask)
        if self.keep_terminal:
            return
        if not self.optimizations.sparse_reset:
            stable = {name: getattr(self, name) for name in STATE}
            super().reset(mask)
            for name, slot in stable.items():
                value = getattr(self, name)
                if value is not slot:
                    slot.copy_(value)
                    setattr(self, name, slot)
            return
        self._mask.copy_(mask)
        torch.rand(self.n, 15, generator=self.rng, device=self.device, out=self._rbase)
        c = self.config
        torch.randint(c.first_attack_min_steps, c.first_attack_max_steps+1, (self.n, 4), generator=self.rng, device=self.device, out=self._rtimer)
        torch.rand(self.n, 4, generator=self.rng, device=self.device, out=self._rspeed)
        if self._mixed:
            torch.rand(self.n, 2, generator=self.rng, device=self.device, out=self._rpos)
            for slot in range(4):
                torch.rand(self.n, 64, generator=self.rng, device=self.device, out=self._rangle[slot])
                torch.rand(self.n, 64, generator=self.rng, device=self.device, out=self._rradius[slot])
            torch.rand(self.n, 4, generator=self.rng, device=self.device, out=self._rch_speed)
        self._ops.launch('reset')

    def obs(self):
        if not self._prepared or not self.optimizations.direct_observation:
            return super().obs()
        self._ops.launch('observation')
        return self._obs.clone()


class StagedNativeEnv(StagedMixin, GraphReadyNativeEnv):
    pass


class StagedCombatSuiteEnv(StagedMixin, GraphReadyCombatSuiteEnv):
    pass
