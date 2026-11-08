"""Opt-in CUDA graph backend for the existing, terminal-safe Isaac simulator.

No new equations or altered RNG sampling. Install in research/gpu_simulator.py.
Construction captures one fixed-size, fixed-configuration simulation step.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from research.rapid_physics import StrictNativeShooterEnv
from combat_suite import CombatSuiteEnv

BACKEND_VERSION = "native-v2-cuda-graph-v1"


class GraphReadyNativeEnv(StrictNativeShooterEnv):
    """Same reset sequence with its CPU-origin constant allocated before capture.

Keep all random draws, including those for lanes that do not reset. Skipping
them would change future attacks and resets at the same seed.
"""

    def __init__(self, *args, **kwargs):
        self._spawn_extent = None
        super().__init__(*args, **kwargs)

    def reset(self, mask):
        if self.keep_terminal:
            return
        if self._spawn_extent is None:
            self._spawn_extent = torch.tensor([40., 20.], device=self.device)
        # CombatEnv.reset: preserve arithmetic ordering and RNG call shapes.
        r = torch.rand(self.n, 15, generator=self.rng, device=self.device)
        pos = self.size / 2 + (r[:, :2] - .5) * self._spawn_extent
        angles = (r[:, 2:3] * 6.283185307
                  + torch.arange(4, device=self.device)[None] * 6.283185307 / self.enemies
                  + (r[:, 3:7] - .5) * .6)
        radius = 90 + 20 * r[:, 7:11]
        targets = pos[:, None] + torch.stack((angles.cos(), angles.sin()), -1) * radius[:, :, None]
        self.pos = torch.where(mask[:, None], pos, self.pos)
        self.targets = torch.where(mask[:, None, None], targets, self.targets)
        self.speed = torch.where(mask[:, None], self.p.target_speed_min
                                 + r[:, 11:15] * (self.p.target_speed_max - self.p.target_speed_min), self.speed)
        hp = (torch.arange(4, device=self.device) < self.enemies).long() * self.p.target_hits
        self.hp = torch.where(mask[:, None], hp, self.hp)
        delta = self.pos[:, None] - self.targets
        velocity = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6) * self.speed[:, :, None]
        self.target_vel = torch.where(mask[:, None, None], velocity, self.target_vel)
        self.vel[mask] = 0
        self.life[mask] = 0
        self.cooldown[mask] = 0
        self.steps[mask] = 0
        # ShooterEnv.reset, then NativeShooterEnv.reset.
        c = self.config
        timers = torch.randint(c.first_attack_min_steps, c.first_attack_max_steps + 1,
                               (self.n, 4), generator=self.rng, device=self.device)
        speeds = c.projectile_speed_min + torch.rand(self.n, 4, generator=self.rng, device=self.device) * (c.projectile_speed_max - c.projectile_speed_min)
        self.attack_timer = torch.where(mask[:, None], timers, self.attack_timer)
        self.attack_speed = torch.where(mask[:, None], speeds, self.attack_speed)
        self.hostile_life[mask] = 0
        self.immunity[mask] = 0
        self.native_velocity[mask] = 0
        self.last_action[mask] = 0
        self.fire_clock[mask] = 0
        self.eye[mask] = 1


class GraphReadyCombatSuiteEnv(CombatSuiteEnv, GraphReadyNativeEnv):
    """Original suite movement, rejection sampling and roles, strict terminals."""


class CudaGraphSimulator:
    """Persistent GPU state and packed outputs, replayed with one graph launch.

    `step` returns owning output buffers: safe to store observations in replay.
    `step_view` avoids these copies but is invalidated by the next step; clone
    anything retained. Actions must already be int64 CUDA tensors of shape [n].
    Use a single CUDA stream. Parameters and terminal mode are fixed at capture.
    Mutate state in place (env.pos.copy_(...)); do not rebind state tensors.
    `reset(mask)` supports explicit resets even with keep_terminal=True.
    A new graph is needed after changing physics, batch size or terminal mode.
    """

    def __init__(self, env, warmup=3):
        if not isinstance(env, GraphReadyNativeEnv):
            raise TypeError("Use GraphReadyNativeEnv or GraphReadyCombatSuiteEnv")
        if env.device.type != "cuda":
            raise ValueError("CUDA graph backend requires a CUDA environment")
        self.env = env
        self.actions = torch.zeros(env.n, device=env.device, dtype=torch.long)
        self._signature = self._config_signature()
        with torch.cuda.device(env.device), torch.no_grad():
            self._stream = torch.cuda.current_stream(env.device)
            original = self._snapshot()
            rng = env.rng.get_state()
            # Initialize lazy diagnostic dtypes (attacks starts float, then long).
            env.step(self.actions)
            self._slots = self._tensor_slots()
            self._restore(original)
            env.rng.set_state(rng)
            saved = self._snapshot()
            stream = torch.cuda.Stream(device=env.device)
            stream.wait_stream(self._stream)
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    self._capture_step()
            self._stream.wait_stream(stream)
            self._restore(saved)
            env.rng.set_state(rng)
            self._stream.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            self.graph.register_generator_state(env.rng)
            with torch.cuda.graph(self.graph, stream=stream):
                self._packed, self._layout, self._info_keys = self._capture_step()
            self._stream.wait_stream(stream)
            self._restore(saved)
            env.rng.set_state(rng)
        self._signature = self._config_signature()

    def _tensor_slots(self):
        return {k: v for k, v in vars(self.env).items() if isinstance(v, torch.Tensor)}

    def _snapshot(self):
        return {k: v.clone() for k, v in self._tensor_slots().items()}

    def _restore(self, saved):
        for k, v in saved.items():
            self._slots[k].copy_(v)
            setattr(self.env, k, self._slots[k])

    def _bind(self):
        # Python rebindings inside the eager implementation must become feedback
        # into the stable graph inputs. Copy nodes are themselves captured.
        for k, slot in self._slots.items():
            current = getattr(self.env, k)
            if current is not slot:
                slot.copy_(current)
                setattr(self.env, k, slot)

    def _capture_step(self):
        obs, reward, done, info = self.env.step(self.actions)
        self._bind()
        values = [obs, reward, done, *info.values()]
        groups = {}
        layout = []
        for value in values:
            group = groups.setdefault(value.dtype, [])
            offset = sum(x.numel() for x in group)
            group.append(value)
            layout.append((value.dtype, offset, value.numel(), value.shape))
        packed = {dtype: torch.cat([v.reshape(-1) for v in group])
                  for dtype, group in groups.items()}
        return packed, layout, tuple(info)

    def _config_signature(self):
        e = self.env
        return (e.n, e.enemies, e.keep_terminal, e.damage_ends_episode,
                tuple(asdict(e.native).items()), tuple(asdict(e.p).items()),
                tuple(asdict(e.config).items()))

    def _check(self, actions):
        e = self.env
        if actions.dtype != torch.long or actions.device != self.actions.device or actions.shape != self.actions.shape:
            raise ValueError("Actions must be int64 on the captured CUDA device with shape [n]")
        if self._config_signature() != self._signature:
            raise RuntimeError("Simulator configuration changed; capture a new graph")
        if torch.cuda.current_stream(e.device) != self._stream:
            raise RuntimeError("Use the CUDA stream that created this simulator")
        if any(getattr(e, k) is not slot for k, slot in self._slots.items()):
            raise RuntimeError("State tensor rebound; use in-place copy_ or reset(mask)")

    def _unpack(self, packed):
        values = [packed[dtype].narrow(0, start, count).view(shape)
                  for dtype, start, count, shape in self._layout]
        return values[0], values[1], values[2], dict(zip(self._info_keys, values[3:]))

    @torch.no_grad()
    def step(self, actions):
        self._check(actions)
        self.actions.copy_(actions)
        self.graph.replay()
        # Two dtype-group copies instead of one allocation/launch per output.
        return self._unpack({dtype: v.clone() for dtype, v in self._packed.items()})

    @torch.no_grad()
    def step_view(self, actions):
        self._check(actions)
        self.actions.copy_(actions)
        self.graph.replay()
        return self._unpack(self._packed)

    @torch.no_grad()
    def reset(self, mask):
        if mask.dtype != torch.bool or mask.device != self.actions.device or mask.shape != self.actions.shape:
            raise ValueError("Reset mask must be bool on the captured device with shape [n]")
        self._check(self.actions)
        terminal = self.env.keep_terminal
        try:
            self.env.keep_terminal = False
            self.env.reset(mask)
        finally:
            self.env.keep_terminal = terminal
        self._bind()

    def obs(self):
        return self.env.obs()

    @property
    def n(self):
        return self.env.n
