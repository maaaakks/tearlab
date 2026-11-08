"""GPU-resident rollout and PPO network graphs; opt-in, unchanged task/math.

Keeps the original pilot as an immutable reference. Each optimization can be
selected independently. Compilation/capture is outside steady-state timing.
"""
from dataclasses import dataclass, asdict
import copy
import time
import torch
from torch import nn
from research import pilot_10m as p


@dataclass(frozen=True)
class Options:
    rollout_graph: bool = False
    network_graph: bool = False
    fused_adam: bool = False
    deferred_metrics: bool = False


class PolicyView(nn.Module):
    """Share live parameters without replacing the checkpoint's module names."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs, previous):
        return self.policy(obs, previous)


class ParallelArm(p.Arm):
    def __init__(self, *args, options=Options(), minibatch=4096, **kwargs):
        super().__init__(*args, **kwargs)
        if self.device != 'cuda':
            raise ValueError('ParallelArm explicitly requires CUDA')
        self.options = options
        self.minibatch = minibatch
        if options.fused_adam:
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4,
                                             eps=1e-5, fused=True)
        self.obs = self.obs.clone()
        self.previous = self.previous.clone()
        self._ones = torch.ones(self.n, device=self.device, dtype=torch.long)
        self._train_policy = self.policy
        if options.network_graph:
            if self.t*self.n % minibatch:
                raise ValueError('Network graph requires complete fixed-size minibatches')
            obs = torch.zeros(minibatch, 236, device=self.device)
            previous = torch.zeros(minibatch, device=self.device, dtype=torch.long)
            self._train_policy = torch.cuda.make_graphed_callables(
                PolicyView(self.policy), (obs, previous), num_warmup_iters=3)
            self.optimizer.zero_grad(set_to_none=True)
        self.collect_graph = None
        if options.rollout_graph:
            self._capture_collection()

    @torch.no_grad()
    def _collect(self):
        b = self.batch
        for t in range(self.t):
            b['obs'][t].copy_(self.obs)
            b['previous'][t].copy_(self.previous)
            logits, value = self.policy(self.obs, self.previous)
            logp = logits.log_softmax(-1)
            action = torch.multinomial(logp.exp(), 1, generator=self.action_rng).squeeze(-1)
            # bincount has a data-dependent output shape even with minlength.
            # scatter_add has the same integer counts with a fixed output buffer.
            self.action_counts.scatter_add_(0, action, self._ones)
            obs, _, done, info = self.env.step(action)
            self.sim._bind()
            reward, parts = p.task_reward(info, self.initial_hp, self.dense)
            for key, val in [('value', value), ('action', action),
                             ('logp', logp.gather(1, action[:, None]).squeeze(1)),
                             ('reward', reward), ('done', done),
                             ('timeout', info['timeout']),
                             ('terminal_obs', info['terminal_observation'])]:
                b[key][t].copy_(val)
            self.totals.add_(torch.stack((done.sum(), info['success'].sum(),
                info['contact'].sum(), info['damage'].sum(), parts['task'].sum(),
                parts['progress'].sum(), parts['time'].sum(), info['fired'].sum())))
            self.obs.copy_(obs)
            self.previous.copy_(torch.where(done, 45, action))
        return self.policy(self.obs, self.previous)[1]

    def _capture_collection(self):
        saved = self.sim._snapshot()
        carry = [x.clone() for x in (self.obs, self.previous, self.totals, self.action_counts)]
        erng, arng = self.env.rng.get_state(), self.action_rng.get_state()

        def restore():
            self.sim._restore(saved)
            for target, value in zip((self.obs, self.previous, self.totals, self.action_counts), carry):
                target.copy_(value)
            self.env.rng.set_state(erng)
            self.action_rng.set_state(arng)

        current = torch.cuda.current_stream()
        stream = torch.cuda.Stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._collect()
        current.wait_stream(stream)
        restore()
        current.synchronize()
        graph = torch.cuda.CUDAGraph()
        graph.register_generator_state(self.env.rng)
        graph.register_generator_state(self.action_rng)
        with torch.cuda.graph(graph, stream=stream):
            self._bootstrap = self._collect()
        current.wait_stream(stream)
        restore()
        self.collect_graph = graph
        self._capture_stream = current

    def _validate_capture(self):
        self.sim._check(self.sim.actions)
        if torch.cuda.current_stream() != self._capture_stream:
            raise RuntimeError('Rollout must replay on its creation stream')

    def update(self, epochs=2, minibatch=None):
        minibatch = self.minibatch if minibatch is None else minibatch
        if self.options.network_graph and minibatch != self.minibatch:
            raise ValueError('Captured network requires the configured minibatch')
        torch.cuda.synchronize()
        start = time.perf_counter()
        b = self.batch
        with torch.no_grad():
            if self.collect_graph is not None:
                self._validate_capture()
                self.collect_graph.replay()
                bootstrap = self._bootstrap
            else:
                # Original simulator graph: independent ablation of other flags.
                for t in range(self.t):
                    b['obs'][t].copy_(self.obs)
                    b['previous'][t].copy_(self.previous)
                    logits, value = self.policy(self.obs, self.previous)
                    logp = logits.log_softmax(-1)
                    action = torch.multinomial(logp.exp(), 1, generator=self.action_rng).squeeze(-1)
                    self.action_counts.add_(torch.bincount(action, minlength=45))
                    obs, _, done, info = self.sim.step(action)
                    reward, parts = p.task_reward(info, self.initial_hp, self.dense)
                    for key, val in [('value', value), ('action', action),
                        ('logp', logp.gather(1, action[:, None]).squeeze(1)), ('reward', reward),
                        ('done', done), ('timeout', info['timeout']),
                        ('terminal_obs', info['terminal_observation'])]:
                        b[key][t].copy_(val)
                    self.totals.add_(torch.stack((done.sum(), info['success'].sum(),
                        info['contact'].sum(), info['damage'].sum(), parts['task'].sum(),
                        parts['progress'].sum(), parts['time'].sum(), info['fired'].sum())))
                    self.obs.copy_(obs)
                    self.previous.copy_(torch.where(done, 45, action))
                bootstrap = self.policy(self.obs, self.previous)[1]
            timeout_values = torch.zeros_like(b['value'])
            index = b['timeout'].flatten().nonzero().squeeze(-1)
            if len(index):
                value = self.policy(b['terminal_obs'].flatten(0, 1)[index],
                                    b['action'].flatten()[index])[1]
                timeout_values.view(-1)[index] = value
            advantage, target = p.gae(b['reward'], b['done'], b['timeout'],
                                     b['value'], bootstrap, timeout_values)
            advantage = (advantage-advantage.mean())/advantage.std().clamp_min(1e-8)
        torch.cuda.synchronize()
        collected = time.perf_counter()
        steps, klmax, clip_mean, entropy_mean = 0, 0., 0., 0.
        metrics = torch.zeros(2, device=self.device)
        flat_obs = b['obs'].flatten(0, 1)
        flat_prev = b['previous'].flatten()
        old = b['logp'].flatten()
        for _ in range(epochs):
            permutation = torch.randperm(self.t*self.n, device=self.device, generator=self.shuffle_rng)
            for ids in permutation.split(minibatch):
                logits, value = self._train_policy(flat_obs[ids], flat_prev[ids])
                lp = logits.log_softmax(-1)
                probs = lp.exp()
                entropy = -(probs*lp).sum(-1).mean()
                selected = lp.gather(1, b['action'].flatten()[ids, None]).squeeze(1)
                delta = selected-old[ids]
                kl = (delta.exp()-1-delta).mean()
                loss = p.clipped_actor_loss(selected, old[ids], advantage.flatten()[ids])
                loss = loss+.5*(value-target.flatten()[ids]).square().mean()-.01*entropy
                if self.options.deferred_metrics:
                    kval, finite = torch.stack((kl.detach(), torch.isfinite(loss).float())).tolist()
                else:
                    kval = float(kl.detach())
                    finite = None
                klmax = max(klmax, kval)
                if kval > .02:
                    break
                if not (finite if finite is not None else torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite PPO loss')
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), .5)
                self.optimizer.step()
                steps += 1
                clipping = ((delta.detach().exp()-1).abs() > .2).float().mean()
                if self.options.deferred_metrics:
                    metrics.add_(torch.stack((clipping, entropy.detach())))
                else:
                    clip_mean += float(clipping)
                    entropy_mean += float(entropy.detach())
            if kval > .02:
                break
        if self.options.deferred_metrics:
            clip_mean, entropy_mean = metrics.tolist()
        torch.cuda.synchronize()
        self.updates += 1
        return dict(collection_seconds=collected-start,
                    optimization_seconds=time.perf_counter()-collected,
                    optimizer_steps=steps, max_kl=klmax,
                    entropy=entropy_mean/max(steps, 1), clip_fraction=clip_mean/max(steps, 1))
