"""Bounded, paired control experiment; fresh weights and simulator data only.

Run from the repository root with the Windows project venv. No Isaac process,
historical replay, pixel reconstruction, tactical controller or oracle search.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from research.direct_combat import ROOT, TRAIN_CELLS, observe
from research.rapid_model import RapidQ
from research.rapid_physics import StrictNativeShooterEnv, StrictCombatSuiteEnv
from physics_v2 import NativePhysics


class MicroEnv(StrictNativeShooterEnv):
    """One stationary, one-hit shooter, arbitrary spawn angle; diagnostic only."""

    def reset(self, mask):
        if self.keep_terminal:
            return
        super().reset(mask)
        self.hp[mask, 0] = 1


class Adapter:
    def __init__(self, suite, count, seed, device):
        if suite == 'micro':
            self.env = MicroEnv(count, device, seed, enemies=1, native=NativePhysics())
        else:
            self.env = StrictCombatSuiteEnv([TRAIN_CELLS[i % 5] for i in range(count)],
                                            device, seed, NativePhysics())
        self.env.p.max_steps = 100
        self.env.keep_terminal = True
        first = observe(self.env)
        self.history = torch.stack((first, first), 1)
        self.previous = torch.full((count,), 45, dtype=torch.long, device=device)
        self.episodes = torch.arange(count, device=device)
        self.reset_count = count

    @torch.no_grad()
    def step(self, actions):
        current = self.history[:, -1]
        _, _, done, info = self.env.step(actions)
        last = observe(self.env)
        final = torch.stack((current, last), 1)
        reward = info['success'].float() - info['contact'].float() - .002
        terminated = done & ~info['timeout']
        # Measurements are observable consequences, never additional rewards.
        hp_before = current[:, 4:20].reshape(-1, 4, 4)[:, :, 2].sum(-1)
        hp_after = last[:, 4:20].reshape(-1, 4, 4)[:, :, 2].sum(-1)
        measurements = torch.cat((last[:, :2] - current[:, :2],
                                  ((hp_after - hp_before) / 4)[:, None],
                                  info['contact'].float()[:, None]), -1)
        self.env.keep_terminal = False
        self.env.reset(done)
        self.env.keep_terminal = True
        first = observe(self.env)
        self.history = torch.where(done[:, None, None], torch.stack((first, first), 1), final)
        self.previous = torch.where(done, 45, actions)
        self.episodes = self.episodes + done * self.env.n
        self.reset_count += int(done.sum())
        return final, reward, terminated, done, measurements, info


class SequenceReplay:
    """Immutable transitions, lane-strided sequences; terminal tails stay valid."""

    def __init__(self, capacity, lanes, device):
        self.capacity, self.lanes, self.used = capacity, lanes, 0
        self.obs = torch.empty(capacity, 2, 140, device=device)
        self.next_obs = torch.empty_like(self.obs)
        self.previous = torch.empty(capacity, dtype=torch.long, device=device)
        self.actions = torch.empty_like(self.previous)
        self.episode = torch.empty_like(self.previous)
        self.reward = torch.empty(capacity, device=device)
        self.terminal = torch.empty(capacity, dtype=torch.bool, device=device)
        self.done = torch.empty_like(self.terminal)
        self.events = torch.empty_like(self.terminal)
        self.measurements = torch.empty(capacity, 4, device=device)
        self.head_mask = torch.empty(capacity, 3, dtype=torch.bool, device=device)

    def append(self, obs, previous, actions, episode, final, reward, terminal, done,
               measurements, head_mask):
        if len(actions) != self.lanes or self.used + self.lanes > self.capacity:
            raise ValueError('Invalid lane count or exhausted simulator budget')
        target = slice(self.used, self.used + self.lanes)
        for key, value in dict(obs=obs, previous=previous, actions=actions, episode=episode,
                               next_obs=final, reward=reward, terminal=terminal, done=done,
                               measurements=measurements, head_mask=head_mask).items():
            getattr(self, key)[target] = value
        self.events[target] = (measurements[:, 2] != 0) | (measurements[:, 3] > 0) | (reward > 0)
        self.used += self.lanes

    def sequences(self, indices, horizon=8):
        requested = indices[:, None] + torch.arange(horizon, device=indices.device)[None] * self.lanes
        inside = requested < self.used
        safe = requested.clamp_max(self.used - 1)
        same_episode = self.episode[safe] == self.episode[indices, None]
        # Explicit done mask protects the contract even for malformed episode IDs.
        before_done = torch.cat((torch.ones_like(inside[:, :1]),
                                 (~self.done[safe[:, :-1]]).cumprod(1).bool()), 1)
        return safe, inside & same_episode & before_done

    def sample(self, batch, rng):
        device = self.obs.device
        indices = torch.randint(self.used, (batch,), device=device, generator=rng)
        event_indices = self.events[:self.used].nonzero().flatten()
        if len(event_indices):
            special = event_indices[torch.randint(len(event_indices), (batch,), device=device, generator=rng)]
            indices = torch.where(torch.rand(batch, device=device, generator=rng) < .5, special, indices)
            probability = .5 / self.used + .5 * self.events[indices].float() / len(event_indices)
            weights = 1 / (self.used * probability)
        else:
            weights = torch.ones(batch, device=device)
        return indices, weights

    def save(self, path):
        tensors = {key: value[:self.used].cpu() for key, value in vars(self).items()
                   if isinstance(value, torch.Tensor)}
        torch.save(dict(lanes=self.lanes, used=self.used, **tensors), path)


def returns_and_endpoint(replay, indices, horizon=4, gamma=.99):
    sequence, valid = replay.sequences(indices, horizon)
    discounts = gamma ** torch.arange(horizon, device=indices.device)
    rewards = (replay.reward[sequence] * valid * discounts).sum(1)
    lengths = valid.sum(1)
    last = sequence.gather(1, (lengths - 1)[:, None]).squeeze(1)
    return rewards, gamma ** lengths, last


def update(model, target, optimizer, replay, rng, batch):
    indices, weights = replay.sample(batch, rng)
    returns, discounts, last = returns_and_endpoint(replay, indices)
    with torch.no_grad():
        final = replay.next_obs[last]
        # The previous command at the bootstrap state is the last applied request,
        # including a time-limit endpoint; BOS is used only after an actual reset.
        final_previous = replay.actions[last]
        choices = model(final, final_previous).argmax(-1, keepdim=True)
        following = target(final, final_previous).gather(2, choices).squeeze(2)
        desired = returns[:, None] + discounts[:, None] * (~replay.terminal[last, None]) * following
    values = model(replay.obs[indices], replay.previous[indices])
    current = values.gather(2, replay.actions[indices, None, None].expand(-1, values.shape[1], 1)).squeeze(2)
    masks = replay.head_mask[indices, :values.shape[1]].float()
    if values.shape[1] == 1:
        masks = torch.ones_like(masks)
    per_sample = (F.smooth_l1_loss(current, desired, reduction='none') * masks).sum(1) / masks.sum(1).clamp_min(1)
    td_loss = (per_sample * weights).mean()
    loss = td_loss
    aux_loss = torch.zeros((), device=values.device)
    reads = int(replay.sequences(indices, 4)[1].sum())
    if model.auxiliary:
        sequence, valid = replay.sequences(indices, 8)
        prediction = model.auxiliary_predictions(replay.obs[indices], replay.previous[indices],
                                                  replay.actions[sequence])
        targets = replay.measurements[sequence]
        regression = F.smooth_l1_loss(prediction[:, :, :3], targets[:, :, :3], reduction='none').mean(2)
        contact = F.binary_cross_entropy_with_logits(prediction[:, :, 3], targets[:, :, 3], reduction='none')
        aux_sample = ((regression + contact) * valid).sum(1) / valid.sum(1)
        aux_loss = (aux_sample * weights).mean()
        loss = loss + .2 * aux_loss
        reads += int(valid.sum())
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
    optimizer.step()
    return float(td_loss.detach()), float(aux_loss.detach()), reads


@torch.no_grad()
def evaluate(model, suite, count, seed, device, policy='learned', repeat=4):
    adapter = Adapter(suite, count, seed, device)
    rng = torch.Generator(device=device).manual_seed(seed + 333)
    active = torch.ones(count, dtype=torch.bool, device=device)
    wins = torch.zeros_like(active)
    contacts = torch.zeros_like(active)
    hits = torch.zeros(count, device=device)
    duration = torch.zeros_like(hits)
    action_counts = torch.zeros(45, dtype=torch.long, device=device)
    issued = 0
    trace = []
    for step in range(100):
        if step % repeat == 0:
            if policy == 'random':
                actions = torch.randint(45, (count,), device=device, generator=rng)
            elif policy.startswith('constant'):
                actions = torch.full((count,), int(policy.removeprefix('constant')), device=device)
            else:
                actions = model(adapter.history, adapter.previous).mean(1).argmax(1)
        action_counts += torch.bincount(actions[active], minlength=45)
        _, reward, terminated, done, _, info = adapter.step(actions)
        issued += count
        trace.append(dict(step=step, active=active.cpu().tolist(), actions=actions.cpu().tolist(),
                          reward=reward.cpu().tolist(), terminated=terminated.cpu().tolist(), done=done.cpu().tolist()))
        duration += active
        hits += active * info['damage']
        wins |= active & done & info['success']
        contacts |= active & info['contact']
        active &= ~done
        if not active.any():
            break
    return dict(policy=policy, episodes=count, wins=int(wins.sum()), clear_rate=float(wins.float().mean()),
                contacts=int(contacts.sum()), mean_hits=float(hits.mean()), mean_decisions=float(duration.mean()),
                action_counts=action_counts.cpu().tolist(), simulator_decisions=issued, simulator_frames=issued * 3,
                scored_decisions=int(duration.sum()), resets=adapter.reset_count, seed=seed, trace=trace)


def sources_snapshot(output):
    folder = output / 'sources'
    hashes = {}
    for relative in ('research/rapid_control.py', 'research/rapid_model.py', 'research/rapid_physics.py',
                     'research/direct_combat.py', 'python/physics_v2.py', 'python/combat_suite.py',
                     'python/shooter_lab.py', 'python/multi_combat_lab.py', 'python/alignment_lab.py'):
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (ROOT / relative).read_bytes()
        path.write_bytes(data)
        hashes[relative] = hashlib.sha256(data).hexdigest()
    return hashes


def train(args, output, suite, variant, seed):
    started = time.perf_counter()
    torch.manual_seed(seed)
    device = torch.device(args.device)
    rng = torch.Generator(device=device).manual_seed(10000 + seed)
    replay_rng = torch.Generator(device=device).manual_seed(20000 + seed)
    head_rng = torch.Generator(device=device).manual_seed(30000 + seed)
    heads = 1 if variant == 'q' else 3
    model = RapidQ(heads=heads, auxiliary=variant == 'predictive').to(device)
    target = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    replay = SequenceReplay(args.decisions, args.envs, device)
    adapter = Adapter(suite, args.envs, 100000 + seed, device)
    selected_heads = torch.randint(heads, (args.envs,), device=device, generator=head_rng)
    remaining = torch.zeros(args.envs, dtype=torch.long, device=device)
    actions = torch.zeros_like(remaining)
    counters = dict(train_decisions=0, train_frames=0, replay_reads=0, updates=0,
                    train_clears=0, train_contacts=0, train_hits=0, first_clear_at=None,
                    evaluation_decisions=0, imagined_transitions=0)
    losses = []
    initial = evaluate(model, suite, args.eval_episodes, 900000 + seed, device, repeat=args.repeat)
    counters['evaluation_decisions'] += initial['simulator_decisions']
    timed_out = False
    for _ in range(args.decisions // args.envs):
        if time.perf_counter() - started >= args.max_seconds:
            timed_out = True
            break
        obs, previous, episodes = adapter.history.clone(), adapter.previous.clone(), adapter.episodes.clone()
        with torch.no_grad():
            choose = remaining == 0
            values = model(obs, previous)
            greedy = values[torch.arange(args.envs, device=device), selected_heads].argmax(1)
            epsilon = max(.1, 1 - .9 * replay.used / (.8 * args.decisions))
            random_actions = torch.randint(45, (args.envs,), device=device, generator=rng)
            proposed = torch.where(torch.rand(args.envs, device=device, generator=rng) < epsilon, random_actions, greedy)
            actions = torch.where(choose, proposed, actions)
            remaining = torch.where(choose, args.repeat, remaining)
            final, reward, terminal, done, measurements, info = adapter.step(actions)
            mask = torch.rand(args.envs, 3, device=device, generator=head_rng) < .8
            mask[:, 0] |= ~mask.any(1)
        replay.append(obs, previous, actions, episodes, final, reward, terminal, done, measurements, mask)
        remaining = torch.where(done, 0, remaining - 1)
        new_heads = torch.randint(heads, (args.envs,), device=device, generator=head_rng)
        selected_heads = torch.where(done, new_heads, selected_heads)
        counters['train_decisions'] = replay.used
        counters['train_frames'] = replay.used * 3
        clears = int(info['success'].sum())
        counters['train_clears'] += clears
        counters['train_contacts'] += int(info['contact'].sum())
        counters['train_hits'] += int(info['damage'].sum())
        if clears and counters['first_clear_at'] is None:
            counters['first_clear_at'] = replay.used
        if replay.used >= 512:
            td, aux, reads = update(model, target, optimizer, replay, replay_rng, args.batch)
            losses.append(dict(decisions=replay.used, td=td, auxiliary=aux))
            counters['updates'] += 1
            counters['replay_reads'] += reads
            if counters['updates'] % 100 == 0:
                target.load_state_dict(model.state_dict())
        if replay.used % 2000 == 0:
            print(json.dumps(dict(suite=suite, variant=variant, seed=seed, decisions=replay.used,
                                  train_clears=counters['train_clears'], elapsed=round(time.perf_counter()-started, 2))), flush=True)
    final = evaluate(model, suite, args.eval_episodes, 900000 + seed, device, repeat=args.repeat)
    counters['evaluation_decisions'] += final['simulator_decisions']
    counters['evaluation_frames'] = counters['evaluation_decisions'] * 3
    counters['train_resets'] = adapter.reset_count
    folder = output / f'{suite}_{variant}_seed{seed}'
    folder.mkdir()
    replay.save(folder / 'replay.pt')
    torch.save(dict(model=model.state_dict(), target=target.state_dict(), optimizer=optimizer.state_dict(),
                    heads=heads, auxiliary=model.auxiliary), folder / 'final.pt')
    (folder / 'losses.json').write_text(json.dumps(losses), encoding='utf-8')
    report = dict(suite=suite, variant=variant, seed=seed, initial=initial, final=final, counters=counters,
                  wall_seconds=time.perf_counter()-started, parameters=sum(p.numel() for p in model.parameters()),
                  stop_reason='wall_budget' if timed_out else 'decision_budget',
                  complete=not timed_out, weights='fresh random initialization',
                  source_sha256=json.loads((output / 'manifest.json').read_text())['sources'])
    (folder / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(finished=str(folder), seconds=round(report['wall_seconds'], 2),
                          wins=final['wins'], initial_wins=initial['wins'], train_clears=counters['train_clears'])), flush=True)
    return report


def run(args):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = args.output / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(protocol='RAPID-1 development pilot', config={**vars(args), 'output': str(args.output)},
                    sources=sources_snapshot(output), native_physics=asdict(NativePhysics()),
                    observations='two structured frames, instrumented present HP, previous request or BOS; no hidden timers/velocity',
                    reward='clear - contact - .002',
                    train='fresh, no demos, historical trajectories or pretrained weights',
                    scenarios='micro: 1 stationary one-hit shooter; mixed: five D1 families; both capped at 100 decisions',
                    difference_from_D1='strict terminal tick freeze; previous action; 4-step TD; repeat4; corrected event replay',
                    variants='q: 1 head; ensemble: 3 heads; predictive: same 3 heads plus factual 8-step consequences',
                    comparison='same decisions, update count, replay mixture, reward, scenario seeds; not equal FLOPs/parameters',
                    eval='paired fresh seeds; development validation only, not compositional heldout test',
                    clocks='train wall cap per run; initial validation included; final validation and save may exceed cap',
                    provenance='simulator construction uses historical physics calibration; no Isaac interaction',
                    limits='2-frame memory; uncorrected off-policy n-step bias; ensemble disagreement not calibrated; no imagination')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(started=str(output))), flush=True)
    reports, controls = [], []
    for suite in args.suites:
        for seed in args.seeds:
            for policy in ('random', 'constant22', 'constant23'):
                result = evaluate(None, suite, args.eval_episodes, 900000 + seed, args.device, policy, args.repeat)
                controls.append(dict(suite=suite, training_seed=seed, **result))
            for variant in args.variants:
                reports.append(train(args, output, suite, variant, seed))
                compact = [{k: r[k] for k in ('suite','variant','seed','initial','final','counters','wall_seconds','complete')} for r in reports]
                summary = dict(reports=compact, controls=controls,
                               total_train_decisions=sum(r['counters']['train_decisions'] for r in reports),
                               total_evaluation_decisions=sum(r['counters']['evaluation_decisions'] for r in reports)
                               + sum(c['simulator_decisions'] for c in controls),
                               imagined_transitions=0, live_isaac_decisions=0)
                (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(dict(summary=str(output / 'summary.json'))), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suites', nargs='+', choices=('micro', 'mixed'), default=['micro'])
    parser.add_argument('--variants', nargs='+', choices=('q', 'ensemble', 'predictive'), default=['q', 'ensemble', 'predictive'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--decisions', type=int, default=10000)
    parser.add_argument('--envs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--eval-episodes', type=int, default=40)
    parser.add_argument('--repeat', type=int, default=4)
    parser.add_argument('--max-seconds', type=float, default=90)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/research/rapid_control')
    args = parser.parse_args()
    if (args.envs <= 0 or args.envs % 5 or args.decisions < 512 or args.decisions % args.envs
            or args.eval_episodes <= 0 or args.eval_episodes % 5 or args.batch <= 0
            or args.repeat <= 0 or args.max_seconds <= 0):
        parser.error('Require positive settings, envs/eval-episodes divisible by 5, decisions >=512 divisible by envs')
    run(args)


if __name__ == '__main__':
    main()
