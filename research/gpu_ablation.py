"""Offline experiment plan and explicit, fidelity-gated GPU ablations.

Default invocation only emits a plan. No CUDA device is queried until --run-gpu.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time
import traceback

import torch
from research.gpu_simulator import CudaGraphSimulator, GraphReadyNativeEnv, GraphReadyCombatSuiteEnv
from research.staged_simulator import StagedNativeEnv, StagedCombatSuiteEnv, Optimizations, ALL, STATE
from research.native_runtime import source_digest, build_native
from research.rapid_physics import StrictNativeShooterEnv, StrictCombatSuiteEnv
from physics_v2 import NativePhysics

UNFUSED = replace(ALL, tick_fusion=False, decision_fusion=False)
STAGES = ('movement', 'attacks', 'projectiles', 'terminal_mask', 'sparse_reset', 'direct_observation')
VARIANTS = {'graph_v1': None, 'staged_control': Optimizations(),
            **{name: Optimizations(**{name: True}) for name in STAGES},
            'unfused_all': UNFUSED,
            **{'without_' + name: replace(UNFUSED, **{name: False}) for name in STAGES},
            'fused_tick': replace(ALL, decision_fusion=False),
            'fused_decision': ALL, 'fused_decision_warp': replace(ALL, mapping='warp')}


def manifest():
    here = Path(__file__).parent
    files = [here/'gpu_ablation.py', here/'native_runtime.py', here/'staged_simulator.py',
             here/'gpu_simulator.py', here/'rapid_physics.py',
             *[(here.parent/'python'/p) for p in ('physics_v2.py', 'shooter_lab.py', 'combat_suite.py', 'multi_combat_lab.py', 'alignment_lab.py')]]
    return {str(p.relative_to(here.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def plan(args):
    return dict(schema=1, status='pending_gpu_validation', gpu_executed=False,
                utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                torch=torch.__version__, native_source_sha256=source_digest(), sources=manifest(),
                cuda_arch=getattr(args, 'arch', '89'),
                cuda_compiler_flags=['--fmad=false','--ftz=false','--prec-div=true','--prec-sqrt=true'],
                variants={k: asdict(v) if v is not None else 'existing graph backend' for k, v in VARIANTS.items()},
                selected=args.only or list(VARIANTS), scenarios=args.scenarios, reset_modes=args.reset_modes,
                lanes=args.lanes, seeds=args.seeds, repeats=args.repeats, decisions_per_repeat=args.steps,
                native_ticks_per_decision=3, warmup=args.warmup,
                fidelity=dict(steps=args.verify_steps, atol=args.atol, rtol=args.rtol,
                              discrete='exact', rng='exact', scope='all state tensors, observations, reward, done and every info field'),
                comparisons=dict(single_add='stage vs staged_control',
                                 single_remove='unfused_all vs without_stage',
                                 tick_fusion='fused_tick vs unfused_all',
                                 decision_fusion='fused_decision vs fused_tick',
                                 mapping='fused_decision_warp vs fused_decision',
                                 end_to_end='each candidate vs graph_v1'),
                timing='wall median/p95 + CUDA-event median/p95; capture, fidelity and profiler excluded',
                output_ownership='owning outputs for every backend; no step_view comparisons',
                cases=[])


def factory(name, scenario, n, seed, reset_mode, cache, graph=True, device='cuda', arch='89'):
    mixed = scenario == 'mixed'
    ids = [i % 13 for i in range(n)]
    options = VARIANTS[name]
    cls = ((GraphReadyCombatSuiteEnv if mixed else GraphReadyNativeEnv) if options is None
           else (StagedCombatSuiteEnv if mixed else StagedNativeEnv))
    kw = {} if options is None else dict(optimizations=options, cache=cache, native_arch=arch)
    e = cls(ids, device, seed, NativePhysics(), **kw) if mixed else cls(n, device, seed, enemies=4, **kw)
    if reset_mode == 'frequent':
        e.p.max_steps = 7
    elif reset_mode == 'none':
        e.keep_terminal = True
    if options is not None:
        e.prepare()
    return (CudaGraphSimulator(e) if graph else e), e


def assert_pair(reference, candidate, x, y, atol, rtol, errors):
    pairs = [(f'output.{i}', x[i], y[i]) for i in range(3)]
    if x[3].keys() != y[3].keys():
        raise AssertionError('Info keys differ')
    pairs += [(f'info.{k}', x[3][k], y[3][k]) for k in x[3]]
    pairs += [(f'state.{k}', v, getattr(candidate, k)) for k, v in vars(reference).items() if isinstance(v, torch.Tensor)]
    for label, a, b in pairs:
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(f'{label}: shape or dtype differs')
        numeric = a.is_floating_point()
        if numeric:
            if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
                raise AssertionError(f'{label}: nonfinite state')
            errors[label] = max(errors.get(label, 0.), (a-b).abs().max().item() if a.numel() else 0.)
        torch.testing.assert_close(a, b, atol=atol if numeric else 0, rtol=rtol if numeric else 0,
                                   msg=lambda s: f'{label}: {s}')


def qualify(name, scenario, n, seed, reset_mode, args):
    mixed = scenario == 'mixed'
    reference = (StrictCombatSuiteEnv([i % 13 for i in range(n)], 'cuda', seed, NativePhysics())
                 if mixed else StrictNativeShooterEnv(n, 'cuda', seed, enemies=4))
    if reset_mode == 'frequent': reference.p.max_steps = 7
    if reset_mode == 'none': reference.keep_terminal = True
    sim, e = factory(name, scenario, n, seed, reset_mode, args.cache, arch=getattr(args, 'arch', '89'))
    errors = {}
    for tick in range(args.verify_steps):
        actions = (torch.arange(n, device='cuda') + tick*17) % 45
        x = reference.step(actions)
        y = sim.step(actions)
        assert_pair(reference, e, x, y, args.atol, args.rtol, errors)
    if not torch.equal(reference.rng.get_state(), e.rng.get_state()):
        raise AssertionError('RNG state differs')
    # Output lifetime and manually synchronized resets are also exercised by tests.
    return dict(status='passed', decisions=args.verify_steps*n, max_abs_by_field=errors,
                max_abs=max(errors.values(), default=0), discrete_exact=True, rng_exact=True)


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered)-1, int((len(ordered)-1)*q+.5))]


def timings(sim, actions, args):
    for i in range(args.warmup):
        sim.step(actions[i % len(actions)])
    wall, gpu, resets = [], [], []
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    # Create CUDA events before timed iterations.
    for event in events: event.record()
    torch.cuda.synchronize()
    for _ in range(args.repeats):
        dones = []
        torch.cuda.synchronize()
        events[0].record()
        start = time.perf_counter()
        for action in actions:
            out = sim.step(action)
            dones.append(out[2])
        events[1].record()
        torch.cuda.synchronize()
        wall.append((time.perf_counter()-start)/len(actions))
        gpu.append(events[0].elapsed_time(events[1])/1000/len(actions))
        resets.append(int(torch.stack(dones).sum()))
    med = statistics.median(wall)
    return dict(wall_ms_median=med*1000, wall_ms_p95=percentile(wall,.95)*1000,
                cuda_elapsed_ms_median=statistics.median(gpu)*1000,
                cuda_elapsed_ms_p95=percentile(gpu,.95)*1000,
                lane_decisions_s=sim.n/med, issued_lane_ticks_s=3*sim.n/med,
                done_counts=resets, done_fraction=sum(resets)/(len(resets)*len(actions)*sim.n),
                wall_samples_s=wall, cuda_elapsed_samples_s=gpu)


def run(args):
    result = plan(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, indent=2), encoding='utf8')
    save()
    if not args.run_gpu:
        print(f'Offline plan only; no CUDA initialized. {args.output}')
        return
    torch.set_num_threads(1)
    build_native('cuda', args.cache, args.arch)
    torch.cuda.set_device(args.device)
    result.update(status='running', gpu_executed=True, gpu=torch.cuda.get_device_name(args.device))
    save()
    selected = args.only or list(VARIANTS)
    chooser = random.Random(315)
    failed = False
    for scenario in args.scenarios:
        for reset_mode in args.reset_modes:
            for n in args.lanes:
                for seed in args.seeds:
                    order = selected.copy()
                    chooser.shuffle(order)
                    for name in order:
                        row = dict(variant=name, scenario=scenario, reset_mode=reset_mode, lanes=n, seed=seed)
                        result['cases'].append(row)
                        try:
                            # Qualify the actual batch shape, not only a tiny proxy.
                            row['fidelity'] = qualify(name, scenario, n, seed, reset_mode, args)
                            torch.cuda.synchronize()
                            before = torch.cuda.memory_allocated()
                            torch.cuda.reset_peak_memory_stats()
                            start = time.perf_counter()
                            sim, e = factory(name, scenario, n, seed, reset_mode, args.cache, arch=args.arch)
                            torch.cuda.synchronize()
                            row['capture_seconds'] = time.perf_counter()-start
                            if getattr(e, '_ops', None) is not None:
                                row['native_binary_sha256'] = hashlib.sha256(e._ops.binary.read_bytes()).hexdigest()
                            row['persistent_extra_bytes'] = max(0, torch.cuda.memory_allocated()-before)
                            row['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                            row['graph_private_pool_bytes'] = None  # Not inferred from allocator totals.
                            actions = [(torch.arange(n, device='cuda') + t*17) % 45 for t in range(args.steps)]
                            row['timing'] = timings(sim, actions, args)
                            if args.profile:
                                trace = args.output.parent / f'{scenario}_{reset_mode}_{n}_{seed}_{name}.trace.json'
                                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                                    for action in actions[:3]: sim.step(action)
                                    torch.cuda.synchronize()
                                prof.export_chrome_trace(str(trace))
                                row['trace'] = str(trace)
                                row['kernels'] = prof.key_averages().table(sort_by='self_device_time_total', row_limit=25)
                            row['status'] = 'qualified_and_measured'
                            print(f"{scenario} {reset_mode} n={n} {name}: {row['timing']['wall_ms_median']:.4f} ms", flush=True)
                            del sim, e, actions
                        except Exception:
                            row['status'] = 'failed_not_promoted'
                            row['error'] = traceback.format_exc()
                            failed = True
                            print(f'{name}: FAILED (not timed/promoted)', flush=True)
                        save()
    result['status'] = 'completed_with_failures' if failed else 'completed'
    result['comparisons'] = compare_rows(result['cases'])
    save()
    if failed:
        raise SystemExit(1)


def compare_rows(rows):
    output = []
    groups = {}
    for row in rows:
        if row.get('status') != 'qualified_and_measured': continue
        key = (row['scenario'], row['reset_mode'], row['lanes'], row['seed'])
        groups.setdefault(key, {})[row['variant']] = row
    pairs = [('graph_v1', k, 'end_to_end') for k in VARIANTS if k != 'graph_v1']
    pairs += [('staged_control', k, 'single_add') for k in STAGES]
    pairs += [('without_'+k, 'unfused_all', 'single_remove') for k in STAGES]
    pairs += [('unfused_all', 'fused_tick', 'tick_fusion'), ('fused_tick', 'fused_decision', 'decision_fusion'),
              ('fused_decision', 'fused_decision_warp', 'mapping')]
    for key, group in groups.items():
        for a, b, kind in pairs:
            if a in group and b in group:
                output.append(dict(case=key, kind=kind, baseline=a, candidate=b,
                                   speedup=group[a]['timing']['wall_ms_median']/group[b]['timing']['wall_ms_median']))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-gpu', action='store_true', help='Explicitly authorize GPU qualification and timing')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path)
    p.add_argument('--arch', default='89')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--only', nargs='+', choices=list(VARIANTS))
    p.add_argument('--lanes', nargs='+', type=int, default=[512, 8192])
    p.add_argument('--seeds', nargs='+', type=int, default=[29, 481])
    p.add_argument('--scenarios', nargs='+', choices=['shooters','mixed'], default=['shooters','mixed'])
    p.add_argument('--reset-modes', nargs='+', choices=['natural','frequent','none'], default=['natural','frequent'])
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--warmup', type=int, default=8)
    p.add_argument('--verify-steps', type=int, default=64)
    p.add_argument('--atol', type=float, default=3e-4)
    p.add_argument('--rtol', type=float, default=1e-5)
    p.add_argument('--profile', action='store_true')
    args = p.parse_args()
    if min(args.lanes+[args.steps,args.repeats,args.warmup,args.verify_steps]) < 1:
        p.error('Counts must be positive')
    if args.atol < 0 or args.rtol < 0: p.error('Nonnegative tolerances required')
    run(args)


if __name__ == '__main__':
    main()
