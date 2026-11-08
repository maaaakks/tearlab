"""Paired, interleaved confirmation. Run from the IsaacRL repository root."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time
import torch
sys.path.insert(0, str(Path.cwd()))
from research.gpu_ablation import factory, qualify, assert_pair, manifest, percentile
from research.native_runtime import source_digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-gpu', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path)
    p.add_argument('--arch', default='89')
    p.add_argument('--steps', type=int, default=128)
    p.add_argument('--rounds', type=int, default=7)
    p.add_argument('--verify-steps', type=int, default=64)
    p.add_argument('--atol', type=float, default=3e-4)
    p.add_argument('--rtol', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=29)
    p.add_argument('--lanes', nargs='+', type=int, default=[512, 8192])
    p.add_argument('--profile', action='store_true')
    a = p.parse_args()
    if not a.run_gpu:
        print('No GPU execution: pass --run-gpu explicitly')
        return
    torch.set_num_threads(1)
    names = ['graph_v1', 'fused_decision', 'fused_decision_warp']
    result = dict(status='running', gpu_executed=True, gpu=torch.cuda.get_device_name(),
                  sources=manifest(), native_sha256=source_digest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  seed=a.seed, steps=a.steps, rounds=a.rounds,
                  atol=a.atol, rtol=a.rtol, order='randomized per round, same actions per backend', cases=[])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    def save(): a.output.write_text(json.dumps(result, indent=2), encoding='utf8')
    save()
    for scenario in ['shooters', 'mixed']:
        for n in a.lanes:
            case = dict(scenario=scenario, lanes=n, status='qualifying', qualification={}, variants={})
            result['cases'].append(case); save()
            sims, envs = {}, {}
            for name in names:
                case['qualification'][name] = qualify(name, scenario, n, a.seed, 'natural', a)
                sims[name], envs[name] = factory(name, scenario, n, a.seed, 'natural', a.cache, arch=a.arch)
            actions = [(torch.arange(n, device='cuda') + t*17) % 45 for t in range(a.steps)]
            for name in names:
                for t in range(8): sims[name].step(actions[t % len(actions)])
                case['variants'][name] = dict(wall_samples_ms=[], cuda_samples_ms=[], done_counts=[])
            choice = random.Random(4091)
            errors = {}
            for round_id in range(a.rounds):
                order = names.copy();choice.shuffle(order)
                outputs, dones = {}, {}
                for name in order:
                    first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    first.record();last.record();torch.cuda.synchronize()
                    done = []
                    first.record(); start = time.perf_counter()
                    for action in actions:
                        out = sims[name].step(action)
                        done.append(out[2])
                    last.record();torch.cuda.synchronize()
                    wall = (time.perf_counter()-start)*1000/a.steps
                    case['variants'][name]['wall_samples_ms'].append(wall)
                    case['variants'][name]['cuda_samples_ms'].append(first.elapsed_time(last)/a.steps)
                    outputs[name] = out
                    dones[name] = torch.stack(done)
                    case['variants'][name]['done_counts'].append(int(dones[name].sum()))
                # Outside timing: every done bit, and full state at each round
                # boundary, across a much longer trajectory than the preflight.
                for name in names[1:]:
                    torch.testing.assert_close(dones['graph_v1'], dones[name], atol=0, rtol=0)
                    assert_pair(envs['graph_v1'], envs[name], outputs['graph_v1'], outputs[name], a.atol, a.rtol, errors)
                    if not torch.equal(envs['graph_v1'].rng.get_state(), envs[name].rng.get_state()):
                        raise AssertionError('RNG mismatch after interleaved round')
                case['completed_rounds'] = round_id+1;save()
            for name in names:
                v = case['variants'][name]
                v['wall_ms_median'] = statistics.median(v['wall_samples_ms'])
                v['wall_ms_p95'] = percentile(v['wall_samples_ms'], .95)
                v['decisions_s'] = n*1000/v['wall_ms_median']
                v['issued_ticks_s'] = 3*v['decisions_s']
                v['speedup_vs_graph_v1'] = statistics.median(case['variants']['graph_v1']['wall_samples_ms'])/v['wall_ms_median']
            case.update(status='passed', discrete_and_rng_exact=True, max_abs_by_field=errors,
                        long_trajectory_decisions_per_lane=a.steps*a.rounds+8)
            if a.profile:
                for name in names:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for action in actions[:5]: sims[name].step(action)
                        torch.cuda.synchronize()
                    table = prof.key_averages()
                    case['variants'][name]['cpu_profile'] = table.table(sort_by='self_cpu_time_total', row_limit=10)
                    case['variants'][name]['gpu_profile'] = table.table(sort_by='self_device_time_total', row_limit=16)
                    trace = a.output.parent/f'{scenario}_{n}_{name}.trace.json'
                    prof.export_chrome_trace(str(trace))
            print(scenario, n, {name:round(case['variants'][name]['wall_ms_median'],4) for name in names}, flush=True)
            save()
            del sims, envs, outputs, dones, actions
    result['status']='completed';save()


if __name__ == '__main__':
    main()
