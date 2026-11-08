"""Paired wall-clock benchmark; warm-up/capture excluded and reported separately.

    python -m research.benchmark_gpu_simulator --output path/to/results.json
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
from research.gpu_simulator import (GraphReadyNativeEnv, GraphReadyCombatSuiteEnv,
                                    CudaGraphSimulator, BACKEND_VERSION)
from research.rapid_physics import StrictNativeShooterEnv, StrictCombatSuiteEnv
from physics_v2 import NativePhysics, NativeShooterEnv


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lanes', nargs='+', type=int, default=[128, 512, 2048, 8192])
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.lanes + [args.steps, args.repeats]) < 1:
        parser.error('Positive counts required')
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    paths = [root/'research/gpu_simulator.py', root/'research/rapid_physics.py',
             *[root/'python'/name for name in ('physics_v2.py', 'shooter_lab.py', 'combat_suite.py', 'multi_combat_lab.py', 'alignment_lab.py')]]
    result = dict(backend=BACKEND_VERSION, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(), native=asdict(NativePhysics()),
                  units='lane decisions/s; three native ticks/decision', steps=args.steps,
                  repeats=args.repeats, outputs='owning buffers; obs/reward/done/all info included',
                  source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  cases=[])
    for mixed in (False, True):
        for n in args.lanes:
            seed = 481
            ids = [i % 13 for i in range(n)]
            b = StrictCombatSuiteEnv(ids, 'cuda', seed, NativePhysics()) if mixed else StrictNativeShooterEnv(n, 'cuda', seed, enemies=4)
            e = GraphReadyCombatSuiteEnv(ids, 'cuda', seed, NativePhysics()) if mixed else GraphReadyNativeEnv(n, 'cuda', seed, enemies=4)
            torch.cuda.synchronize()
            start = time.perf_counter()
            g = CudaGraphSimulator(e)
            torch.cuda.synchronize()
            prepare = time.perf_counter()-start
            actions = [((torch.arange(n, device='cuda') + i*17) % 45) for i in range(args.steps)]
            for k in range(8):
                b.step(actions[k % args.steps]); g.step(actions[k % args.steps])
            samples = {'eager': [], 'graph': []}
            for repeat in range(args.repeats):
                # Alternate execution order to limit ordering and thermal effects.
                order = [('eager', b), ('graph', g)]
                if repeat % 2: order.reverse()
                for name, sim in order:
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    for a in actions: sim.step(a)
                    torch.cuda.synchronize()
                    samples[name].append((time.perf_counter()-t)/args.steps)
            eager = statistics.median(samples['eager'])
            graph = statistics.median(samples['graph'])
            row = dict(scenario='mixed' if mixed else 'shooters', lanes=n,
                       capture_seconds=prepare, eager_ms=eager*1000, graph_ms=graph*1000,
                       eager_decisions_s=n/eager, graph_decisions_s=n/graph,
                       speedup=eager/graph, samples_seconds=samples)
            result['cases'].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(f"{row['scenario']:8s} {n:5d}: {eager*1000:7.3f} -> {graph*1000:7.3f} ms, x{eager/graph:.2f}, {n/graph:,.0f} decisions/s", flush=True)
            del g, e, b
    print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    run()
