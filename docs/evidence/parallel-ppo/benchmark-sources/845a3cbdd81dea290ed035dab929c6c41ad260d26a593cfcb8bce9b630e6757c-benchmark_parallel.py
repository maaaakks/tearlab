"""Paired, interleaved full-update ablation; disposable policies, no live game."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=(Path(__file__).resolve().parents[1] if Path(__file__).parent.name=='research'
      else Path(r'C:\Users\maxbr\Desktop\game\IsaacRL\binding-of-ai'))
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
import torch
from research import pilot_10m as p
from research.staged_simulator import STATE
from research.gpu_simulator import GraphReadyCombatSuiteEnv, CudaGraphSimulator
try:
    from research.parallel_ppo import ParallelArm, Options
except ModuleNotFoundError:
    from parallel_ppo import ParallelArm, Options

VARIANTS = {
    'tensor_eager': None,
    'previous_graph': None,
    'native_reference': None,
    'control': Options(),
    'rollout_graph': Options(rollout_graph=True),
    'network_graph': Options(network_graph=True),
    'fused_adam': Options(fused_adam=True),
    'deferred_metrics': Options(deferred_metrics=True),
    'all': Options(True, True, True, True),
    'recommended': Options(True, True, True, False),
    'graphs_only': Options(True, True, False, False),
}

class PreviousGraphEnv(p.AutoResetMixin, GraphReadyCombatSuiteEnv):
    pass

def differences(a,b):
    discrete={}
    floats={}
    tensors={**{f'batch/{k}':v for k,v in a.batch.items()},
             'obs':a.obs,'previous':a.previous,'totals':a.totals,'action_counts':a.action_counts,
             **{f'env/{k}':getattr(a.env,k) for k in STATE}}
    other={**{f'batch/{k}':v for k,v in b.batch.items()},
           'obs':b.obs,'previous':b.previous,'totals':b.totals,'action_counts':b.action_counts,
           **{f'env/{k}':getattr(b.env,k) for k in STATE}}
    for key,x in tensors.items():
        y=other[key]
        if x.is_floating_point():
            error=float((x-y).abs().max()) if x.numel() else 0.
            floats[key]=error
            torch.testing.assert_close(x,y,atol=3e-4,rtol=1e-5,msg=lambda m:key+': '+m)
        else:
            discrete[key]=torch.equal(x,y)
            if not discrete[key]:raise AssertionError('Discrete mismatch: '+key)
    for key in ['action_rng','shuffle_rng']:
        if not torch.equal(getattr(a,key).get_state(),getattr(b,key).get_state()):
            raise AssertionError('RNG mismatch: '+key)
    if not torch.equal(a.env.rng.get_state(),b.env.rng.get_state()):
        raise AssertionError('Environment RNG mismatch')
    param_error=max(float((x-y).detach().abs().max()) for x,y in zip(a.policy.parameters(),b.policy.parameters()))
    for x,y in zip(a.policy.parameters(),b.policy.parameters()):
        torch.testing.assert_close(x,y,atol=2e-5,rtol=1e-4)
    return dict(max_tensor_abs=max(floats.values()),max_parameter_abs=param_error,
                discrete_exact=all(discrete.values()),rng_exact=True)

def make(initial,native,cache,n,variant):
    kwargs=dict(initial=initial,scenes=p.balanced_training(n),seed=104729,
                native=native,device='cuda',cache=cache,dense=True)
    if variant=='tensor_eager':return p.Arm(**kwargs,reference=True)
    if variant=='previous_graph':
        original=p.ReferenceTrainingEnv
        try:
            p.ReferenceTrainingEnv=PreviousGraphEnv
            arm=p.Arm(**kwargs,reference=True)
        finally:
            p.ReferenceTrainingEnv=original
        arm.sim=CudaGraphSimulator(arm.env)
        return arm
    return p.Arm(**kwargs) if variant=='native_reference' else ParallelArm(**kwargs,options=VARIANTS[variant])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-gpu',action='store_true')
    ap.add_argument('--variants',nargs='+',default=list(VARIANTS))
    ap.add_argument('--lanes',nargs='+',type=int,default=[1024])
    ap.add_argument('--rounds',type=int,default=7)
    ap.add_argument('--qualification-only',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if not args.run_gpu:return
    torch.set_num_threads(1)
    torch.manual_seed(104729)
    torch.backends.cuda.matmul.allow_tf32=False
    native=p.NativePhysics.load(ROOT/'data/physics_v2/calibration_r5.json')
    cache=args.output.parent/'cache'
    initial=p.PilotPolicy().cuda()
    result=dict(complete=False,device=torch.cuda.get_device_name(),torch=torch.__version__,
                rollout=32,epochs=2,minibatch=4096,training_seed=104729,
                disposable=True,live_game=False,qualification={},measurements={},
                sources={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                         for name in ('parallel_ppo.py','benchmark_parallel.py')})
    def write():args.output.write_text(json.dumps(result,indent=2))
    write()
    for n in args.lanes:
        for name in args.variants:
            print('QUALIFY',n,name,flush=True)
            start=time.perf_counter()
            reference=make(initial,native,cache,n,'native_reference')
            candidate=make(initial,native,cache,n,name)
            # Real timeout/bootstrap path as well as contacts and autoresets.
            reference.env.steps.fill_(398)
            candidate.env.steps.fill_(398)
            reference.update();candidate.update()
            gate=differences(reference,candidate)
            gate['seconds']=time.perf_counter()-start
            gate['timeouts']=int(candidate.batch['timeout'].sum())
            gate['done']=int(candidate.batch['done'].sum())
            result['qualification'][f'{n}/{name}']=gate
            write()
            print('PASS',gate,flush=True)
            del reference,candidate
            gc.collect();torch.cuda.empty_cache()
        if args.qualification_only:continue
        arms={}
        samples={}
        torch.cuda.reset_peak_memory_stats()
        for name in args.variants:
            allocated_before=torch.cuda.memory_allocated()
            start=time.perf_counter()
            arms[name]=make(initial,native,cache,n,name)
            setup=time.perf_counter()-start
            arms[name].update()
            samples[name]=dict(setup_seconds=setup,samples=[],
                               persistent_bytes=torch.cuda.memory_allocated()-allocated_before)
        # Rotate variant order each round, so late-clock/heat bias is shared.
        for repetition in range(args.rounds):
            order=args.variants[repetition%len(arms):]+args.variants[:repetition%len(arms)]
            for name in order:
                sample=arms[name].update()
                samples[name]['samples'].append(sample)
            print('ROUND',n,repetition+1,flush=True)
        for name,row in samples.items():
            seconds=[s['collection_seconds']+s['optimization_seconds'] for s in row['samples']]
            row.update(median_update_seconds=statistics.median(seconds),
                       median_collection_seconds=statistics.median(s['collection_seconds'] for s in row['samples']),
                       median_optimization_seconds=statistics.median(s['optimization_seconds'] for s in row['samples']),
                       decisions_s=n*32/statistics.median(seconds))
        result['measurements'][str(n)]=samples
        result.setdefault('peak_memory_bytes',{})[str(n)]=torch.cuda.max_memory_allocated()
        write()
        print('TIMINGS',n,{k:round(v['median_update_seconds'],4) for k,v in samples.items()},flush=True)
        del arms
        gc.collect();torch.cuda.empty_cache()
    result['complete']=True
    write()

if __name__=='__main__':main()
