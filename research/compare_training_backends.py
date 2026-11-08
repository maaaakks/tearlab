"""Disposable full-update comparison: eager, previous graph, native graph.

Run from the IsaacRL root; no saved agent is loaded or modified.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import torch
sys.path.insert(0,str(Path.cwd()))
from research import pilot_10m as p
from research.gpu_simulator import GraphReadyCombatSuiteEnv,CudaGraphSimulator

class PreviousGraphEnv(p.AutoResetMixin,GraphReadyCombatSuiteEnv):pass

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-gpu',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cache',type=Path)
    args=ap.parse_args()
    if not args.run_gpu:return
    torch.set_num_threads(1);torch.manual_seed(104729)
    torch.backends.cuda.matmul.allow_tf32=False
    native=p.NativePhysics.load(p.ROOT/'data/physics_v2/calibration_r5.json')
    initial=p.PilotPolicy().cuda()
    original=p.ReferenceTrainingEnv
    result=dict(envs=1024,rollout=32,epochs=2,models='disposable fresh clones',measurements={})
    for backend in ['reference_eager','previous_graph','native_graph']:
        p.ReferenceTrainingEnv=PreviousGraphEnv if backend=='previous_graph' else original
        arm=p.Arm(initial,p.balanced_training(1024),104729,native,'cuda',args.cache,True,reference=backend!='native_graph')
        if backend=='previous_graph':arm.sim=CudaGraphSimulator(arm.env)
        arm.update()
        samples=[arm.update() for _ in range(3)]
        seconds=statistics.median(s['collection_seconds']+s['optimization_seconds'] for s in samples)
        result['measurements'][backend]=dict(samples=samples,median_update_seconds=seconds,decisions_s=32768/seconds)
        print(backend,round(seconds,4),round(32768/seconds),flush=True)
        del arm
    p.ReferenceTrainingEnv=original
    args.output.write_text(json.dumps(result,indent=2),encoding='utf8')

if __name__=='__main__':main()
