"""Closed-factor final exam; refuse to run until all requested arms are frozen."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(r'C:\Users\maxbr\Desktop\game\IsaacRL\binding-of-ai')
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
import torch
try:
    from research.crossfire_core import *
    from research.diagnose_crossfire import load_policy,suites
except ModuleNotFoundError:
    from crossfire_core import *
    from diagnose_crossfire import load_policy,suites
from physics_v2 import NativePhysics


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--selection',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--per-cell',type=int,default=128)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    selection=json.loads(args.selection.read_text());assert selection['frozen']
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    native=NativePhysics.load(ROOT/'data/physics_v2/calibration_r5.json')
    report=dict(complete=False,selection_sha256=hashlib.sha256(args.selection.read_bytes()).hexdigest(),
                seeds=[970101,970201],per_cell=args.per_cell,models={})
    for key,model in selection['models'].items():
        path=Path(model['checkpoint']);assert hashlib.sha256(path.read_bytes()).hexdigest()==model['sha256']
        policy=load_policy(path);row=dict(checkpoint=str(path),sha256=model['sha256'],suites={});report['models'][key]=row
        for offset,(label,(scenes,width,height)) in enumerate(suites(final=True).items()):
            row['suites'][label]=[]
            for seed in (970101,970201):
                examiner=CrossExaminer(policy,scenes,args.per_cell,seed+offset*10,native,args.cache,width=width,height=height)
                result=examiner.run();row['suites'][label].append(result)
                print('FINAL',key,label,seed,result['macro']['clear_rate'],flush=True)
                (args.output/'report.json').write_text(json.dumps(report,indent=2))
                del examiner;gc.collect()
        # Historical distribution is a regression exam, not a newly closed test.
        scenes=[c.Scene(0,3),c.Scene(0,4)]+c.holdout_cells()
        examiner=c.Examiner(policy,scenes,args.per_cell,970301,native,args.cache,budget=1)
        row['historical']=examiner.run();del examiner
        del policy;gc.collect();torch.cuda.empty_cache()
    report['complete']=True;(args.output/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
