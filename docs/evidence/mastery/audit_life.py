"""Frozen, shared-examiner comparison with repeatability and per-episode evidence."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=Path(r'C:\Users\maxbr\Desktop\game\IsaacRL\binding-of-ai')
sys.path[:0]=[str(REPO),str(REPO/'python')]
import numpy as np
import torch
from research.mastery_core import Examiner,dev_cells,x
from physics_v2 import NativePhysics
def main():
    output=ROOT/'life-pilot/frozen-audit';output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    paths={'parent':ROOT.parent/'isaac-crossfire/focus-seed129773/pool-best.pt',
           'contacts1':ROOT/'life-pilot/contacts1/potential-latest.pt',
           'contacts3':ROOT/'life-pilot/contacts3/potential-latest.pt'}
    model=x.CrossPolicy().cuda();model.load_state_dict(torch.load(paths['parent'],weights_only=True,map_location='cpu')['state_dict'])
    native=NativePhysics.load(REPO/'data/physics_v2/calibration_r5.json')
    report=dict(complete=False,per_cell=128,models={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),cohorts=[]) for k,p in paths.items()},repeatable=True)
    def save():(output/'report.json').write_text(json.dumps(report,indent=2))
    save()
    for budget in (1,3):
        examiner=Examiner(model,dev_cells(),128,1000101,native,ROOT.parent/'isaac-training-parallel/cache',budget=budget)
        for name,path in paths.items():
            model.load_state_dict(torch.load(path,weights_only=True,map_location='cpu')['state_dict'])
            for seed in (1000101,1000201):
                examiner.seed=seed;result=examiner.run();raw=examiner.stats.cpu().numpy().copy()
                np.savez_compressed(output/f'{name}-b{budget}-s{seed}.npz',stats=raw)
                if name=='parent' and seed==1000101:
                    examiner.run();difference=float(np.max(np.abs(raw-examiner.stats.cpu().numpy())))
                    report[f'repeat_max_difference_budget{budget}']=difference
                    report['repeatable'] &= difference==0
                report['models'][name]['cohorts'].append(result);save()
                print('AUDIT',budget,name,seed,flush=True)
        del examiner
        import gc
        gc.collect();torch.cuda.empty_cache()
    report['complete']=True;save()
    print('REPEATABLE',report['repeatable'],flush=True)
if __name__=='__main__':main()
