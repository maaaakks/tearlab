"""Frozen new-layout audit after curricula are selected on development only."""
import gc,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=Path(r'C:\Users\maxbr\Desktop\game\IsaacRL\binding-of-ai')
sys.path[:0]=[str(REPO),str(REPO/'python')]
import numpy as np
import torch
from research.mastery_core import Examiner,dev_cells,x
from physics_v2 import NativePhysics

def main():
    run=ROOT/'focus-paired';training=json.loads((run/'report.json').read_text());assert training['complete']
    out=run/'frozen-audit';out.mkdir(exist_ok=False)
    parent=Path(training['parent']);paths={'parent':parent}
    for name,arm in training['arms'].items():
        paths[name+'-selected']=run/name/'best.pt' if arm['selected']=='best' else parent
        paths[name+'-latest']=run/name/'latest.pt'
    dev_scores={strategy:np.mean([a['best_score'] for a in training['arms'].values() if a['strategy']==strategy],axis=0).tolist() for strategy in ('strict','recovery')}
    chosen=max(('strict','recovery'),key=lambda s:tuple(dev_scores[s]))
    (run/'frozen-recipe.json').write_text(json.dumps(dict(strategy=chosen,development_scores=dev_scores,
        rule='Mean development ranking across training seeds, strict first on exact tie; no audit used',
        models={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()}),indent=2))
    torch.set_num_threads(1);model=x.CrossPolicy().cuda();model.load_state_dict(torch.load(parent,map_location='cpu',weights_only=True)['state_dict'])
    native=NativePhysics.load(REPO/'data/physics_v2/calibration_r5.json')
    report=dict(complete=False,repeatable=True,seeds=[1100101,1100201],per_cell=128,
                note='Fresh random layouts of known development factors, not reserved OOD factors; model selection was frozen before audit',
                models={name:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),cohorts=[]) for name,p in paths.items()})
    def save():(out/'report.json').write_text(json.dumps(report,indent=2))
    save()
    for budget in (1,3):
        exam=Examiner(model,dev_cells(),128,1100101,native,ROOT.parent/'isaac-training-parallel/cache',budget=budget)
        for name,path in paths.items():
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'])
            for seed in report['seeds']:
                exam.seed=seed;result=exam.run();raw=exam.stats.cpu().numpy().copy()
                np.savez_compressed(out/f'{name}-b{budget}-s{seed}.npz',stats=raw)
                if name=='parent' and seed==report['seeds'][0]:
                    exam.run();difference=float(np.max(np.abs(raw-exam.stats.cpu().numpy())))
                    report[f'repeat_max_diff_b{budget}']=difference;report['repeatable'] &= difference==0
                report['models'][name]['cohorts'].append(result);save()
                print('AUDIT',budget,name,seed,flush=True)
        del exam;gc.collect();torch.cuda.empty_cache()
    report['complete']=True;save();print('REPEATABLE',report['repeatable'],flush=True)
if __name__=='__main__':main()
