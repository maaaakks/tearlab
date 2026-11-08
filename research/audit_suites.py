"""Fresh fixed-weight audits, including original pilot baselines when requested."""
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
    from research.campaign_core import *
except ModuleNotFoundError:
    from campaign_core import *
from physics_v2 import NativePhysics

class LegacyAdapter(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,obs,previous):return self.model(obs[:,:236],previous)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoints',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=930001)
    parser.add_argument('--per-cell',type=int,default=128)
    args=parser.parse_args();torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    native=NativePhysics.load(ROOT/'data/physics_v2/calibration_r5.json')
    report=dict(complete=False,seed=args.seed,per_cell=args.per_cell,
                description='Fixed weights, greedy actions, fresh shared cohorts; no selection on these scores',models={})
    args.output.write_text(json.dumps(report,indent=2))
    hp_cells=sorted({Scene(s.chasers,s.shooters,12) for s in old_id_cells() if s.chasers},key=lambda s:s.name)
    tests={
        'skills':([Scene(shooter_hp=2,attacks=False),Scene(),Scene(0,2),Scene(1,1,4)],520,280),
        'legacy_suite':(old_id_cells(),520,280),
        'compositions':(holdout_cells(),520,280),
        'hp12':(hp_cells,520,280),
        'geometry':(old_id_cells(),640,360)}
    for index,path in enumerate(args.checkpoints):
        checkpoint=torch.load(path,map_location='cuda',weights_only=True)
        legacy=checkpoint.get('schema')!='combat-generalization-v1'
        model=p.PilotPolicy().cuda() if legacy else Policy().cuda()
        model.load_state_dict(checkpoint['state_dict'])
        policy=LegacyAdapter(model) if legacy else model
        key=f'{path.parent.name}/{path.name}'
        report['models'][key]=dict(checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  legacy=legacy,metrics={})
        for budget in (1,3):
            for offset,(label,(scenes,width,height)) in enumerate(tests.items()):
                examiner=Examiner(policy,scenes,args.per_cell,args.seed+offset*10+budget,
                                  native,args.cache,budget=budget,width=width,height=height)
                result=examiner.run()
                report['models'][key]['metrics'][f'life{budget}/{label}']=result
                args.output.write_text(json.dumps(report,indent=2))
                print('AUDIT',key,budget,label,json.dumps(result['macro']),flush=True)
                del examiner;gc.collect()
        del policy,model,checkpoint;gc.collect();torch.cuda.empty_cache()
    report['complete']=True;args.output.write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
