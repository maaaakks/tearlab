"""Random and all 45 constant-action controls; no training or game interaction."""
import argparse
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

class Constants(torch.nn.Module):
    def __init__(self,per_action):
        super().__init__()
        self.actions=torch.arange(45,device='cuda').repeat_interleave(per_action)

    def forward(self,obs,previous):
        return torch.nn.functional.one_hot(self.actions,45).float()*10,torch.zeros_like(obs[:,0])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1)
    native=NativePhysics.load(ROOT/'data/physics_v2/calibration_r5.json')
    result={}
    scenes=[Scene(shooter_hp=2,attacks=False),Scene(),Scene(0,2),Scene(1,1,4)]
    for budget in (1,3):
        result[f'life{budget}']={}
        random=Examiner(None,scenes,256,910091+budget,native,args.cache,budget=budget).run()
        result[f'life{budget}']['random']=random
        result[f'life{budget}']['constant']={}
        for scene in scenes:
            examiner=Examiner(Constants(64),[scene]*45,64,910093+budget,native,args.cache,budget=budget)
            data=examiner.run()
            for action,row in enumerate(data['cells']):row['action']=action
            data['best_clear']=max(data['cells'],key=lambda row:row['clear_rate'])
            result[f'life{budget}']['constant'][scene.name]=data
            print(budget,scene.name,'best constant',data['best_clear']['action'],data['best_clear']['clear_rate'],flush=True)
        args.output.write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
