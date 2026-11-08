"""CPU audit of controlled initial distributions, including rejection fallbacks."""
import json
from pathlib import Path
import sys
ROOT=Path(r'C:\Users\maxbr\Desktop\game\IsaacRL\binding-of-ai')
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
import torch
from research.crossfire_core import CrossScene,CrossEnv,PLACEMENTS,c
from physics_v2 import NativePhysics


def main():
    torch.set_num_threads(1);rows=[]
    for arrangement in ('ring','grouped'):
        for placement in PLACEMENTS:
            n=2048
            env=CrossEnv([CrossScene(0,4,placement=placement,arrangement=arrangement)]*n,981117,
                         NativePhysics(),'cpu',budget=1,optimizations=c.Optimizations())
            env.keep_terminal=False;env.reset_fallbacks.zero_();env.rng.manual_seed(981119)
            env.reset(torch.ones(n,dtype=torch.bool))
            radii=(env.targets-env.pos[:,None]).norm(dim=-1)
            pairs=torch.stack([(env.targets[:,i]-env.targets[:,j]).norm(dim=-1) for i in range(4) for j in range(i)])
            assert ((env.targets>=20)&(env.targets<=env.size-20)).all()
            assert radii.min()>=79.99 and pairs.min()>=34.99
            rows.append(dict(arrangement=arrangement,placement=placement,episodes=n,positions=n*4,
                             fallback_positions=int(env.reset_fallbacks.sum()),fallback_fraction=float(env.reset_fallbacks.sum())/(n*4),
                             minimum_player_distance=float(radii.min()),minimum_enemy_spacing=float(pairs.min())))
    out=Path(__file__).parent/'spawn-audit.json';out.write_text(json.dumps(dict(cpu_only=True,passed=True,rows=rows),indent=2))
    for row in rows:print(row)


if __name__=='__main__':main()
