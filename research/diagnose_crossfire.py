"""First-episode diagnostics with compact last-two-second windows at every terminal."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(r'C:\Users\maxbr\Desktop\game\IsaacRL\binding-of-ai')
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
import numpy as np
import torch
try:from research.crossfire_core import *
except ModuleNotFoundError:from crossfire_core import *
from physics_v2 import NativePhysics


def load_policy(path,device='cuda'):
    cp=torch.load(path,map_location=device,weights_only=True)
    policy=CrossPolicy(cp.get('variant','pool')).to(device)
    policy.load_state_dict(cp['state_dict']);policy.eval()
    return policy


def suites(final=False):
    basic=[CrossScene(0,k,shooter_hp=hp,placement=p) for k in (2,3,4) for hp in (4,12) for p in ('center','free','edge','corner')]
    if not final:return dict(controlled=(basic,520,280))
    return dict(controlled=(basic,520,280),
                regression=([CrossScene(s.chasers,s.shooters,s.chaser_hp) for s in c.holdout_cells()],520,280),
                grouped=([CrossScene(0,k,placement=p,arrangement='grouped') for k in (2,3,4) for p in ('free','edge','corner')],520,280),
                synchronized=([CrossScene(0,k,placement=p,synchronized=True) for k in (2,3,4) for p in ('free','edge','corner')],520,280),
                geometry=(basic,640,360))


METRICS=['x','y','wall_distance','corner','hostile_count','nearest_bullet','closest_passage_distance','closest_passage_time','speed','hp_removed',
         'vx','vy','requested_action','previous_action','shot_axis_enemy_proxy','shot_axis_distance_proxy']


@torch.no_grad()
def diagnose(policy,scenes,per_cell,seed,native,cache,width=520,height=280,historical=False):
    expanded=[s for s in scenes for _ in range(per_cell)]
    if historical:
        sim,env,_=c.make_env(expanded,seed,native,'cuda',cache,budget=1,reward_mode='sparse',width=width,height=height)
    else:sim,env,_=make_env(expanded,seed,native,'cuda',cache,width=width,height=height)
    n=env.n;device=env.device
    active=torch.ones(n,device=device,dtype=torch.bool);previous=torch.full((n,),45,device=device,dtype=torch.long)
    # clear, death projectile, death contact, timeout, duration, progress, shots
    totals=torch.zeros(n,7,device=device)
    ring=torch.zeros(20,n,len(METRICS),device=device);windows=torch.zeros_like(ring)
    end_step=torch.zeros(n,device=device,dtype=torch.long)
    # Exposure denominator: compare near-wall frequency at death with time spent there.
    exposure=torch.zeros(n,3,device=device)
    for step in range(400):
        obs=env.obs();bullets=obs[:,76:236].reshape(n,32,5);mask=bullets[:,:,4]>0
        relative=bullets[:,:,:2]*320
        nearest=relative.norm(dim=-1).masked_fill(~mask,1e4).amin(-1)
        passage=closest_passage(obs)
        passage_distance=(passage[:,:,1]*320).masked_fill(~mask,1e4)
        nearest_index=passage_distance.argmin(-1)
        minpass=passage_distance.amin(-1)
        min_time=(passage[:,:,0]*2).gather(1,nearest_index[:,None]).squeeze(1)
        boundaries=torch.minimum(env.pos,env.size-env.pos)
        wall=boundaries.amin(-1);corner=(boundaries<35).all(-1)
        action=policy(obs,previous)[0].argmax(-1)
        axis=env.move[action%5];relative_enemy=env.targets-env.pos[:,None]
        ahead=(relative_enemy*axis[:,None]).sum(-1)>0
        distance=(relative_enemy[:,:,0]*axis[:,None,1]-relative_enemy[:,:,1]*axis[:,None,0]).abs()
        distance=distance.masked_fill(~ahead|(env.hp<=0)|(action[:,None]%5==0),1e4)
        axis_distance=distance.amin(-1);axis_enemy=distance.argmin(-1)
        axis_enemy=torch.where(axis_distance<1e4,axis_enemy,-1)
        diagnostic=torch.stack((env.pos[:,0],env.pos[:,1],wall,corner.float(),mask.sum(-1).float(),nearest,minpass,min_time,env.vel.norm(dim=-1),1-env.hp.sum(-1)/env.initial_hp,
                                env.vel[:,0],env.vel[:,1],action.float(),previous.float(),axis_enemy.float(),axis_distance),-1)
        ring[step%20].copy_(diagnostic)
        exposure[:,0].add_(active);exposure[:,1].add_(active*(wall<35));exposure[:,2].add_(active*corner)
        _,_,done,info=sim.step(action)
        newly=active&done
        totals[:,0].add_(active*info['success']);totals[:,1].add_(active*info['dead']*info['projectile_damage'])
        totals[:,2].add_(active*info['dead']*info['contact_damage']);totals[:,3].add_(active*info['timeout'])
        totals[:,4].add_(active);totals[:,5].add_(newly*info['progress_fraction']);totals[:,6].add_(active*info['fired'])
        windows.copy_(torch.where(newly[None,:,None],ring,windows));end_step.copy_(torch.where(newly,step,end_step))
        active.logical_and_(~done);previous.copy_(torch.where(done,45,action))
        if not bool(active.any()):break
    if active.any():raise RuntimeError('Unfinished diagnostic')
    values=totals.cpu().numpy();raw=windows.cpu().numpy();ends=end_step.cpu().numpy();exp=exposure.cpu().numpy()
    # Rotate every terminal window into chronological order, NaN padding early deaths.
    ordered=np.full_like(raw,np.nan)
    for i,t in enumerate(ends):
        length=min(int(t)+1,20);ids=np.arange(int(t)-length+1,int(t)+1)%20
        ordered[-length:,i]=raw[ids,i]
    rows=[]
    for index,scene in enumerate(scenes):
        part=values[index*per_cell:(index+1)*per_cell];trace=ordered[:,index*per_cell:(index+1)*per_cell]
        ex=exp[index*per_cell:(index+1)*per_cell];dead=part[:,1:3].sum(-1)>0
        last=trace[-1]
        rows.append(dict(name=scene.name,scene=asdict(scene),episodes=per_cell,clear_rate=float(part[:,0].mean()),
                         projectile_death_rate=float(part[:,1].mean()),contact_death_rate=float(part[:,2].mean()),
                         timeout_rate=float(part[:,3].mean()),decisions=float(part[:,4].mean()),hp_removed=float(part[:,5].mean()),
                         shots=float(part[:,6].mean()),wall_exposure_rate=float(ex[:,1].sum()/max(ex[:,0].sum(),1)),
                         corner_exposure_rate=float(ex[:,2].sum()/max(ex[:,0].sum(),1)),
                         wall_at_death_rate=float((last[dead,2]<35).mean()) if dead.any() else None,
                         corner_at_death_rate=float(last[dead,3].mean()) if dead.any() else None,
                         hostile_at_death=float(last[dead,4].mean()) if dead.any() else None))
    fallback=int(env.reset_fallbacks.sum()) if not historical else None
    return dict(seed=seed,historical=historical,width=width,height=height,cells=rows,reset_fallbacks=fallback),dict(totals=values,last20=ordered,end_step=ends,exposure=exp,metric_names=np.array(METRICS))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--per-cell',type=int,default=128);ap.add_argument('--final',action='store_true')
    ap.add_argument('--historical',action='store_true');ap.add_argument('--seed',type=int,default=950101)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    native=NativePhysics.load(ROOT/'data/physics_v2/calibration_r5.json')
    report=dict(complete=False,final=args.final,seed=args.seed,per_cell=args.per_cell,models={})
    for modelindex,path in enumerate(args.checkpoints):
        policy=load_policy(path);key=str(path);result={};report['models'][key]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),suites=result)
        for offset,(name,(scenes,width,height)) in enumerate(suites(args.final).items()):
            row,raw=diagnose(policy,scenes,args.per_cell,args.seed+offset*10,native,args.cache,width,height)
            result[name]=row;np.savez_compressed(args.output/f'model{modelindex}-{name}.npz',**raw)
            print('DIAG',modelindex,name,'clear',sum(x['clear_rate'] for x in row['cells'])/len(row['cells']),flush=True)
            (args.output/'report.json').write_text(json.dumps(report,indent=2));gc.collect()
        if args.historical:
            scenes=[c.Scene(0,k) for k in (2,3,4)]+c.holdout_cells()
            row,raw=diagnose(policy,scenes,args.per_cell,args.seed+100,native,args.cache,historical=True)
            result['historical']=row;np.savez_compressed(args.output/f'model{modelindex}-historical.npz',**raw)
        del policy;gc.collect();torch.cuda.empty_cache()
    report['complete']=True;(args.output/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
