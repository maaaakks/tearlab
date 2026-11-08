"""Bounded, reproducible combat experiments; all outputs in a new run folder."""
import argparse
import copy
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import tempfile

ROOT=Path(r'C:\Users\maxbr\Desktop\game\IsaacRL\binding-of-ai')
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
import torch
try:
    from research.campaign_core import *
except ModuleNotFoundError:
    from campaign_core import *
from physics_v2 import NativePhysics


def snapshot(folder):
    hashes={}
    candidates=[ROOT/'research'/n for n in ['pilot_10m.py','parallel_ppo.py','gpu_simulator.py',
                 'staged_simulator.py','native_runtime.py','native/isaac_kernels.h','native/isaac_kernels.cu']]
    candidates += [Path(__file__).parent/n for n in ['campaign_core.py','run_campaign.py','test_campaign.py']]
    for source in candidates:
        rel=source.relative_to(ROOT) if source.is_relative_to(ROOT) else Path('research')/source.name
        target=folder/'sources'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);hashes[str(rel)]=hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['aim','budget','curriculum','reward'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--seed',type=int,default=129773)
    parser.add_argument('--envs',type=int,default=1024)
    parser.add_argument('--updates',type=int,default=600)
    parser.add_argument('--seconds',type=float,default=600)
    parser.add_argument('--check-every',type=int,default=100)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--budget',type=int,default=3)
    parser.add_argument('--reward',default='potential')
    parser.add_argument('--lr',type=float,default=.0003)
    parser.add_argument('--gamma',type=float,default=.995)
    parser.add_argument('--audit',action='store_true')
    parser.add_argument('--phase',choices=['two4','two12','mixed'],default='two4')
    args=parser.parse_args()
    args.output=args.output.resolve()
    args.cache=args.cache.resolve()
    folder=args.output;folder.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    native=NativePhysics.load(ROOT/'data/physics_v2/calibration_r5.json')
    initial=Policy().cuda()
    parent=None
    if args.parent:
        parent=torch.load(args.parent,map_location='cuda',weights_only=True)
        initial.load_state_dict(parent['state_dict'])
    report=dict(complete=False,config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                source_sha256=snapshot(folder),from_scratch=args.parent is None,
                parent_sha256=hashlib.sha256(args.parent.read_bytes()).hexdigest() if args.parent else None,
                native=asdict(native),live_validated=False,baseline={},checks=[],phases=[],final={})
    def write(): (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    write()
    for key,name in [('WANDB_CONFIG_DIR','wandb_config'),('WANDB_CACHE_DIR','wandb_cache'),('WANDB_DATA_DIR','wandb_data')]:
        os.environ[key]=str(folder/name)
    # Short writable path avoids Windows AF_UNIX path limits and restricted
    # sandbox temp rename permissions in the local W&B service.
    service_temp=Path(r'C:\Users\maxbr\Documents\Codex\.wandb-tmp')
    service_temp.mkdir(exist_ok=True)
    tempfile.tempdir=str(service_temp)
    os.environ['TEMP']=str(service_temp)
    os.environ['TMP']=str(service_temp)
    import wandb
    logger=wandb.init(project='IsaacRL',name='generalization-'+args.mode+'-'+folder.name,
                      tags=['generalization','curriculum',args.mode,'native-cuda','synthetic'],
                      config=report['config'],mode='offline',dir=str(folder))
    report['wandb']=dict(mode='offline',id=logger.id,dir=logger.dir)
    if args.mode=='aim':
        settings=[(1,args.reward)]
        schedules=[('aim', [Scene(shooter_hp=2,attacks=False)],args.updates)]
    elif args.mode=='budget':
        settings=[(1,args.reward),(3,args.reward)]
        schedules=[(f'one-t{hp}',[Scene(shooter_hp=hp)],args.updates//4) for hp in [2,4,8,12]]
    elif args.mode=='reward':
        settings=[(args.budget,'sparse'),(args.budget,'potential')]
        schedules=[('two-t12',[Scene(0,2)]*8+[Scene(0,1)]*2+[Scene(0,2,shooter_hp=4)],args.updates)]
    else:
        settings=[(args.budget,args.reward)]
        stage_scenes={
            'two4':[Scene(0,2,shooter_hp=4)]*6+[Scene(0,1)]*2+[Scene(shooter_hp=2,attacks=False)],
            'two12':[Scene(0,2)]*8+[Scene(0,1)]*2+[Scene(0,2,shooter_hp=4)],
            'mixed':[Scene(1,1,4)]*3+[Scene(1,2,4)]*3+[Scene(3,1,4)]*3+
                    [Scene(0,2)]*2+[Scene(0,1)]*2+[Scene(2,0,4)]*2}
        schedules=[(args.phase,stage_scenes[args.phase],args.updates)]
    for _,scenes,_ in schedules:
        assert not {(s.chasers,s.shooters) for s in scenes}&HOLDOUT
    report['planned_schedule']=[dict(name=name,scenes=[asdict(s) for s in scenes],updates=updates) for name,scenes,updates in schedules]
    learners={}
    examiners={}
    best={}
    started=time.perf_counter()
    def save(learner,label,metric):
        path=folder/f'{learner.name}-{label}.pt'
        torch.save(dict(schema='combat-generalization-v1',state_dict=learner.policy.state_dict(),
                        optimizer=learner.optimizer.state_dict(),updates=learner.updates,
                        budget=learner.env.budget,reward=learner.env.reward_mode,
                        gamma=learner.gamma,native=asdict(native),metric=metric,
                        sources=report['source_sha256'],parent=report['parent_sha256']),path)
        metadata=dict(run=folder.name,learner=learner.name,label=label,metric=metric,
                      updates=learner.updates,decisions=learner.updates*args.envs*32,
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest(),live_validated=False,
                      wandb_artifact_reference=f'offline:{logger.id}/{folder.name}-{learner.name}-{label}')
        path.with_suffix('.meta.json').write_text(json.dumps(metadata,indent=2))
        return path
    try:
        for budget,reward in settings:
            learner=Learner(initial,balance(schedules[0][1],args.envs),args.seed,native,args.cache,
                            budget,reward,args.gamma,args.lr)
            if parent:
                learner.optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
                for group in learner.optimizer.param_groups:group['lr']=args.lr
            learners[learner.name]=learner
            scenes=([Scene(shooter_hp=2,attacks=False)] if args.mode=='aim' else
                    [Scene(),Scene(0,2),Scene(1,1,4),Scene(shooter_hp=2,attacks=False)])
            if args.mode=='curriculum':scenes += [Scene(0,2,shooter_hp=4),Scene(1,2,4),Scene(3,1,4),Scene(2,0,4)]
            examiners[learner.name]={f'life{b}':Examiner(learner.policy,scenes,256,910001+b,native,args.cache,budget=b)
                                      for b in [1,3]}
            baseline={k:e.run() for k,e in examiners[learner.name].items()}
            report['baseline'][learner.name]=baseline
            best[learner.name]=(-1.,-1.,-float('inf'))
        write()
        print('BASELINE',json.dumps({k:{b:x['cells'][0]['clear_rate'] for b,x in v.items()} for k,v in report['baseline'].items()}),flush=True)
        checks_passed={k:0 for k in learners}
        for phase,scenes,updates in schedules:
            for learner in learners.values():learner.set_scenes(balance(scenes,args.envs))
            report['phases'].append(dict(name=phase,start_seconds=time.perf_counter()-started,updates=updates))
            print('PHASE',phase,flush=True)
            for step in range(updates):
                if time.perf_counter()-started>=args.seconds:break
                stats={name:learner.update() for name,learner in learners.items()}
                total_updates=next(iter(learners.values())).updates
                if step%20==19:
                    print('TRAIN',phase,total_updates,round(time.perf_counter()-started,1),flush=True)
                if (step+1)%args.check_every==0 or step+1==updates:
                    row=dict(phase=phase,updates=total_updates,seconds=time.perf_counter()-started,
                             scores={},training=stats)
                    for name,learner in learners.items():
                        scores={k:e.run() for k,e in examiners[name].items()}
                        row['scores'][name]=scores
                        own_cells=scores[f'life{learner.env.budget}']['cells']
                        primary=0 if args.mode in ['aim','budget'] else (4 if args.mode=='curriculum' and args.phase=='two4' else 1)
                        own=own_cells[primary]
                        metric=(own['clear_rate'],own['nohit_rate'],-own['decisions'])
                        mastered=own['clear_rate']>=.9
                        if args.mode=='curriculum':
                            retention=own_cells[0]['clear_rate']
                            threshold=.85 if args.phase=='two4' else .7
                            quality=min(retention/.8,own['clear_rate']/threshold)
                            mastered=quality>=1
                            if args.phase=='mixed':
                                mixed=[own_cells[i] for i in [2,5,6]]
                                own={key:sum(x[key] for x in mixed)/3 for key in ['clear_rate','nohit_rate','decisions']}
                                quality=min(retention/.8,own_cells[1]['clear_rate']/.6,
                                            min(x['clear_rate'] for x in mixed)/.5,own['clear_rate']/.7)
                                mastered=quality>=1
                            metric=(quality,own['clear_rate'],own['nohit_rate'],-own['decisions'])
                        if metric>best[name]:best[name]=metric;save(learner,'best',metric)
                        save(learner,'latest',metric)
                        checks_passed[name]=checks_passed[name]+1 if mastered else 0
                        logger.log({f'{name}/clear':own['clear_rate'],f'{name}/nohit':own['nohit_rate'],
                                    f'{name}/entropy':stats[name]['entropy'],f'{name}/kl':stats[name]['max_kl']},step=total_updates)
                    report['checks'].append(row);write()
                    print('CHECK',json.dumps(dict(phase=phase,updates=total_updates,seconds=round(row['seconds'],1),
                        scores={k:{b:{s['name']:round(s['clear_rate'],3) for s in x['cells']} for b,x in v.items()}
                                for k,v in row['scores'].items()})),flush=True)
                    if args.mode in ['aim','curriculum'] and min(checks_passed.values())>=2:break
            if time.perf_counter()-started>=args.seconds:break
            if args.mode in ['aim','curriculum'] and min(checks_passed.values())>=2:break
        for name,learner in learners.items():
            # Keep last separate; final audit loads solely the ID-selected best.
            save(learner,'latest',best[name])
            selected=folder/f'{name}-best.pt'
            if not selected.exists():save(learner,'best',best[name])
            checkpoint=torch.load(selected,map_location='cuda',weights_only=True)
            learner.policy.load_state_dict(checkpoint['state_dict'])
            report['final'][name]=dict(updates=learner.updates,selected_updates=checkpoint['updates'],
                                      dev={k:e.run() for k,e in examiners[name].items()},totals=learner.totals.tolist())
            if args.audit:
                report['final'][name]['audit']={}
                for b in [1,3]:
                    tests={
                        'id':(old_id_cells(),920001+b,520,280),
                        'compositions':(holdout_cells(),920011+b,520,280),
                        'hp12':([Scene(s.chasers,s.shooters,12) for s in old_id_cells() if s.chasers],920021+b,520,280),
                        'geometry':(old_id_cells(),920031+b,640,360)}
                    for label,(test_scenes,seed,width,height) in tests.items():
                        examiner=Examiner(learner.policy,test_scenes,64,seed,native,args.cache,budget=b,width=width,height=height)
                        result=examiner.run()
                        report['final'][name]['audit'][f'life{b}/{label}']=result
                        print('AUDIT',name,b,label,result['macro'],flush=True)
                        del examiner;gc.collect()
            for label in ['best','latest']:
                path=folder/f'{name}-{label}.pt'
                artifact=wandb.Artifact(f'{folder.name}-{name}-{label}',type='model')
                artifact.add_file(str(path));artifact.add_file(str(path.with_suffix('.meta.json')))
                logger.log_artifact(artifact)
        report.update(complete=True,elapsed_seconds=time.perf_counter()-started,
                      stop='mastery' if args.mode in ['aim','curriculum'] and min(checks_passed.values())>=2 else 'bounded_budget',
                      training_decisions={k:l.updates*args.envs*32 for k,l in learners.items()})
        write()
        print('COMPLETE',report['elapsed_seconds'],report['stop'],flush=True)
    except BaseException as exc:
        report['error']=repr(exc);write();raise
    finally:logger.finish()


if __name__=='__main__':main()
