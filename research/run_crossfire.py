"""Fixed-exposure, paired-seed curriculum and representation ablation, offline."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
ROOT=Path(r'C:\Users\maxbr\Desktop\game\IsaacRL\binding-of-ai')
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
import torch
try:from research.crossfire_core import *
except ModuleNotFoundError:from crossfire_core import *
from physics_v2 import NativePhysics


def write_json(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf8')


def sources(folder):
    hashes=c.p.sources(folder)
    for name in ['campaign_core.py','parallel_ppo.py','crossfire_core.py','run_crossfire.py','diagnose_crossfire.py','test_crossfire.py']:
        source=ROOT/'research'/name
        if not source.exists():source=Path(__file__).parent/name
        dest=folder/'sources/research'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest);hashes['research/'+name]=hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes


def score(result):
    rows=result['cells'];ret=rows[:len(RETENTION)];target=rows[len(RETENTION):]
    quality=min(1.,ret[0]['clear_rate']/.8,ret[1]['clear_rate']/.6,ret[2]['clear_rate']/.65,
                sum(x['clear_rate'] for x in ret[3:])/3/.5)
    target_score=sum(x['clear_rate'] for x in target)/len(target)
    return (quality,target_score,min(x['clear_rate'] for x in target),-sum(x['decisions'] for x in target)/len(target))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parent',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--seed',type=int,required=True);ap.add_argument('--variants',nargs='+',default=['pool','risk','attention'])
    ap.add_argument('--per-stage',type=int,default=400);ap.add_argument('--check-every',type=int,default=200)
    ap.add_argument('--envs',type=int,default=1024);ap.add_argument('--per-cell',type=int,default=64)
    ap.add_argument('--stages',nargs='+',default=['3:4','3:12','4:4','4:12'])
    args=ap.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    native=NativePhysics.load(ROOT/'data/physics_v2/calibration_r5.json')
    parent=torch.load(args.parent,map_location='cpu',weights_only=True)
    report=dict(complete=False,config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                parent_sha256=hashlib.sha256(args.parent.read_bytes()).hexdigest(),source_sha256=sources(args.output),
                native=asdict(native),reward='sparse unchanged',budget=1,live_validated=False,arms={})
    write=lambda:write_json(args.output/'report.json',report)
    write()
    service=Path(r'C:\Users\maxbr\Documents\Codex\.wandb-tmp');service.mkdir(exist_ok=True)
    tempfile.tempdir=str(service);os.environ['TEMP']=str(service);os.environ['TMP']=str(service)
    for key,name in [('WANDB_CONFIG_DIR','wandb_config'),('WANDB_CACHE_DIR','wandb_cache'),('WANDB_DATA_DIR','wandb_data')]:os.environ[key]=str(args.output/name)
    import wandb
    logger=wandb.init(project='IsaacRL',name='crossfire-'+args.output.name,
                      tags=['crossfire','curriculum','representation-ablation','synthetic','offline'],mode='offline',dir=str(args.output),config=report['config'])
    report['wandb']=dict(mode='offline',id=logger.id,dir=logger.dir)
    start=time.perf_counter()
    def save(learner,variant,label,metric):
        path=args.output/f'{variant}-{label}.pt'
        torch.save(dict(schema='crossfire-v1',variant=variant,state_dict=learner.policy.state_dict(),
                        optimizer=learner.optimizer.state_dict(),updates=learner.updates,metric=metric,
                        sources=report['source_sha256'],parent=report['parent_sha256'],budget=1,reward='sparse',gamma=.995),path)
        meta=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),variant=variant,updates=learner.updates,
                  decisions=learner.updates*args.envs*32,metric=metric,live_validated=False,
                  wandb_artifact_reference=f'offline:{logger.id}/{args.output.name}-{variant}-{label}')
        write_json(path.with_suffix('.meta.json'),meta)
    try:
        for variant in args.variants:
            torch.manual_seed(args.seed)
            initial=CrossPolicy(variant)
            missing,unexpected=initial.load_state_dict(parent['state_dict'],strict=False)
            assert not unexpected
            assert all(k.startswith(('extra.','residual.','query.','key.','val.')) for k in missing)
            initial=initial.cuda()
            # Optimizer is deliberately fresh for ALL arms, including unchanged pool.
            # Different parameter lists prevent a fair reuse of the parent's Adam state.
            stages=[tuple(map(int,s.split(':'))) for s in args.stages]
            learner=CrossLearner(initial,c.balance(schedule(*stages[0]),args.envs),args.seed,native,args.cache)
            exams=[CrossExaminer(learner.policy,RETENTION+TARGETS,args.per_cell,s,native,args.cache) for s in (960101,960201)]
            arm=dict(parameters=sum(x.numel() for x in learner.policy.parameters()),baseline=[e.run() for e in exams],checks=[],stages=[])
            report['arms'][variant]=arm;write()
            best=(-1.,-1.,-1.,-float('inf'));arm_start=time.perf_counter();train_seconds=0.
            for count,hp in stages:
                scenes=schedule(count,hp)
                assert not any(s.synchronized or s.arrangement=='grouped' for s in scenes)
                assert not {(s.chasers,s.shooters) for s in scenes}&c.HOLDOUT
                learner.set_scenes(c.balance(scenes,args.envs))
                arm['stages'].append(dict(count=count,hp=hp,scenes=[asdict(s) for s in scenes]))
                for step in range(args.per_stage):
                    stats=learner.update();train_seconds+=stats['collection_seconds']+stats['optimization_seconds']
                    if (step+1)%args.check_every==0 or step+1==args.per_stage:
                        results=[e.run() for e in exams]
                        metrics=[score(r) for r in results]
                        # Retention must hold on BOTH cohorts, then average target success.
                        metric=(min(m[0] for m in metrics),sum(m[1] for m in metrics)/len(metrics),
                                min(m[2] for m in metrics),sum(m[3] for m in metrics)/len(metrics))
                        row=dict(updates=learner.updates,stage=[count,hp],metric=metric,cohorts=results,training=stats)
                        arm['checks'].append(row)
                        if metric>best:best=metric;save(learner,variant,'best',metric)
                        logger.log({f'{variant}/target_clear':metric[1],f'{variant}/retention_quality':metric[0],
                                    f'{variant}/entropy':stats['entropy'],f'{variant}/kl':stats['max_kl']},step=learner.updates)
                        print('CHECK',variant,learner.updates,'metric',metric,'seconds',round(time.perf_counter()-arm_start,1),flush=True)
                        write()
            save(learner,variant,'latest',arm['checks'][-1]['metric'])
            arm.update(complete=True,elapsed_seconds=time.perf_counter()-arm_start,training_seconds=train_seconds,
                       training_decisions=learner.updates*args.envs*32,best_metric=best,reset_fallbacks=int(learner.env.reset_fallbacks.sum()))
            for label in ('best','latest'):
                path=args.output/f'{variant}-{label}.pt';artifact=wandb.Artifact(args.output.name+'-'+variant+'-'+label,type='model')
                artifact.add_file(str(path));artifact.add_file(str(path.with_suffix('.meta.json')));logger.log_artifact(artifact)
            write();print('ARM COMPLETE',variant,arm['elapsed_seconds'],best,flush=True)
            del learner,exams,initial;gc.collect();torch.cuda.empty_cache()
        report.update(complete=True,elapsed_seconds=time.perf_counter()-start);write()
    except BaseException as exc:report['error']=repr(exc);write();raise
    finally:logger.finish()


if __name__=='__main__':main()
