"""Equal-exposure objective ablation across simultaneous native geometries."""
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
try:from research.mastery_core import *
except ModuleNotFoundError:from mastery_core import *
from physics_v2 import NativePhysics


def write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf8')


def snapshot(folder):
    hashes=c.p.sources(folder)
    for name in ['parallel_ppo.py','campaign_core.py','crossfire_core.py','mastery_core.py','run_mastery.py','test_mastery.py']:
        source=ROOT/'research'/name
        if not source.exists():source=Path(__file__).parent/name
        target=folder/'sources/research'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        hashes['research/'+name]=hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--modes',nargs='+',default=['legacy','win','potential']);ap.add_argument('--envs',type=int,default=1024)
    ap.add_argument('--updates',nargs='+',type=int,default=[400,400,1000]);ap.add_argument('--hp',nargs='+',type=int,default=[4,8,12])
    ap.add_argument('--check-every',type=int,default=200);ap.add_argument('--per-cell',type=int,default=64)
    args=ap.parse_args();args.parent=args.parent.resolve();args.output=args.output.resolve();args.cache=args.cache.resolve()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    native=NativePhysics.load(ROOT/'data/physics_v2/calibration_r5.json');parent=torch.load(args.parent,map_location='cpu',weights_only=True)
    report=dict(complete=False,config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                native=asdict(native),sources=snapshot(args.output),parent_sha256=hashlib.sha256(args.parent.read_bytes()).hexdigest(),arms={},live_validated=False)
    save_report=lambda:write(args.output/'report.json',report)
    save_report();service=Path(r'C:\Users\maxbr\Documents\Codex\.wandb-tmp');service.mkdir(exist_ok=True)
    tempfile.tempdir=str(service);os.environ['TEMP']=str(service);os.environ['TMP']=str(service)
    for key,name in [('WANDB_CONFIG_DIR','wandb_config'),('WANDB_CACHE_DIR','wandb_cache'),('WANDB_DATA_DIR','wandb_data')]:os.environ[key]=str(args.output/name)
    import wandb
    logger=wandb.init(project='IsaacRL',name='mastery-reward-'+args.output.name,tags=['mastery','reward-ablation','geometry','synthetic'],mode='offline',dir=str(args.output),config=report['config'])
    report['wandb']=dict(mode='offline',id=logger.id,dir=logger.dir);started=time.perf_counter()
    def checkpoint(learner,mode,label,metric):
        path=args.output/f'{mode}-{label}.pt'
        torch.save(dict(schema='mastery-v1',state_dict=learner.policy.state_dict(),optimizer=learner.optimizer.state_dict(),
                        updates=learner.updates,mode=mode,gamma=learner.gamma,metric=metric,sources=report['sources'],parent=report['parent_sha256']),path)
        write(path.with_suffix('.meta.json'),dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),updates=learner.updates,
              decisions=learner.updates*args.envs*32,mode=mode,metric=metric,live_validated=False,
              wandb_artifact_reference=f'offline:{logger.id}/{args.output.name}-{mode}-{label}'))
    try:
        for mode in args.modes:
            torch.manual_seed(args.seed);initial=x.CrossPolicy().cuda();initial.load_state_dict(parent['state_dict'])
            learner=Learner(initial,training_cases(args.hp[0],args.envs),args.seed,native,args.cache,mode)
            exams=[Examiner(learner.policy,dev_cells(),args.per_cell,seed,native,args.cache) for seed in (1000101,1000201)]
            baseline=[e.run() for e in exams]
            arm=dict(baseline=baseline,baseline_metric=quality(baseline),checks=[],stages=[],complete=False)
            report['arms'][mode]=arm;save_report();best=(-1.,-1.,-1.);wall=time.perf_counter();train=0.
            for hp,updates in zip(args.hp,args.updates):
                cases=training_cases(hp,args.envs);learner.set_cases(cases)
                arm['stages'].append(dict(hp=hp,updates=updates,cases=[asdict(s) for s in cases]))
                for step in range(updates):
                    stats=learner.update();train+=stats['collection_seconds']+stats['optimization_seconds']
                    if (step+1)%args.check_every==0 or step+1==updates:
                        results=[e.run() for e in exams];metric=quality(results)
                        arm['checks'].append(dict(updates=learner.updates,hp=hp,metric=metric,cohorts=results,training=stats))
                        if metric>best:best=metric;checkpoint(learner,mode,'best',metric)
                        logger.log({f'{mode}/retention_quality':metric[0],f'{mode}/worst_hard':metric[1],f'{mode}/mean_hard':metric[2],f'{mode}/entropy':stats['entropy']},step=learner.updates)
                        save_report();print('CHECK',mode,learner.updates,metric,'seconds',round(time.perf_counter()-wall,1),flush=True)
            checkpoint(learner,mode,'latest',arm['checks'][-1]['metric'])
            arm.update(complete=True,best_metric=best,elapsed_seconds=time.perf_counter()-wall,training_seconds=train,decisions=learner.updates*args.envs*32)
            for label in ('best','latest'):
                path=args.output/f'{mode}-{label}.pt';artifact=wandb.Artifact(args.output.name+'-'+mode+'-'+label,type='model')
                artifact.add_file(str(path));artifact.add_file(str(path.with_suffix('.meta.json')));logger.log_artifact(artifact)
            save_report();print('ARM COMPLETE',mode,best,flush=True)
            del learner,exams,initial;gc.collect();torch.cuda.empty_cache()
        report.update(complete=True,elapsed_seconds=time.perf_counter()-started);save_report()
    except BaseException as exc:report['error']=repr(exc);save_report();raise
    finally:logger.finish()


if __name__=='__main__':main()
