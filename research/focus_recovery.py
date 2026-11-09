"""Matched recovery->strict vs strict->strict; long-range focus, frozen exams."""
import argparse,copy,gc,hashlib,json,os,shutil,sys,tempfile,time
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'python')]
import torch
from research.mastery_core import Case,HardScene,RETENTION,Learner,Examiner,dev_cells,quality,x
from physics_v2 import NativePhysics

def focus_cases(n=1024):
    cases=[]
    sizes=[n//5,3*n//10,n-n//5-3*n//10]
    long=[HardScene(0,4,pattern='long',placement=p) for p in ('free','edge','corner')]
    other=[HardScene(0,4,pattern='surround'),HardScene(0,3,pattern='surround')]+[HardScene(0,4,placement=p) for p in ('free','edge','corner')]
    for (w,h),size in zip([(440,240),(520,280),(640,360)],sizes):
        a=size//2;b=size//5
        scenes=[long[i%len(long)] for i in range(a)]+[other[i%len(other)] for i in range(b)]+[RETENTION[i%len(RETENTION)] for i in range(size-a-b)]
        cases.extend(Case(s,w,h) for s in scenes)
    return cases

def score(cohorts):
    strict=[r for r in cohorts if r['budget']==1]
    retention,_,_=quality(strict)
    # Pool cohorts per cell before choosing the worst large-room long-range cell.
    cells=strict[0]['cells'];rates=[sum(r['cells'][i]['clear_rate'] for r in strict)/len(strict) for i in range(len(cells))]
    long=[v for c,v in zip(cells,rates) if c['case']['width']==640 and c['case']['scene']['pattern']=='long']
    hard=[v for c,v in zip(cells,rates) if c['case']['scene']['pattern']!='random']
    return (retention,min(long),sum(long)/len(long),sum(hard)/len(hard))

def write(path,obj):path.write_text(json.dumps(obj,indent=2),encoding='utf8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--parent',type=Path,required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=[129773,291773]);ap.add_argument('--updates',type=int,nargs=2,default=[200,400])
    ap.add_argument('--per-cell',type=int,default=32);ap.add_argument('--envs',type=int,default=1024)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if not all(u>0 for u in args.updates) or args.envs*32%4096:raise ValueError('Positive phase budgets and complete 4096 minibatches required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    parent=args.parent.resolve()
    initial_state=torch.load(parent,map_location='cpu',weights_only=True)['state_dict']
    cache=REPO/'artifacts/native-cache';native=NativePhysics.load(REPO/'data/physics_v2/calibration_r5.json')
    cases=focus_cases(args.envs)
    report=dict(complete=False,config={**vars(args),'output':str(out),'parent':str(parent)},parent=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
                native=asdict(native),cases=[asdict(s) for s in cases],arms={},live_validated=False,
                phase_boundary='Both arms restart Adam, episodes and generators at the boundary; weights preserved. This isolates the whole recovery-first curriculum, not optimizer continuity.')
    source_dir=out/'sources';source_dir.mkdir()
    report['sources']={}
    for p in [Path(__file__),*[REPO/'research'/n for n in ('mastery_core.py','parallel_ppo.py','campaign_core.py','crossfire_core.py','staged_simulator.py')]]:
        shutil.copyfile(p,source_dir/p.name);report['sources'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    save=lambda:write(out/'report.json',report)
    save()
    service=Path(tempfile.gettempdir())/'isaacrl-wandb'
    service.mkdir(exist_ok=True);tempfile.tempdir=str(service);os.environ['TEMP']=str(service);os.environ['TMP']=str(service)
    for key in ('WANDB_CONFIG_DIR','WANDB_CACHE_DIR','WANDB_DATA_DIR'):os.environ[key]=str(out/key.lower())
    import wandb
    log=wandb.init(project='IsaacRL',name='long-range-recovery-'+out.name,tags=['hard-focus','contact-curriculum','paired','synthetic'],mode='offline',dir=str(out),config=report['config'])
    report['wandb']=dict(mode='offline',id=log.id,dir=log.dir)
    eval_policy=x.CrossPolicy().cuda();eval_policy.load_state_dict(initial_state)
    exams=[Examiner(eval_policy,dev_cells(),args.per_cell,seed,native,cache,budget=b) for b in (1,3) for seed in (1000101,1000201)]
    def evaluate(policy):
        eval_policy.load_state_dict(policy.state_dict());return [e.run() for e in exams]
    baseline=[e.run() for e in exams];report['baseline']=baseline;report['baseline_score']=score(baseline);save()
    try:
        for seed in args.seeds:
            for strategy,budgets in [('strict',[1,1]),('recovery',[3,1])]:
                key=f'{strategy}-seed{seed}';folder=out/key;folder.mkdir()
                arm=dict(seed=seed,strategy=strategy,budgets=budgets,complete=False,checks=[],best_score=report['baseline_score'],selected='parent',decisions=0)
                report['arms'][key]=arm;save();state=copy.deepcopy(initial_state);started=time.perf_counter();total_updates=0
                def checkpoint(learner,label,metric):
                    path=folder/(label+'.pt')
                    torch.save(dict(schema='focus-recovery-v1',state_dict=learner.policy.state_dict(),optimizer=learner.optimizer.state_dict(),
                                    updates=total_updates,strategy=strategy,seed=seed,metric=metric,parent=report['parent_sha256'],sources=report['sources']),path)
                    meta=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),updates=total_updates,decisions=total_updates*args.envs*32,
                              strategy=strategy,seed=seed,metric=metric,live_validated=False,wandb_artifact_reference=f'offline:{log.id}/{key}-{label}')
                    write(path.with_suffix('.meta.json'),meta)
                for phase,(budget,updates) in enumerate(zip(budgets,args.updates)):
                    torch.manual_seed(seed+phase*1000000)
                    initial=x.CrossPolicy().cuda();initial.load_state_dict(state)
                    learner=Learner(initial,cases,seed+phase*1000000,native,cache,'potential',budget)
                    for step in range(updates):
                        stats=learner.update();total_updates+=1;arm['decisions']+=args.envs*32
                        if (step+1)%100==0 or step+1==updates:
                            results=evaluate(learner.policy);metric=score(results)
                            arm['checks'].append(dict(updates=total_updates,phase=phase,budget=budget,score=metric,cohorts=results,training=stats))
                            if phase==1 and tuple(metric)>tuple(arm['best_score']):
                                arm['best_score']=metric;arm['selected']='best';checkpoint(learner,'best',metric)
                            log.log({key+'/strict_retention':metric[0],key+'/worst_large_long':metric[1],key+'/mean_large_long':metric[2],key+'/entropy':stats['entropy']})
                            save();print('CHECK',key,total_updates,'budget',budget,metric,flush=True)
                    checkpoint(learner,f'phase{phase+1}',metric)
                    if phase==1:checkpoint(learner,'latest',metric)
                    state={k:v.detach().cpu().clone() for k,v in learner.policy.state_dict().items()}
                    del learner,initial;gc.collect();torch.cuda.empty_cache()
                arm.update(complete=True,seconds=time.perf_counter()-started)
                for p in folder.glob('*.pt'):
                    artifact=wandb.Artifact(key+'-'+p.stem,type='model');artifact.add_file(str(p));artifact.add_file(str(p.with_suffix('.meta.json')));log.log_artifact(artifact)
                save();print('ARM COMPLETE',key,arm['best_score'],'seconds',round(arm['seconds'],1),flush=True)
        report['complete']=True;save()
    except BaseException as exc:report['error']=repr(exc);save();raise
    finally:log.finish()
if __name__=='__main__':main()
