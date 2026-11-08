"""Serialize GPU experiments; no concurrent training, no final-test access."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
REPO=Path(r'C:\Users\maxbr\Desktop\game\IsaacRL\binding-of-ai')
PARENT=ROOT.parent/'isaac-generalization'
CACHE=ROOT.parent/'isaac-training-parallel/cache'


def main():
    first=ROOT/'paired-seed129773/report.json';deadline=time.monotonic()+1800
    workflow=dict(complete=False,status='waiting-first-seed',steps=[])
    def write():(ROOT/'workflow.json').write_text(json.dumps(workflow,indent=2))
    write()
    while True:
        report=json.loads(first.read_text())
        if report.get('error'):raise RuntimeError(report['error'])
        if report.get('complete'):break
        if time.monotonic()>deadline:raise TimeoutError('First run exceeded workflow deadline')
        time.sleep(5)
    best={k:v['best_metric'] for k,v in report['arms'].items()}
    control=max(best[k] for k in ('risk','attention'))>best['pool']
    workflow['capacity_control_required']=control;workflow['first_seed_development']=best
    jobs=[]
    if control:
        jobs.append(('capacity-seed129773',129773,PARENT/'mixed-seed129773/life1-sparse-best.pt',['capacity']))
    jobs.append(('paired-seed242579',242579,PARENT/'replication-seed242579/mixed/life1-sparse-best.pt',
                 ['pool','risk','attention']+(['capacity'] if control else [])))
    for name,seed,parent,variants in jobs:
        workflow['status']=name;write()
        cmd=[sys.executable,'-B',str(ROOT/'run_crossfire.py'),'--parent',str(parent),'--output',str(ROOT/name),
             '--cache',str(CACHE),'--seed',str(seed),'--variants',*variants]
        started=time.time()
        with (ROOT/(name+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=REPO)
        workflow['steps'].append(dict(name=name,seconds=time.time()-started));write()
    workflow.update(complete=True,status='training-finished-no-final-test-opened');write()


if __name__=='__main__':main()
