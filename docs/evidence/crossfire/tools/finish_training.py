"""Bounded preregistered capacity checks and weak-task focus, before final exams."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
CACHE=ROOT.parent/'isaac-training-parallel/cache'


def reports():
    outputs={}
    for seed in (129773,242579):
        report=json.loads((ROOT/f'paired-seed{seed}/report.json').read_text());assert report['complete']
        outputs[seed]={v:(arm,ROOT/f'paired-seed{seed}'/(v+'-best.pt')) for v,arm in report['arms'].items()}
        path=ROOT/f'capacity-seed{seed}/report.json'
        if path.exists():
            extra=json.loads(path.read_text());assert extra['complete']
            outputs[seed]['capacity']=(extra['arms']['capacity'],path.parent/'capacity-best.pt')
    return outputs


def main():
    workflow=dict(complete=False,status='waiting-primary-training',steps=[])
    def write():(ROOT/'finish-training.json').write_text(json.dumps(workflow,indent=2))
    write();deadline=time.monotonic()+2400
    while not json.loads((ROOT/'workflow.json').read_text())['complete']:
        if time.monotonic()>deadline:raise TimeoutError('Primary workflow deadline')
        time.sleep(5)
    def run(name,seed,parent,variant,focus=False):
        cmd=[sys.executable,'-B',str(ROOT/'run_crossfire.py'),'--parent',str(parent),'--output',str(ROOT/name),
             '--cache',str(CACHE),'--seed',str(seed),'--variants',variant]
        if focus:cmd+=['--stages','4:12','--per-stage','1200']
        workflow['status']=name;write();started=time.time()
        with (ROOT/(name+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        workflow['steps'].append(dict(name=name,seconds=time.time()-started));write()
    arms=reports()
    need_control=any(max(arms[s][v][0]['best_metric'] for v in ('risk','attention'))>arms[s]['pool'][0]['best_metric'] for s in arms)
    if need_control:
        for seed in arms:
            if 'capacity' not in arms[seed]:
                parent=json.loads((ROOT/f'paired-seed{seed}/report.json').read_text())['config']['parent']
                run(f'capacity-seed{seed}',seed,parent,'capacity')
        arms=reports()
    common=set.intersection(*(set(a) for a in arms.values()))
    ranking={v:(min(arms[s][v][0]['best_metric'][0] for s in arms),
                sum(arms[s][v][0]['best_metric'][1] for s in arms)/len(arms),
                min(arms[s][v][0]['best_metric'][2] for s in arms)) for v in sorted(common)}
    winner=max(ranking,key=ranking.get);workflow.update(ranking=ranking,selected_variant=winner)
    hard_scores={}
    for seed in arms:
        arm,path=arms[seed][winner]
        cp=json.loads(path.with_suffix('.meta.json').read_text())
        chosen=next(row for row in arm['checks'] if row['updates']==cp['updates'])
        hard=[cell['clear_rate'] for cohort in chosen['cohorts'] for cell in cohort['cells']
              if cell['scene']['shooters']==4 and cell['scene']['shooter_hp']==12]
        hard_scores[str(seed)]=sum(hard)/len(hard)
    focus_needed=min(hard_scores.values())<.7;workflow.update(focus_needed=focus_needed,pre_focus_4T12_development=hard_scores);write()
    if focus_needed:
        for seed in arms:run(f'focus-seed{seed}',seed,arms[seed][winner][1],winner,focus=True)
    workflow.update(complete=True,status='all-training-finished-no-final-test-opened');write()


if __name__=='__main__':main()
