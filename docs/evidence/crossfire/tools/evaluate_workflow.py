"""Wait for frozen training, then audit and validate without further optimization."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
REPO=Path(r'C:\Users\maxbr\Desktop\game\IsaacRL\binding-of-ai')
CACHE=ROOT.parent/'isaac-training-parallel/cache'


def main():
    resume='--resume-after-tests' in sys.argv
    workflow=(json.loads((ROOT/'evaluation-workflow.json').read_text()) if resume else dict(complete=False,status='waiting-finish-training',steps=[]))
    def write():(ROOT/'evaluation-workflow.json').write_text(json.dumps(workflow,indent=2))
    write();deadline=time.monotonic()+3000
    while not resume and not json.loads((ROOT/'finish-training.json').read_text())['complete']:
        if time.monotonic()>deadline:raise TimeoutError('Training not finished')
        time.sleep(5)
    def run(name,script,args=(),cwd=ROOT):
        workflow['status']=name;write();started=time.time()
        with (ROOT/(name+'.log')).open('w',encoding='utf8') as log:
            subprocess.run([sys.executable,'-B',str(ROOT/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=cwd)
        workflow['steps'].append(dict(name=name,seconds=time.time()-started));write()
    if not resume:
        run('freeze','freeze_crossfire.py')
        run('final-audit','audit_crossfire.py',['--selection',ROOT/'selection.json','--output',ROOT/'final-audit','--cache',CACHE])
        selection=json.loads((ROOT/'selection.json').read_text());selected=selection.get('selected_model',selection['selected_variant'])
        paths=[selection['models'][f'{seed}-{selected}']['checkpoint'] for seed in (129773,242579)]
        run('post-diagnostic','diagnose_crossfire.py',['--checkpoints',*paths,'--output',ROOT/'post-diagnostic','--cache',CACHE,'--historical'])
        workflow['status']='tests-final';write();started=time.time()
        env=os.environ.copy();env['RUN_ISAAC_GPU_TESTS']='1';env['ISAAC_NATIVE_CACHE']=str(CACHE);env['PYTHONDONTWRITEBYTECODE']='1'
        with (ROOT/'tests-final.log').open('w',encoding='utf8') as log:
            subprocess.run([sys.executable,'-B','-m','unittest','research.test_campaign','research.test_parallel_ppo',
                            'research.test_pilot_10m','research.test_crossfire','-v'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        workflow['steps'].append(dict(name='tests-final',seconds=time.time()-started));write()
    else:
        assert (ROOT/'tests-final.log').read_text(encoding='utf8').strip().endswith('OK')
        assert (ROOT/'tests-confirmation.log').read_text(encoding='utf8').strip().endswith('OK')
        assert json.loads((ROOT/'final-audit/report.json').read_text())['complete']
        workflow['validation_recovery']='Explicit CUDA graph garbage collection between suites and model variants; two full normal-mode passes; no retraining or re-auditing'
    for name in ['analyze_crossfire','paired_intervals','validate_crossfire','render_results','write_results']:
        run(name,name+'.py')
    workflow.update(complete=True,status='all-audits-tests-and-report-complete');write()


if __name__=='__main__':main()
