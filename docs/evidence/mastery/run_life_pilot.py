"""Matched training exposure, changed contact budget; frozen live agent untouched."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
def main():
    output=ROOT/'life-pilot';output.mkdir(exist_ok=False)
    report=dict(complete=False,arms={},seed=129773,decisions_per_arm=300*1024*32,
                purpose='Development pilot: life budget 1 vs 3, same potential objective and hard cases')
    def write():(output/'workflow.json').write_text(json.dumps(report,indent=2))
    write()
    for budget in (1,3):
        started=time.perf_counter();folder=output/f'contacts{budget}'
        command=[sys.executable,'-B',str(ROOT/'run_mastery.py'),'--parent',str(ROOT.parent/'isaac-crossfire/focus-seed129773/pool-best.pt'),
                 '--output',str(folder),'--cache',str(ROOT.parent/'isaac-training-parallel/cache'),
                 '--seed','129773','--modes','potential','--budget',str(budget),'--updates','100','200','--hp','8','12',
                 '--check-every','100','--per-cell','32']
        report['arms'][str(budget)]=dict(status='running',command=command);write()
        with (output/f'contacts{budget}.log').open('w',encoding='utf8') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
        report['arms'][str(budget)].update(exit_code=result.returncode,seconds=time.perf_counter()-started,status='complete' if result.returncode==0 else 'failed');write()
        if result.returncode:raise RuntimeError(f'Life budget {budget} failed; see log')
    report['complete']=True;write()
if __name__=='__main__':main()
