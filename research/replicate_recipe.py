"""Second-seed recipe with gated progression and bounded subprocesses.

Never starts a second stage until the previous process exited and its report is
complete. Existing completed stages can be resumed without overwriting outputs.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve();root.mkdir(exist_ok=True)
    runner=Path(__file__).parent/'run_campaign.py'
    progress=dict(seed=args.seed,complete=False,stages=[])
    def write():(root/'workflow.json').write_text(json.dumps(progress,indent=2))
    def stage(label,mode,updates,seconds,parent=None,phase=None,reward='potential'):
        folder=root/label
        if (folder/'report.json').exists():
            report=json.loads((folder/'report.json').read_text())
            if not report.get('complete'):raise RuntimeError('Incomplete existing run: '+str(folder))
        else:
            command=[sys.executable,'-B',str(runner),'--mode',mode,'--seed',str(args.seed),
                     '--updates',str(updates),'--seconds',str(seconds),'--check-every','100',
                     '--output',str(folder),'--cache',str(args.cache.resolve()),'--budget','1','--reward',reward]
            if parent:command+=['--parent',str(parent)]
            if phase:command+=['--phase',phase]
            print('WORKFLOW START',label,flush=True)
            subprocess.run(command,check=True)
            report=json.loads((folder/'report.json').read_text())
        progress['stages'].append(dict(label=label,folder=str(folder),complete=report['complete'],stop=report['stop']))
        write()
        return folder,report
    write()
    folder,report=stage('aim','aim',1200,240)
    if report['stop']!='mastery':raise RuntimeError('Aim did not pass: no automatic promotion')
    parent=folder/'life1-potential-best.pt'
    folder,report=stage('budget','budget',800,600,parent)
    if report['final']['life1-potential']['dev']['life1']['cells'][0]['clear_rate']<.9:
        raise RuntimeError('One shooter did not pass: no automatic promotion')
    # Fixed strict path retained from the first experiment; health ablation still reported.
    parent=folder/'life1-potential-best.pt'
    folder,report=stage('two4','curriculum',1600,480,parent,'two4')
    parent=folder/'life1-potential-best.pt'
    if report['stop']!='mastery':
        folder,report=stage('two4-continue','curriculum',800,240,parent,'two4')
        parent=folder/'life1-potential-best.pt'
    if report['stop']!='mastery':raise RuntimeError('Two4 did not pass: no automatic promotion')
    folder,report=stage('reward','reward',400,400,parent)
    # First-seed outcome freezes sparse for independent replication.
    parent=folder/'life1-sparse-best.pt'
    own=report['final']['life1-sparse']['dev']['life1']['cells']
    if own[0]['clear_rate']<.8 or own[1]['clear_rate']<.7:
        folder,report=stage('two12-confirm','curriculum',800,300,parent,'two12','sparse')
        parent=folder/'life1-sparse-best.pt'
        if report['stop']!='mastery':raise RuntimeError('Two12 did not pass: no automatic promotion')
    folder,report=stage('chasers','curriculum',1000,400,parent,'chasers','sparse')
    parent=folder/'life1-sparse-best.pt'
    if report['stop']!='mastery':
        folder,report=stage('chasers-focus','curriculum',800,300,parent,'chasers-focus','sparse')
        parent=folder/'life1-sparse-best.pt'
    if report['stop']!='mastery':raise RuntimeError('Chasers did not pass: no automatic promotion')
    folder,report=stage('mixed','curriculum',1600,480,parent,'mixed','sparse')
    progress.update(complete=True,checkpoint=str(folder/'life1-sparse-best.pt'),mixed_mastery=report['stop']=='mastery')
    write();print('WORKFLOW COMPLETE',progress['checkpoint'],flush=True)

if __name__=='__main__':main()
