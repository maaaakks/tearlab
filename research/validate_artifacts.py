"""CPU-only checkpoint/source/held-out integrity audit for completed runs."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import torch

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1)
    reports=[];models=[];by_hash={}
    with patch('torch.cuda._lazy_init',side_effect=AssertionError('Integrity audit initialized CUDA')):
        for report_path in args.root.rglob('report.json'):
            try:report=json.loads(report_path.read_text())
            except json.JSONDecodeError:continue
            if not report.get('complete') or 'planned_schedule' not in report:continue
            folder=report_path.parent
            seen=set()
            for phase in report['planned_schedule']:
                for scene in phase['scenes']:seen.add((scene['chasers'],scene['shooters']))
            assert not seen&{(2,1),(2,2),(1,3)},folder
            for relative,digest in report['source_sha256'].items():
                source=folder/'sources'/relative
                assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,source
            reports.append(dict(run=str(folder),complete=True,heldout_compositions_unseen=True,sources_verified=True))
            for path in folder.glob('*.pt'):
                meta=json.loads(path.with_suffix('.meta.json').read_text())
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                assert digest==meta['sha256'],path
                checkpoint=torch.load(path,map_location='cpu',weights_only=True)
                tensors=list(checkpoint['state_dict'].values())
                assert all(torch.isfinite(x).all() for x in tensors),path
                assert meta['wandb_artifact_reference'].startswith('offline:'),path
                row=dict(checkpoint=str(path),sha256=digest,finite=True,metadata_verified=True,
                         updates=checkpoint['updates'],decisions=meta['decisions'],
                         parent=checkpoint['parent'],offline_artifact=meta['wandb_artifact_reference'])
                models.append(row);by_hash[digest]=row
        for model in models:
            total=model['decisions'];parent=model['parent'];visited=set()
            while parent:
                assert parent in by_hash,('Parent not found',model['checkpoint'],parent)
                assert parent not in visited,'Cyclic lineage'
                visited.add(parent);row=by_hash[parent];total+=row['decisions'];parent=row['parent']
            model['lineage_training_decisions']=total
    result=dict(passed=True,cpu_only=True,reports=reports,checkpoints=models)
    args.output.write_text(json.dumps(result,indent=2))
    print('Verified',len(reports),'completed runs and',len(models),'checkpoints on CPU')

if __name__=='__main__':main()
