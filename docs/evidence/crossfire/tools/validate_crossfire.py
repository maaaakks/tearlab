"""CPU-only integrity check of sources, equal exposure and closed training factors."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import torch
import numpy as np
ROOT=Path(__file__).parent


def main():
    checkpoints=[];runs=[]
    with patch('torch.cuda._lazy_init',side_effect=AssertionError('Integrity initialized CUDA')):
        for folder in (ROOT/'paired-seed129773',ROOT/'capacity-seed129773',ROOT/'paired-seed242579',ROOT/'capacity-seed242579',ROOT/'focus-seed129773',ROOT/'focus-seed242579'):
            if not folder.exists():continue
            report=json.loads((folder/'report.json').read_text());assert report['complete']
            for name,digest in report['source_sha256'].items():
                assert hashlib.sha256((folder/'sources'/name).read_bytes()).hexdigest()==digest
            parent=Path(report['config']['parent'])
            if not parent.is_absolute():parent=ROOT.parents[1]/parent
            assert hashlib.sha256(parent.read_bytes()).hexdigest()==report['parent_sha256']
            exposures=set()
            for variant,arm in report['arms'].items():
                assert arm['complete'];exposures.add(arm['training_decisions'])
                for stage in arm['stages']:
                    for scene in stage['scenes']:
                        assert not scene['synchronized'] and scene['arrangement']!='grouped'
                        assert (scene['chasers'],scene['shooters']) not in {(2,1),(2,2),(1,3)}
                for label in ('best','latest'):
                    path=folder/f'{variant}-{label}.pt';cp=torch.load(path,map_location='cpu',weights_only=True)
                    meta=json.loads(path.with_suffix('.meta.json').read_text())
                    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
                    assert cp['sources']==report['source_sha256'];assert cp['parent']==report['parent_sha256']
                    assert all(torch.isfinite(x).all() for x in cp['state_dict'].values())
                    assert cp['updates']==meta['updates'];assert meta['wandb_artifact_reference'].startswith('offline:')
                    checkpoints.append(dict(path=str(path),sha256=meta['sha256'],updates=cp['updates'],decisions=meta['decisions']))
            assert len(exposures)==1
            runs.append(dict(path=str(folder),sources_verified=True,closed_factors_absent=True,equal_exposure=True,decisions_per_arm=exposures.pop()))
        selection=json.loads((ROOT/'selection.json').read_text());assert selection['frozen']
        for model in selection['models'].values():assert hashlib.sha256(Path(model['checkpoint']).read_bytes()).hexdigest()==model['sha256']
        audit=json.loads((ROOT/'final-audit/report.json').read_text());assert audit['complete']
        assert audit['selection_sha256']==hashlib.sha256((ROOT/'selection.json').read_bytes()).hexdigest()
        for model,row in audit['models'].items():
            raw=np.load(ROOT/'final-audit'/(model+'-episodes.npz'))
            for label,cohorts in row['suites'].items():
                assert len(cohorts)==2
                for seed,cohort in zip(audit['seeds'],cohorts):
                    values=raw[f'{label}-{seed}'].reshape(len(cohort['cells']),audit['per_cell'],7)
                    assert np.isin(values[:,:,0],[0,1]).all()
                    np.testing.assert_allclose(values[:,:,0].mean(1),[cell['clear_rate'] for cell in cohort['cells']],atol=1e-7)
                    for cell in cohort['cells']:assert 0<=cell['clear_rate']<=1 and 0<=cell['timeout_rate']<=1
    result=dict(passed=True,cpu_only=True,runs=runs,checkpoints=checkpoints,selection_and_final_hashes_verified=True)
    (ROOT/'integrity.json').write_text(json.dumps(result,indent=2));print('Verified',len(runs),'runs,',len(checkpoints),'checkpoints and frozen audit on CPU')


if __name__=='__main__':main()
