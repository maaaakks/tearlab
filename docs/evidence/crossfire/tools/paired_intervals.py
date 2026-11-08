"""Paired stratified bootstrap of evaluation episodes, NOT training-seed uncertainty."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent


def main():
    audit=json.loads((ROOT/'final-audit/report.json').read_text());assert audit['complete']
    selection=json.loads((ROOT/'selection.json').read_text());choice=selection.get('selected_model',selection['selected_variant'])
    rng=np.random.default_rng(982113);output={}
    for seed in (129773,242579):
        parent=np.load(ROOT/f'final-audit/{seed}-parent-episodes.npz')
        candidate=np.load(ROOT/f'final-audit/{seed}-{choice}-episodes.npz')
        cells=audit['models'][f'{seed}-parent']['suites']['controlled'][0]['cells']
        indexes=[i for i,cell in enumerate(cells) if cell['scene']['shooters']==4 and cell['scene']['shooter_hp']==12]
        diffs=[]
        for cohort in audit['seeds']:
            key=f'controlled-{cohort}'
            a=candidate[key].reshape(len(cells),audit['per_cell'],7)[indexes,:,0]
            b=parent[key].reshape(len(cells),audit['per_cell'],7)[indexes,:,0]
            diffs.extend(a-b)
        diffs=np.stack(diffs)
        samples=[]
        for _ in range(2000):
            ids=rng.integers(diffs.shape[1],size=diffs.shape)
            samples.append(float(np.take_along_axis(diffs,ids,axis=1).mean()))
        output[str(seed)]=dict(mean_difference_pp=float(diffs.mean()*100),
                              bootstrap_95_percent_pp=(np.quantile(samples,[.025,.975])*100).tolist(),
                              episodes=int(diffs.size),description='Paired within fixed scene/cohort; interval covers evaluation sampling only, not training variability')
    (ROOT/'paired-intervals.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))


if __name__=='__main__':main()
