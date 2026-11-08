"""Choose recipe using development only, then hash every candidate before final audit."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).parent


def main():
    workflow=json.loads((ROOT/'finish-training.json').read_text());assert workflow['complete']
    models={};scores={};runs=[]
    for folder in [ROOT/'paired-seed129773',ROOT/'capacity-seed129773',ROOT/'paired-seed242579']:
        if not folder.exists():continue
        report=json.loads((folder/'report.json').read_text());assert report['complete']
        seed=report['config']['seed'];runs.append(str(folder/'report.json'))
        for variant,arm in report['arms'].items():
            assert arm['complete']
            scores.setdefault(variant,{})[str(seed)]=arm['best_metric']
            cp=folder/(variant+'-best.pt');digest=hashlib.sha256(cp.read_bytes()).hexdigest()
            assert json.loads(cp.with_suffix('.meta.json').read_text())['sha256']==digest
            models[f'{seed}-{variant}']=dict(checkpoint=str(cp.resolve()),sha256=digest,seed=seed,variant=variant)
    ranking={v:(min(s[0] for s in seeds.values()),sum(s[1] for s in seeds.values())/len(seeds),
                min(s[2] for s in seeds.values())) for v,seeds in scores.items() if len(seeds)==2}
    winner=max(ranking,key=ranking.get)
    assert winner==workflow['selected_variant']
    focus={}
    if workflow['focus_needed']:
        for seed in (129773,242579):
            folder=ROOT/f'focus-seed{seed}';report=json.loads((folder/'report.json').read_text());assert report['complete']
            metric=report['arms'][winner]['best_metric'];accepted=tuple(metric)>tuple(scores[winner][str(seed)])
            path=folder/(winner+'-best.pt') if accepted else Path(models[f'{seed}-{winner}']['checkpoint'])
            models[f'{seed}-focus']=dict(checkpoint=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seed=seed,variant=winner)
            focus[str(seed)]=dict(accepted=accepted,metric=metric,parent_metric=scores[winner][str(seed)],report=str(folder/'report.json'))
    parentroot=ROOT.parent/'isaac-generalization'
    for seed,path in [(129773,parentroot/'mixed-seed129773/life1-sparse-best.pt'),
                      (242579,parentroot/'replication-seed242579/mixed/life1-sparse-best.pt')]:
        models[f'{seed}-parent']=dict(checkpoint=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seed=seed,variant='parent')
    output=dict(frozen=True,selected_variant=winner,selected_model='focus' if focus else winner,focus=focus,
                selection='development only; minimum retention quality across seeds, then mean target score, then worst cell; focus retained only if development improves',
                ranking=ranking,reports=runs,models=models,final_test_read=False)
    path=ROOT/'selection.json';assert not path.exists();path.write_text(json.dumps(output,indent=2))
    print('FROZEN',winner,ranking)


if __name__=='__main__':main()
