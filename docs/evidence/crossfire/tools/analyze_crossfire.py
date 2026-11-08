"""Local numerical summaries; no checkpoint selection from final test scores."""
import csv
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent


def average(rows,key):return sum(r[key] for r in rows)/len(rows)


def main():
    baseline=json.loads((ROOT/'baseline-diagnostic/report.json').read_text())
    result={'baseline':{},'training':{}}
    for index,(key,model) in enumerate(baseline['models'].items()):
        controlled=model['suites']['controlled']['cells'];historical=model['suites']['historical']['cells']
        out={}
        for count in (3,4):
            rows=[r for r in controlled if r['scene']['shooters']==count and r['scene']['shooter_hp']==12]
            out[str(count)+'T12']={k:average(rows,k) for k in ['clear_rate','projectile_death_rate','contact_death_rate','timeout_rate','wall_exposure_rate']}
        out['historical_compositions']=average(historical[3:],'clear_rate')
        result['baseline'][str(index)]=out
    for seed in (129773,242579):
        path=ROOT/f'paired-seed{seed}/report.json'
        if not path.exists():continue
        report=json.loads(path.read_text());out={}
        arms=dict(report['arms'])
        extra=ROOT/f'capacity-seed{seed}/report.json'
        if extra.exists():arms.update(json.loads(extra.read_text())['arms'])
        for variant,arm in arms.items():
            out[variant]=dict(complete=arm.get('complete',False),parameters=arm['parameters'],baseline=[x['macro']['clear_rate'] for x in arm['baseline']],
                              checks=[dict(updates=x['updates'],metric=x['metric']) for x in arm['checks']],
                              elapsed_seconds=arm.get('elapsed_seconds'),training_seconds=arm.get('training_seconds'),
                              best_metric=arm.get('best_metric'),training_decisions=arm.get('training_decisions'))
        result['training'][str(seed)]=out
    final=ROOT/'final-audit/report.json'
    result['focus']={}
    for seed in (129773,242579):
        path=ROOT/f'focus-seed{seed}/report.json'
        if path.exists():result['focus'][str(seed)]=json.loads(path.read_text())
    records=[]
    if final.exists():
        audit=json.loads(final.read_text());result['final']={}
        for model,row in audit['models'].items():
            output={};result['final'][model]=output
            for suite,cohorts in row['suites'].items():
                rows=[cell for cohort in cohorts for cell in cohort['cells']]
                # Equal episodes/cell; intentionally weight cells, not family macros
                output[suite]=dict(clear_rate=average(rows,'clear_rate'),timeout_rate=average(rows,'timeout_rate'),
                                   episodes=sum(x['episodes'] for x in cohorts))
                hard=[r for r in rows if r['scene']['shooters']>=3 and r['scene']['shooter_hp']==12]
                if hard:
                    output[suite]['hard_clear_rate']=average(hard,'clear_rate')
                    output[suite]['hard_timeout_rate']=average(hard,'timeout_rate')
                if suite=='controlled':
                    for count in (3,4):
                        part=[r for r in hard if r['scene']['shooters']==count]
                        output[suite][str(count)+'T12']=average(part,'clear_rate')
                for cell in rows:records.append(dict(model=model,suite=suite,name=cell['name'],clear_rate=cell['clear_rate'],timeout_rate=cell['timeout_rate']))
            if 'historical' in row:
                rows=row['historical']['cells'];output['historical_3T']=rows[0]['clear_rate'];output['historical_4T']=rows[1]['clear_rate']
                output['historical_compositions']=average(rows[2:],'clear_rate')
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2))
    if records:
        with (ROOT/'cells.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
