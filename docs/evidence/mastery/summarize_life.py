"""Separate clear/nohit/contact metrics for the matched development pilot."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'life-pilot'
GROUPS={
 'encerclement_4T':lambda s:s['shooters']==4 and s['pattern']=='surround',
 'longue_distance_4T':lambda s:s['shooters']==4 and s['pattern']=='long',
 'difficiles_tous':lambda s:s['pattern']!='random',
 'retention':lambda s:s['pattern']=='random',
}
def aggregate(cohorts):
    result={}
    for budget in (1,3):
        for name,predicate in GROUPS.items():
            cells=[c for r in cohorts if r['budget']==budget for c in r['cells'] if predicate(c['case']['scene'])]
            result[f'budget{budget}/{name}']={k:sum(c[k] for c in cells)/len(cells) for k in ('clear_rate','nohit_rate','damage','timeout_rate','hp_removed','decisions')}
            result[f'budget{budget}/{name}']['worst_cell_clear']=min(c['clear_rate'] for c in cells)
    return result
def main():
    reports={str(b):json.loads((ROOT/f'contacts{b}/report.json').read_text()) for b in (1,3)}
    assert all(r['complete'] for r in reports.values())
    result=dict(baseline=aggregate(reports['1']['arms']['potential']['baseline']),arms={},note='Development pilot, one training seed, 32 episodes/cell/cohort; not a final generalization test')
    for budget,report in reports.items():
        arm=report['arms']['potential'];meta=json.loads((ROOT/f'contacts{budget}/potential-best.meta.json').read_text())
        best=next(c for c in arm['checks'] if c['updates']==meta['updates'])
        result['arms'][budget]=dict(seconds=arm['elapsed_seconds'],decisions=arm['decisions'],best_updates=meta['updates'],
                                  latest=aggregate(arm['checks'][-1]['cohorts']),best=aggregate(best['cohorts']),checks=[dict(updates=c['updates'],metrics=aggregate(c['cohorts'])) for c in arm['checks']])
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# Pilote cas difficiles : budget de contacts','',result['note'],'',
           '| Entraînement / checkpoint | Examen | Scènes | Victoires | No-hit | Contacts | Timeouts |',
           '|---|---|---|---:|---:|---:|---:|']
    rows=[('Parent',result['baseline'])]+[(f'{b} contact(s) / {label}',result['arms'][b][label]) for b in ('1','3') for label in ('latest','best')]
    for label,metrics in rows:
        for key,m in metrics.items():
            exam,group=key.split('/')
            lines.append(f"| {label} | {exam} | {group} | {m['clear_rate']:.1%} | {m['nohit_rate']:.1%} | {m['damage']:.2f} | {m['timeout_rate']:.1%} |")
    lines+=['','Même exposition : 9 830 400 décisions par bras. Budget de contacts expérimental ; aucune équivalence aux cœurs Isaac.',
            'Best sélectionné sur les examens stricts de développement, avec priorité à la rétention. Latest est le même budget final pour les deux bras.',
            'Aucun changement du modèle en jeu. Nouveaux facteurs finaux réservés non consultés.']
    (ROOT/'RESULTATS.md').write_text('\n'.join(lines),encoding='utf8')
    print('\n'.join(lines))
    frozen_path=ROOT/'frozen-audit/report.json'
    if frozen_path.exists():
        frozen=json.loads(frozen_path.read_text())
        assert frozen['complete'] and frozen['repeatable']
        audited={name:aggregate(m['cohorts']) for name,m in frozen['models'].items()}
        (ROOT/'frozen-summary.json').write_text(json.dumps(audited,indent=2))
        lines=['# Cas difficiles : examen commun des modèles figés','',
               'Pilote à une seed d’entraînement, deux cohortes de développement, 128 épisodes par cellule et cohorte.',
               'Même objet d’examen, chargement des poids en place ; répétition du parent strictement identique sous les deux budgets.',
               'Modèles latest à exposition égale : 9 830 400 décisions chacun. Aucun test final réservé consulté.','',
               '| Modèle | Examen | Scènes | Victoires | No-hit | Contacts | Timeouts |',
               '|---|---|---|---:|---:|---:|---:|']
        for name,metrics in audited.items():
            for key,m in metrics.items():
                exam,group=key.split('/')
                lines.append(f"| {name} | {exam} | {group} | {m['clear_rate']:.1%} | {m['nohit_rate']:.1%} | {m['damage']:.2f} | {m['timeout_rate']:.1%} |")
        lines+=['','## Encerclement 4 tireurs, examen no-hit par taille','',
                '| Modèle | Taille | Victoires no-hit |','|---|---|---:|']
        for name,model in frozen['models'].items():
            for width in (440,520,640):
                cells=[c for r in model['cohorts'] if r['budget']==1 for c in r['cells'] if c['case']['width']==width and GROUPS['encerclement_4T'](c['case']['scene'])]
                lines.append(f"| {name} | {width} | {sum(c['clear_rate'] for c in cells)/len(cells):.1%} |")
        lines+=['','## Interprétation','',
                'Le budget 3 facilite les victoires après contact mais ne démontre pas une meilleure esquive. La rétention no-hit doit rester séparée.',
                'Les deux entraînements changent aussi la distribution et l’objectif par rapport au parent ; seul le contraste contacts1/contacts3 isole le budget.',
                'Les premiers scores du pilote utilisent 32 épisodes/cellule ; le tableau ci-dessus remplace ces estimations par un examen commun plus large.',
                'Suite proposée : tester un apprentissage de récupération puis une finition stricte, à exposition égale contre une continuation stricte ; ne pas le déclarer gagnant sans rétention.',
                'Résultats du simulateur, pas taux de réussite mesurés dans Isaac. Budget de contacts expérimental, sans équivalence aux cœurs du jeu.',
                'Poids en jeu inchangés. Checkpoints, SHA, sidecars, références W&B hors ligne et trajectoires de métriques par épisode conservés.']
        (ROOT/'RESULTATS.md').write_text('\n'.join(lines),encoding='utf8')
        print('\nFROZEN AUDIT\n'+'\n'.join(lines))
if __name__=='__main__':main()
