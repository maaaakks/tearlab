import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'focus-paired'
def aggregate(cohorts):
    groups={
      'large_long':lambda c:c['case']['width']==640 and c['case']['scene']['pattern']=='long',
      'surround4':lambda c:c['case']['scene']['pattern']=='surround' and c['case']['scene']['shooters']==4,
      'retention':lambda c:c['case']['scene']['pattern']=='random',
    }
    result={}
    for budget in (1,3):
        selected=[r for r in cohorts if r['budget']==budget]
        for key,predicate in groups.items():
            cells=[[c for c in r['cells'] if predicate(c)] for r in selected]
            rates=[sum(row[i]['clear_rate'] for row in cells)/len(cells) for i in range(len(cells[0]))]
            flat=[c for row in cells for c in row]
            result[f'b{budget}/{key}']={k:sum(c[k] for c in flat)/len(flat) for k in ('clear_rate','nohit_rate','damage','timeout_rate')}
            result[f'b{budget}/{key}']['worst_cell_clear']=min(rates)
    return result
def main():
    train=json.loads((ROOT/'report.json').read_text());audit=json.loads((ROOT/'frozen-audit/report.json').read_text())
    assert train['complete'] and audit['complete'] and audit['repeatable']
    summary={k:aggregate(m['cohorts']) for k,m in audit['models'].items()}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Récupération puis no-hit contre strict, focus longue distance','',
      'Deux seeds d’optimisation depuis le parent strict du pilote. Chaque bras reçoit19 660 800 décisions nouvelles,200+400 updates.',
      '50% longue distance4T,20% autres scènes3/4T,30% rétention ;50% des environnements en640×360.',
      'Selected : choix de développement après finition stricte, ou parent conservé si le rang ne progresse pas.',
      'Audit après gel : deux nouvelles cohortes de layouts connus ×128 épisodes/cellule. Répétition parent exacte. Facteurs OOD réservés non consultés.','',
      '| Modèle selected | Grande salle longue distance no-hit | Pire placement | Encerclement4T no-hit | Rétention no-hit | Grande salle victoire3contacts |',
      '|---|---:|---:|---:|---:|---:|']
    for name,m in summary.items():
        if name!='parent' and not name.endswith('-selected'):continue
        a=m['b1/large_long'];lines.append(f"| {name} | {a['clear_rate']:.1%} | {a['worst_cell_clear']:.1%} | {m['b1/surround4']['clear_rate']:.1%} | {m['b1/retention']['clear_rate']:.1%} | {m['b3/large_long']['clear_rate']:.1%} |")
    lines+=['','## Contrôle latest à exposition égale','',
      '| Modèle latest | Longue distance no-hit | Rétention no-hit | Contacts grande salle, budget3 | Timeouts grande salle, budget3 |',
      '|---|---:|---:|---:|---:|']
    for name,m in summary.items():
        if not name.endswith('-latest'):continue
        lines.append(f"| {name} | {m['b1/large_long']['clear_rate']:.1%} | {m['b1/retention']['clear_rate']:.1%} | {m['b3/large_long']['damage']:.2f} | {m['b3/large_long']['timeout_rate']:.1%} |")
    lines+=['','## Coûts et limites','']
    lines+=['La récupération suivie de no-hit améliore la longue distance dans les deux continuations à budget final égal.',
      'Seule recovery-seed129773 produit un checkpoint sélectionné par le garde-fou de développement ; les autres choix reviennent au parent.',
      'Ne pas interpréter les lignes selected revenues au parent comme la performance de leur entraînement latest.',
      'La rétention moyenne masque des familles plus faibles ; les rangs sur32 épisodes/cellule restent bruyants. La sélection de cet essai reste figée malgré l’audit.',
      'Priorité suivante : stabiliser la rétention par famille et les placements longue distance, puis valider des facteurs réservés après gel de la recette.','']
    for name,arm in train['arms'].items():lines.append(f"- {name}: {arm['decisions']:,} décisions nouvelles, {arm['seconds']:.1f}s, choix={arm['selected']}, rang={arm['best_score']}")
    lines+=['','Deux seeds de continuation du même parent, pas deux lignées entièrement indépendantes. Simulateur à deux archétypes,4ennemis max et budget de contacts expérimental.',
      'Adam est réinitialisé dans les deux bras à la frontière ; la comparaison isole le curriculum complet avec cette procédure.',
      'Le parent strict avait déjà reçu9 830 400 décisions nouvelles au pilote, en plus de sa lignée antérieure. Les coûts ci-dessus sont supplémentaires.',
      'Aucun modèle installé ou promu en jeu. Les résultats de développement et d’audit sont conservés cellule par cellule ; aucune conclusion sur un transfert réel non mesuré.']
    (ROOT/'RESULTATS.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines))
if __name__=='__main__':main()
