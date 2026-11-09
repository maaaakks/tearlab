# Récupération puis no-hit contre strict, focus longue distance

Deux seeds d’optimisation depuis le parent strict du pilote. Chaque bras reçoit19 660 800 décisions nouvelles,200+400 updates.
50% longue distance4T,20% autres scènes3/4T,30% rétention ;50% des environnements en640×360.
Selected : choix de développement après finition stricte, ou parent conservé si le rang ne progresse pas.
Audit après gel : deux nouvelles cohortes de layouts connus ×128 épisodes/cellule. Répétition parent exacte. Facteurs OOD réservés non consultés.

| Modèle selected | Grande salle longue distance no-hit | Pire placement | Encerclement4T no-hit | Rétention no-hit | Grande salle victoire3contacts |
|---|---:|---:|---:|---:|---:|
| parent | 3.9% | 3.1% | 43.0% | 88.7% | 5.7% |
| strict-seed129773-selected | 3.9% | 3.1% | 43.0% | 88.7% | 5.7% |
| recovery-seed129773-selected | 23.4% | 22.7% | 55.6% | 89.1% | 49.0% |
| strict-seed291773-selected | 3.9% | 3.1% | 43.0% | 88.7% | 5.7% |
| recovery-seed291773-selected | 3.9% | 3.1% | 43.0% | 88.7% | 5.7% |

## Contrôle latest à exposition égale

| Modèle latest | Longue distance no-hit | Rétention no-hit | Contacts grande salle, budget3 | Timeouts grande salle, budget3 |
|---|---:|---:|---:|---:|
| strict-seed129773-latest | 4.3% | 88.6% | 0.98 | 74.5% |
| recovery-seed129773-latest | 24.3% | 89.2% | 0.94 | 44.4% |
| strict-seed291773-latest | 14.5% | 89.7% | 0.73 | 72.4% |
| recovery-seed291773-latest | 21.5% | 88.1% | 0.99 | 40.0% |

## Coûts et limites

La récupération suivie de no-hit améliore la longue distance dans les deux continuations à budget final égal.
Seule recovery-seed129773 produit un checkpoint sélectionné par le garde-fou de développement ; les autres choix reviennent au parent.
Ne pas interpréter les lignes selected revenues au parent comme la performance de leur entraînement latest.
La rétention moyenne masque des familles plus faibles ; les rangs sur32 épisodes/cellule restent bruyants. La sélection de cet essai reste figée malgré l’audit.
Priorité suivante : stabiliser la rétention par famille et les placements longue distance, puis valider des facteurs réservés après gel de la recette.

- strict-seed129773: 19,660,800 décisions nouvelles, 134.8s, choix=parent, rang=[0.9830729166666666, 0.015625, 0.041666666666666664, 0.421875]
- recovery-seed129773: 19,660,800 décisions nouvelles, 115.4s, choix=best, rang=[0.9895833333333333, 0.109375, 0.19270833333333334, 0.4822916666666667]
- strict-seed291773: 19,660,800 décisions nouvelles, 114.4s, choix=parent, rang=[0.9830729166666666, 0.015625, 0.041666666666666664, 0.421875]
- recovery-seed291773: 19,660,800 décisions nouvelles, 134.5s, choix=parent, rang=[0.9830729166666666, 0.015625, 0.041666666666666664, 0.421875]

Deux seeds de continuation du même parent, pas deux lignées entièrement indépendantes. Simulateur à deux archétypes,4ennemis max et budget de contacts expérimental.
Adam est réinitialisé dans les deux bras à la frontière ; la comparaison isole le curriculum complet avec cette procédure.
Le parent strict avait déjà reçu9 830 400 décisions nouvelles au pilote, en plus de sa lignée antérieure. Les coûts ci-dessus sont supplémentaires.
Aucun modèle installé ou promu en jeu. Les résultats de développement et d’audit sont conservés cellule par cellule ; aucune conclusion sur un transfert réel non mesuré.