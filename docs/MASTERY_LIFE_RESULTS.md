# Cas difficiles : examen commun des modèles figés

Pilote à une seed d’entraînement, deux cohortes de développement, 128 épisodes par cellule et cohorte.
Même objet d’examen, chargement des poids en place ; répétition du parent strictement identique sous les deux budgets.
Modèles latest à exposition égale : 9 830 400 décisions chacun. Aucun test final réservé consulté.

| Modèle | Examen | Scènes | Victoires | No-hit | Contacts | Timeouts |
|---|---|---|---:|---:|---:|---:|
| parent | budget1 | encerclement_4T | 26.7% | 26.7% | 0.59 | 14.2% |
| parent | budget1 | longue_distance_4T | 8.3% | 8.3% | 0.70 | 21.9% |
| parent | budget1 | difficiles_tous | 26.5% | 26.5% | 0.57 | 16.0% |
| parent | budget1 | retention | 87.4% | 87.4% | 0.13 | 0.0% |
| parent | budget3 | encerclement_4T | 59.6% | 24.2% | 0.84 | 37.1% |
| parent | budget3 | longue_distance_4T | 21.8% | 6.4% | 1.25 | 64.6% |
| parent | budget3 | difficiles_tous | 44.6% | 24.5% | 0.96 | 46.5% |
| parent | budget3 | retention | 99.6% | 86.8% | 0.15 | 0.1% |
| contacts1 | budget1 | encerclement_4T | 46.2% | 46.2% | 0.52 | 2.1% |
| contacts1 | budget1 | longue_distance_4T | 26.8% | 26.8% | 0.50 | 23.0% |
| contacts1 | budget1 | difficiles_tous | 42.1% | 42.1% | 0.44 | 14.2% |
| contacts1 | budget1 | retention | 88.6% | 88.6% | 0.11 | 0.0% |
| contacts1 | budget3 | encerclement_4T | 88.7% | 43.8% | 0.71 | 7.9% |
| contacts1 | budget3 | longue_distance_4T | 50.2% | 26.2% | 0.68 | 46.4% |
| contacts1 | budget3 | difficiles_tous | 67.8% | 41.5% | 0.58 | 29.5% |
| contacts1 | budget3 | retention | 99.5% | 87.8% | 0.14 | 0.0% |
| contacts3 | budget1 | encerclement_4T | 34.0% | 34.0% | 0.66 | 0.1% |
| contacts3 | budget1 | longue_distance_4T | 21.2% | 21.2% | 0.73 | 5.6% |
| contacts3 | budget1 | difficiles_tous | 32.4% | 32.4% | 0.64 | 3.4% |
| contacts3 | budget1 | retention | 73.3% | 73.3% | 0.27 | 0.0% |
| contacts3 | budget3 | encerclement_4T | 92.2% | 35.8% | 0.97 | 0.9% |
| contacts3 | budget3 | longue_distance_4T | 61.7% | 21.2% | 1.24 | 24.2% |
| contacts3 | budget3 | difficiles_tous | 75.3% | 33.5% | 1.01 | 14.7% |
| contacts3 | budget3 | retention | 99.1% | 73.6% | 0.32 | 0.1% |

## Encerclement 4 tireurs, examen no-hit par taille

| Modèle | Taille | Victoires no-hit |
|---|---|---:|
| parent | 440 | 31.2% |
| parent | 520 | 27.3% |
| parent | 640 | 21.5% |
| contacts1 | 440 | 43.8% |
| contacts1 | 520 | 47.3% |
| contacts1 | 640 | 47.7% |
| contacts3 | 440 | 33.6% |
| contacts3 | 520 | 29.7% |
| contacts3 | 640 | 38.7% |

## Interprétation

Le budget 3 facilite les victoires après contact mais ne démontre pas une meilleure esquive. La rétention no-hit doit rester séparée.
Les deux entraînements changent aussi la distribution et l’objectif par rapport au parent ; seul le contraste contacts1/contacts3 isole le budget.
Les premiers scores du pilote utilisent 32 épisodes/cellule ; le tableau ci-dessus remplace ces estimations par un examen commun plus large.
Suite proposée : tester un apprentissage de récupération puis une finition stricte, à exposition égale contre une continuation stricte ; ne pas le déclarer gagnant sans rétention.
Résultats du simulateur, pas taux de réussite mesurés dans Isaac. Budget de contacts expérimental, sans équivalence aux cœurs du jeu.
Poids en jeu inchangés. Checkpoints, SHA, sidecars, références W&B hors ligne et trajectoires de métriques par épisode conservés.