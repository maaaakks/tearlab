# Décisions d'expérience consignées avant les runs suivants

## Après la visée et pendant l'ablation du budget

La pratique atteint 100 % sans dégât sur deux checks de développement ; parent
conservé à l'update 400. Le meilleur contrôle constant fait 10,9 % dans cette
pratique sans dégât ; le contrôle aléatoire 61,7 %. Ce dernier montre que le test
facile sert à amorcer la compétence, pas à démontrer le combat.

Sur l'ablation à exposition identique, à l'update 600, le bras 1-contact fait
96,5 % contre un tireur à douze PV ; le bras 3-contacts fait 65,2 % avec tolérance,
et 4,7 % sur son examen 1-contact. Le choix final attend le run complet. Cette
observation réfute l'hypothèse simple « autoriser des dégâts aide forcément ».

## Passage aux deux tireurs — règle avant lancement

Chaque run de curriculum traite UN palier ; aucune promotion forcée au palier
suivant lorsqu'un budget de temps expire. Le mélange garde des scénarios précédents.

- `two4` : 6 parts deux tireurs à 4 PV, 2 parts un tireur à 12 PV, 1 part pratique.
  Passage après deux examens à >=85 % sur deux tireurs 4 PV et >=80 % sur un 12 PV.
- `two12` : 8 parts deux tireurs 12 PV, 2 parts un 12 PV, 1 part deux tireurs 4 PV.
  Passage après deux examens à >=70 % sur deux 12 PV et >=80 % sur un 12 PV.
- `mixed` : familles 1P1T, 1P2T, 3P1T avec rétention un/deux tireurs et poursuivants.
  Passage après deux examens : >=70 % de moyenne des trois mélanges, >=50 % pour
  chacun, rétention >=80 % sur un tireur et >=60 % sur deux.

Les poids sont choisis sur qualité équilibrée (minimum des taux rapportés à ces
seuils), puis réussite de la tâche, sans dégât et durée. Les compositions réservées
ne sélectionnent jamais les poids.

La comparaison reward sparse/potentiel utilise exactement le même mélange que
`two12`, depuis le même parent et le même optimiseur initial, avec exposition égale.
Les checkpoints choisissent le score des deux tireurs, pas celui d'un tireur.
