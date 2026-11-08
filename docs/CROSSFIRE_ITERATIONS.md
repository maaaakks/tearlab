# Décisions — tirs croisés, 9 novembre 2026

## Avant les entraînements

Protocole figé dans PROTOCOLE.md : récompense inchangée, 1600 updates par bras,
deux parents, tests fermés jusqu'au gel des poids. Adam neuf pour tous les bras,
y compris pooling, car ses états ne sont pas directement partageables avec des
listes de paramètres différentes. L'égalité porte sur l'exposition et les réglages
PPO, pas sur les FLOPs ni nécessairement sur le nombre de pas Adam (arrêt KL).

## Après le diagnostic, avant de lire les résultats des nouvelles branches

Diagnostic 128 épisodes/cellule, graine 950101. Quatre tireurs 12 PV, placements
contrôlés : 4,7 % / 0,4 % de victoires ; projectiles responsables de 83,2 % / 89,1 %
des épisodes, contacts 11,1 % / 10,4 %, timeouts 1,0 % / 0,2 %. C'est donc surtout
un problème de survie. L'exposition aux murs est très différente entre parents
(40 % / 83 % du temps dans ces cas) ; compter les morts près d'un mur sans ce
dénominateur serait trompeur. Les fichiers bruts gardent les 20 dernières décisions
diagnostiques de tous les épisodes, sans sélection.

Le réseau pooling progresse pendant le curriculum, avec rétention éligible,
mais cela n'établit pas encore la maîtrise du cas difficile 4T12. Les scores
agrégés incluent aussi 3T et des ennemis à quatre PV.

Contrôle supplémentaire décidé AVANT résultats risk/attention : si une branche
supplémentaire surpasse pooling sur le développement, comparer aussi une branche
MLP de capacité comparable, sans calcul cinématique ni attention. Le contrôle
`capacity` reprend exactement la branche `risk` avec trois entrées nulles au lieu
des features de rapprochement. Mêmes initialisations, exposition et scènes.
Cela permettra de distinguer l'effet des features de l'ajout de paramètres. Le
contrôle est adaptatif et ses coûts seront conservés. Aucun test final consulté.
