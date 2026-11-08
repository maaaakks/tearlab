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

## Après la première graine, avant fin de la seconde et avant tout audit final

Pooling gagne sur le développement : cible moyenne 62,6 %, contre cinématique
58,0 % et attention 58,9 %. Le contrôle de capacité n'est donc pas déclenché sur
cette première graine. Quatre tireurs 12 PV restent à environ 30–44 % selon le
placement malgré la progression ; ce palier n'est pas maîtrisé.

Décision bornée : terminer d'abord la comparaison égale sur la seconde graine.
Si une branche supplémentaire gagne sur l'une des graines, exécuter le contrôle
de capacité manquant sur les deux avant de choisir la recette. Choix global sur
les deux développements uniquement. Puis, si la moyenne 4T12 est sous 70 % sur
au moins une graine, prolonger cette seule recette sur 4T12, mêmes quatre
placements et six tâches de rétention, 1200 updates/graine, checks tous les 200.
Budget additionnel 39 321 600 transitions/graine. Adam à nouveau neuf comme dans
le lanceur ; comparaison d'architectures inchangée, cette prolongation est
rapportée séparément. Garder le parent si le meilleur développement prolongé
ne l'améliore pas. Aucun changement de reward, scène réservée ni seuil pour
obtenir artificiellement une promotion. Tests finaux encore fermés.

## Poids figés, audit terminé

Sur les deux graines, pooling conserve le meilleur développement moyen : 61,23 %
contre cinématique 58,56 % et attention 58,24 %. Aucune branche ne gagne sa graine,
donc le contrôle de capacité préparé n'est pas déclenché. Le focus progresse de
62,6 ->67,3 % et 59,9 ->64,8 % sur les cibles de développement, rétention éligible.
Meilleurs poids locaux du focus : update 1000 /1200, pas automatiquement les derniers.

selection.json est écrit avant toute lecture du test final. Audits fixés : 4T12
contrôlé 45,6/36,9 % contre parents 4,4/0,0 % ; anciennes compositions 74,2/62,8 %
contre 49,1/42,7 %. Facteurs groupés et synchronisés progressent sur les deux.
La généralisation géométrique reste instable : 54,9/11,0 % et 3,1/49,9 % de
timeouts sur 3/4 tireurs 12 PV. Placement historique quatre tireurs : 24,2/4,7 %.
Ces échecs sont conservés ; aucun entraînement supplémentaire après leur lecture.

## Validation et incident d'isolation des tests

La première suite combinée passe 26 tests puis rencontre une erreur mémoire CUDA
sur les nouveaux tests GPU ; les neuf nouveaux tests passent isolément. Ajouter
un nettoyage explicite des graphes/pools avant cette suite et entre variantes,
comme le fait déjà le lanceur entre modèles, permet deux exécutions combinées
normales de 28/28. Le journal d'échec reste archivé. Cela corrige l'isolation des
tests, sans modifier les poids, la physique ou les audits. L'intégrité CPU vérifie
quatre runs, 16 checkpoints et la concordance des résultats individuels/rapports.
