# IsaacRL — maîtrise, objectifs et géométrie, 9 novembre 2026

Les scores précédents sont insuffisants. Cette campagne traite les deux échecs
observés : encerclement et inefficacité dans une grande salle. Aucun vrai jeu.

## Récompenses challengées avant entraînement

Le contre-exemple exact reward-audit.json prouve que gamma .995, mort -3.1,
timeout -2 et coût de temps -.002 ne maximisent pas strictement la victoire :
attendre 400 décisions vaut -.617, combattre avec 40% de victoires en 200 décisions
et 60% de morts en 100 vaut -.886. C'est un exemple construit, pas une trajectoire
mesurée de l'agent. Ne pas en déduire une preuve causale de tous les timeouts.

Comparer à même exposition depuis chaque parent figé :

- legacy : objectif précédent, gamma .995, inchangé.
- win : gamma 1, victoire +3, toute défaite -3 (mort/timeout), aucun coût de temps
  ni bonus de simple survie. Espérance =6*P(victoire)-3 à horizon fini.
- potential : même objectif win, plus 1.5*(Phi_suivant-Phi_actuel), Phi=fraction
  des PV retirés et Phi_terminal=0 pour TOUT terminal. Retour total inchangé,
  y compris une défaite après dégâts ; aucune récompense nette de farm.

Pas de récompense pour tirer, marcher, s'aligner ou rester loin d'un ennemi. La
comparaison legacy/win porte sur l'objectif complet, pas sur gamma seul. win vs
potential isole le shaping. Même réseau pooling, même LR/entropie/rollout ; Adam
neuf pour les trois. Horizon principal 400, premier dégât terminal conservé.

## Scènes et apprentissage

Groupes natifs séparés, même politique, concaténés dans chaque rollout GPU :
440x240, 520x280, 640x360. Aucun changement des kernels ni de la physique.
Ajouter encerclements réguliers (3/4 tireurs répartis autour du joueur au centre)
et tirs à longue distance depuis bords/coins, en conservant les scènes précédentes.
Les anciens holdouts compositionnels restent hors entraînement.

Trois paliers fixes 4/8/12 PV, 400/400/1000 updates (1800 total =58 982 400
transitions par bras). Examens développement séparés de l'entraînement, deux
cohortes fixes 1000101/1000201, 64 épisodes/cellule. Priorité au pire groupe dur,
avec rétention un/deux tireurs, poursuivants, mélanges ; pas de moyenne 4/12 PV
qui cache le cas 12 PV. Cible pratique : >=90% en moyenne et >=80% par groupe dur,
sur deux cohortes, rétention >=80% par famille en moyenne sur les tailles, et
>=65% pour toute cellule de rétention. Le pire groupe dur est moyenné sur les
deux cohortes avant classement. Ces seuils ne sont pas abaissés pour conclure.

Comparer une référence de contrôle analytique (esquive/alignement) pour vérifier
la faisabilité de certains cas ; ce n'est ni un optimum ni un teacher d'entraînement.
Si le budget finit sous les seuils, consigner le plateau et décider une seule
modification avant de continuer, toujours sans ouvrir le test final.

## Tests

Tout ancien audit déjà lu devient développement/régression, sans revendication
de nouveau holdout. Nouveaux facteurs finaux fermés jusqu'au gel : 560x320 et
720x400, anneaux à rayons inégaux, 10/14 PV, phases de tir inédites, compositions
toujours non entraînées. Décomposer les facteurs plutôt que tout changer dans
un seul examen opaque. Deux cohortes 1010101/1010201, 128 épisodes/cellule.
Conserver les scores du pire cas, timeouts, dégâts, tirs utiles, durée et coût.
Un gain sur une grande salle ne suffit pas si les petites salles régressent.

Après lecture du test final, aucun ajustement sur ce même test. Sources, parent,
SHA des poids et sidecars, W&B hors ligne. Les tests vérifient l'objectif des
rewards, les terminaux, resets partiels, parité des groupes natifs et collecte.
