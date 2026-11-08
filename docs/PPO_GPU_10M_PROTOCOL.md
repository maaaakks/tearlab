# Pilote PPO — budget partagé de dix minutes

Protocole fixé avant le lancement. Aucune interaction avec Isaac ; aucune
réutilisation d'anciens poids ou de démonstrations. Une graine d'apprentissage :
ce pilote cherche un signal de progression, pas une conclusion statistique générale.

Deux agents démarrent des mêmes poids neufs et voient le même nombre de
transitions. Leur PPO, réseau, mélange de scènes et graines sont identiques ;
seule la récompense diffère. Les modèles sont mis à jour alternativement.

## Récompenses mises à l'épreuve

- Tâche seule : `clear sans dégât - contact - 0.001 par décision`.
- Signal borné : même récompense, plus `0.1 × HP réellement retirés / HP initiaux`.

Le second signal vaut au maximum 0.1 par épisode, quelle que soit la quantité
de PV. Aucun bonus de kill, de visée, de position, ni pénalité spécifique aux
murs. Les récompenses historiques pouvaient donner 2.4 pour les seuls impacts
contre quatre tireurs à douze PV, contre 1 pour la réussite : cette domination
est supprimée. Le moteur de simulation est identique dans les deux bras.

Le bonus borné peut accélérer l'apprentissage offensif mais aussi concentrer
l'agent sur des gains partiels. Il est donc jugé sur des métriques indépendantes
de la récompense : clears sans dégât, dégâts subis, fraction de PV retirés et
temps de clear. Les composantes de reward et l'entropie sont journalisées.

## Scénarios

Le 4v1 est une cellule parmi d'autres. Le mélange simultané couvre 1 à 4 ennemis,
poursuivants et tireurs. Chaque famille reçoit le même poids d'apprentissage,
indépendamment du nombre de variantes de PV. Les poursuivants ont 2/4/8 PV ;
les tireurs conservent les 12 PV natifs du laboratoire. Les états initiaux,
cadences et vitesses changent à chaque reset. Pas de curriculum séquentiel.

Trois familles de compositions sont totalement exclues de l'apprentissage :
2 poursuivants + 1 tireur ; 2 poursuivants + 2 tireurs ; 1 poursuivant + 3 tireurs.
Des tests distincts réservent aussi les poursuivants à 12 PV et une salle vide
plus grande (640×360 contre 520×280). Ce ne sont pas des layouts natifs Isaac
avec obstacles ; aucune généralisation à ceux-ci ne sera revendiquée.

Les validations périodiques utilisent des graines réservées mais des familles
entraînées. Elles seules choisissent les checkpoints. L'audit ID a une autre
graine ; les tests de composition, PV et géométrie ne sont consultés qu'avant
et après le pilote, sans influencer la sélection.

## Algorithme et budget

PPO existant, deux époques, clip 0.2, Adam 3e-4, entropie 0.01, gamma 0.995,
GAE lambda 0.95, arrêt des époques si KL >0.02, gradient limité à 0.5.
Le réseau conserve les encodeurs d'entités et le tronc du laboratoire, avec
46 entrées supplémentaires (commande précédente ou début d'épisode) pour
représenter le tick de commande retardée. Les observations restent instrumentées :
positions, vitesses, PV et préparation de tir, pas des images RGB.

Le bootstrap d'un timeout utilise sa vraie dernière observation ; un reset ne
se propage jamais dans les retours de l'épisode précédent. Un premier pré-vol
compare le wrapper natif à la référence et chronomètre collecte + optimisation
sur des copies jetables des modèles. Ces copies ne servent pas à l'apprentissage.

Le compteur du pilote inclut les évaluations initiales, l'apprentissage, les
contrôles périodiques et les audits finaux. Le début des nouvelles mises à jour
est arrêté 45 secondes avant le plafond de 600 secondes pour réserver les audits.
Préparation, compilation et pré-vol sont enregistrés séparément. Aucun
prolongement automatique au-delà du pilote.

## Critères d'interprétation

Comparer chaque agent à son initialisation sur exactement le même audit. Un
signal de progression en réussite demande au moins +5 points de clear macro et
des gains dans au moins trois familles ; les gains sur PV seuls restent des
progrès partiels. Rapporter les familles séparément : réussir surtout les scènes
à un ennemi ne suffit pas à qualifier le mélange. La généralisation réservée
est un résultat distinct, et la compétence dans le vrai Isaac n'est pas testée.

Les temps de collecte et d'optimisation sont séparés : le gain ×7 du moteur
ne doit pas être présenté comme un gain ×7 du training complet. Checkpoints,
optimiseurs, sources, métadonnées et artefacts W&B hors ligne sont conservés.
