# Focus longue distance et récupération, 9 novembre 2026

Parent commun : dernier checkpoint strict du pilote (9 830 400 décisions),
plus performant en no-hit et rétention. Le modèle en jeu reste inchangé.

Deux seeds d'optimisation, 129773 et 291773, depuis le même parent. Comparer
strict→strict (200+400 updates, budget1) et récupération→strict (200 budget3,
puis400 budget1). Chaque bras : 19 660 800 décisions nouvelles. Même réseau,
reward potential, gamma1, hyperparamètres PPO et distribution. Adam et épisodes
redémarrent à la frontière dans les DEUX bras ; seuls les poids continuent.
Ce n'est pas une ablation de la conservation des moments Adam. Les contacts
ne représentent pas des cœurs Isaac calibrés.

50% des scènes =4 tireurs longue distance ;20% autres cas3/4 tireurs ;30%
rétention. Géométries440×240/520×280/640×360 réparties20/30/50%. Géométrie et
PV restent observés par la politique existante ; aucun kernel changé. Les
compositions réservées et anneaux inégaux restent hors entraînement.

Sélection de développement après la phase finale stricte uniquement : rétention
d'abord, puis pire cellule longue distance640×360, puis moyenne de ces cellules,
puis moyenne difficile. Parent conservé si aucun checkpoint n'améliore ce rang.
Deux cohortes fixes1000101/1000201,32 épisodes/cellule. Pas de promotion en jeu.

Après gel : examiner les checkpoints retenus et latest sur deux nouvelles
cohortes de layouts des mêmes facteurs de développement, sous budgets1et3.
Rapporter no-hit, victoires, contacts, timeouts et pire cellule, séparément pour
chaque seed. Les layouts ne prouvent pas le transfert à de nouvelles familles.
Les facteurs finaux réservés ne sont pas ouverts pendant l'itération.

Le smoke GPU ne mesure aucun gain : il vérifie seulement les deux phases,
la reconstruction des graphes et la sauvegarde des checkpoints/sidecars.
