# RAPID-1 : review et test de contrôle borné

Préenregistré le 12 septembre 2026, avant les entraînements de ce pilote.

Hypothèse : un encodeur d'entités qui conserve les petites informations, la
commande précédente et un apprentissage auxiliaire de conséquences factuelles
peuvent amorcer un contrôle utile sans reconstruire les pixels. Trois têtes Q
explorent avec une tête fixe par épisode. La prédiction ne sert pas encore à
planifier : elle doit d'abord améliorer le contrôle par rapport à son ablation.

Les trois variantes neuves partagent observations, récompense
`clear - contact - .002`, quatre pas de retour Double DQN, répétition des
commandes pendant quatre décisions, replay 50 % uniforme / 50 % événements
avec correction exacte de probabilité. Q possède une tête ; ensemble en possède
trois ; predictive ajoute à ensemble une GRU prédisant les déplacements,
variations de HP ennemis et blessures sur huit commandes réellement exécutées.
La perte auxiliaire vaut 0,2 fois régression Huber + BCE contact. Il n'existe
aucun bonus de visée, de positionnement, de proximité ou de dommage offensif.

Les commandes futures sont confinées à l'auxiliaire. Les séquences se coupent
au reset en conservant la dernière transition ; les suffixes manquants sont
masqués. Une limite de durée bootstrappe depuis le vrai dernier état.
L'observation structurée contient des HP instrumentés : ce n'est pas une
perception humaine ni une découverte des objets depuis les pixels.

Campagne : graines 1/2/3, 10 000 décisions au maximum par variante et graine,
20 environnements, batch 128, une mise à jour par pas vectorisé après 512
décisions. Budget mural d'entraînement : 90 s par run, hors sauvegarde et
évaluation finale. Une évaluation initiale/finale figée utilise 40 épisodes
appariés par graine ; aléatoire et commandes constantes 22/23 sont des contrôles.
Le modèle final seul est évalué, pas de sélection du meilleur checkpoint.

Le microcas contient un tireur stationnaire à un HP, angle initial aléatoire,
et sert à vérifier le démarrage. Le mélange reprend les cinq familles D1.
Chaque épisode est limité à 100 décisions (10 secondes), protocole différent
des 400 décisions possibles de D1. Une décision simule trois ticks ; tous les
appels, y compris les lanes d'évaluation sorties du score, sont comptés.
Les contrôles ne sont jamais injectés dans les trajectoires d'apprentissage.

Critères : publier clears initiaux/finaux, contacts, impacts, premières réussites
train, histogrammes d'actions, données/updates/relectures et temps. Un gain sur
micro seul ne démontre rien sur le mélange ou Isaac. Sans réussite rencontrée,
l'objectif reste non amorcé. Sans gain reproductible de predictive sur ensemble,
ne pas promouvoir l'auxiliaire. Sans gain sur une politique constante, ne pas
revendiquer une stratégie contextuelle. Pas de prolongation automatique.

Limites : comparaison à décisions et updates égaux, pas paramètres/FLOPs égaux ;
retours n-step hors politique sans correction de trajectoire ; deux images
seulement pour la politique, donc pas de mémoire longue des cooldowns ;
trois têtes partageant l'encodeur ne donnent pas une incertitude calibrée ;
les commandes factuelles futures peuvent dépendre d'observations futures.
Les graines nouvelles constituent une validation de développement, pas un
test de généralisation à des règles/compositions inédites. Aucune interaction
avec Isaac, aucun ancien poids ou replay ; calibration historique du simulateur
déclarée comme connaissance de construction du domaine.

Antériorités : [Direct Future Prediction, ICLR 2017](https://vladlen.info/papers/learning-to-act.pdf),
[Bootstrapped DQN, NeurIPS 2016](https://arxiv.org/abs/1602.04621).
Il s'agit d'une combinaison expérimentale adaptée au projet, pas d'une
nouveauté algorithmique établie ni d'une reproduction de ces travaux.
