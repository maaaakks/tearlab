# TODO — Generalist full-run agent

Ce fichier est la source de vérité pour la direction du projet.

## Vision

Construire **un seul agent généraliste** qui découvre et apprend le jeu dans
des runs naturelles : combat, esquive, navigation, exploration, portes, loot,
ressources, shops, deals, objets, boss et routes.

L'objectif n'est pas d'entraîner une succession de politiques spécialisées,
ni d'apprendre chaque feature dans une phase isolée. Le projet doit conserver :

- une seule politique principale ;
- un seul flux d'entraînement ;
- un seul historique de checkpoints ;
- un reward system cohérent à travers tout le jeu ;
- des expériences contenant plusieurs compétences en même temps.

Les anciennes phases spécialisées (`phase0*`, `phase1*`) deviennent des outils
de test, de benchmark et de diagnostic. Elles ne sont pas des prérequis à
l'entraînement généraliste.

La connaissance explicite du jeu reste utile pour les règles déterministes et
les objectifs extrêmement rares. Elle doit guider le même agent via son état et
un objectif courant, pas créer un agent différent par route ou par mécanique.

## Principes non négociables

1. **Apprentissage conjoint** : combat, navigation et loot apparaissent dans
   les mêmes runs et les mêmes batches.
2. **Épisode à l'échelle de la run** : nettoyer une salle donne un reward mais
   ne termine plus l'épisode.
3. **Gameplay naturel par défaut** : layouts, ennemis, drops et progression du
   jeu ne sont pas remplacés par des spawns artificiels dans l'entraînement
   principal.
4. **Une seule politique transférable** : pas de checkpoint séparé par skill.
5. **Reward borné et comparable** : une salle avec beaucoup de petits ennemis
   ne rapporte pas plus uniquement à cause de leur nombre ou de leurs HP.
6. **Mort toujours inacceptable** : les rewards de progression d'une run ne
   doivent pas rendre une mort rentable.
7. **Connaissance observable** : l'agent doit recevoir les informations utiles
   à une décision (prix, type de salle, item, ressources, route), sans devoir
   deviner une règle cachée inaccessible dans son observation.
8. **Scénarios contrôlés non obligatoires** : ils servent aux tests, aux
   évaluations et éventuellement à une petite fraction de resets rares dans le
   même entraînement généraliste.

## ✅ Fondations terminées

- [x] A1. State v2 : identité de run/salle, étage, type et forme de salle,
      portes et destinations, items au piédestal avec prix, pickups structurés,
      rochers spéciaux, inventaire étendu et historique des deals.
- [x] A2. Actions bombe, item actif et carte/pilule avec masques de ressources
      (`MultiDiscrete([9,9,2,2,2])`) et déclenchement one-shot côté Lua.
- [x] A3. RunMemory : graphe de salles 13×9, arêtes bidirectionnelles, salles
      visitées et observation `map` derrière `env.enable_map_memory`.
- [x] Reward room-level borné : dégâts plafonnés, reward de kill supprimé,
      clear, no-hit, perte de santé, mort, timeout et faible pression temporelle.
- [x] Checkpoints live complets, reprise, métriques JSONL et évaluation
      déterministe.

## 🔜 Priorité immédiate — B. Vraies runs multi-salles

Le verrou actuel est dans `mod/main.lua` : `room_cleared` termine encore
l'épisode et redémarre Isaac. Tant que ce comportement existe, l'agent ne peut
pas découvrir le jeu.

### B1. Séparer les événements de salle des fins d'épisode

- [x] Ajouter un mode `full_run` dans la configuration.
- [x] En mode `full_run`, une salle nettoyée émet un événement
      `room_cleared` mais `state.terminal` reste faux.
- [x] Ne plus appeler `resetEpisode()` après chaque salle.
- [x] Conserver le même `episode_id` pendant les changements de salle et
      d'étage.
- [x] Terminer un épisode seulement sur :
      - mort ;
      - victoire / objectif final atteint ;
      - abandon explicite ;
      - timeout global de sécurité.
- [x] Garder le comportement mono-salle pour les tests existants.

### B2. Cycle de vie salle, étage et run

- [ ] Émettre des événements distincts : `room_entered`, `room_cleared`,
      `boss_killed`, `floor_entered`, `floor_cleared`, `run_won`.
- [ ] Réinitialiser les compteurs room-level au changement de salle : budget
      de dégâts et tracker no-hit.
- [ ] Conserver les compteurs run-level : salles visitées, boss, inventaire,
      deals, ressources et route.
- [ ] Réinitialiser RunMemory uniquement au début d'une nouvelle run.
- [ ] Vérifier les transitions spéciales : boss room, trapdoor, alt path,
      Devil/Angel Room, Ascent et Home.

### B3. Configuration générale

- [x] Créer `configs/full_game.yaml` :
      - `full_run: true` ;
      - ennemis et layouts naturels ;
      - mémoire de carte activée ;
      - toutes les branches d'action activées ;
      - timeout global exprimé en temps de run, pas en durée de salle.
- [x] Démarrer depuis Basement I avec un seed naturel.
- [x] Ne pas injecter quatre Gapers dans cette configuration.
- [x] Conserver `phase1a.yaml` comme test rapide du contrôleur de combat.

### B4. Reward system généraliste

- [ ] Conserver un budget de progression borné par salle, indépendant du
      nombre et des HP des ennemis.
- [ ] Corriger le calcul des dégâts pour compter le coup final et représenter
      une vraie fraction de progression, pas seulement saturer le cap au
      premier tir.
- [ ] Donner le bonus de clear uniquement si Isaac est vivant ; la mort est
      prioritaire en cas de mort simultanée avec le dernier ennemi.
- [ ] Conserver le bonus no-hit room-level.
- [ ] Ajouter un signal distinct `deal_protected` basé sur les dégâts aux
      cœurs rouges au niveau de l'étage.
- [ ] Ajouter des rewards bornés pour boss, étage et victoire de run.
- [ ] Ajouter un petit bonus one-shot pour la première visite d'une salle ;
      aucun reward pour boucler entre deux salles déjà visitées.
- [ ] Ne pas donner un reward générique à tous les pickups : leur valeur dépend
      du prix, du besoin, du build et de la route.
- [ ] Garantir par tests qu'une mort reste négative même après la progression
      maximale autorisée.

### B5. Validation minimale en jeu

- [ ] Lancer une run naturelle depuis Basement I.
- [ ] Nettoyer la première salle sans changement d'`episode_id`.
- [ ] Traverser une porte et vérifier que RunMemory contient les deux salles.
- [ ] Ramasser un pickup et utiliser une bombe dans la même run.
- [ ] Changer d'étage sans reset Python/Lua.
- [ ] Mourir et vérifier qu'une nouvelle run commence avec une mémoire vide.
- [ ] Enregistrer et reprendre un checkpoint pendant une run multi-salles.

**Jalon B** : le même agent peut jouer plusieurs salles naturelles, combattre,
explorer et interagir sans que le trainer redémarre après chaque clear.

## C. Observation et modèle généralistes

Toutes les composantes ci-dessous sont intégrées à la même politique et
entraînées conjointement.

- [ ] C1. Activer `map` et les champs State v2 dans `full_game.yaml`.
- [ ] C2. Ajouter un encodeur d'ensemble pour les objets à identité : ennemis,
      items, pickups, prix, portes et objets interactifs.
- [ ] C3. Fusionner grille locale, joueur, carte, objets et inventaire dans un
      seul embedding de politique.
- [ ] C4. Ajouter une mémoire récurrente GRU/LSTM pour les informations
      partiellement observables, avec états cachés correctement stockés dans la
      collecte async PPO.
- [ ] C5. Ajouter des têtes auxiliaires partagées : prédire type de salle,
      accessibilité d'un objet, danger local et prochaine destination. Elles
      améliorent les représentations sans créer de politiques séparées.
- [ ] C6. Vérifier que les anciennes observations peuvent charger leur encodeur
      spatial par transfert sélectif, sans imposer un ancien checkpoint.

**Jalon C** : une seule politique consomme toute l'information locale et
run-level nécessaire aux décisions générales.

## D. Connaissance et décisions nécessaires

La connaissance est une aide structurée donnée au même agent. Les faits
déterministes ne doivent pas être redécouverts par hasard après des millions de
steps.

- [ ] D1. Générer `knowledge/items.json` depuis les données du jeu : IDs,
      qualité, tags, transformations et contraintes connues.
- [ ] D2. Décrire les prérequis de routes dans des DAGs : The Beast, Mega Satan
      et autres objectifs.
- [ ] D3. Construire un planner qui transforme l'état de run et l'objectif en
      un `goal vector` courant : explorer, boss, shop, préserver une bombe,
      chercher le crâne marqué, revenir à une porte, etc.
- [ ] D4. Ajouter un decision engine pour les contraintes dures et
      l'affordabilité : prix des shops, coût en cœurs, clés, bombes et conditions
      de route.
- [ ] D5. Exposer au réseau les utilités et contraintes comme features ; le
      réseau conserve la décision contextuelle lorsque plusieurs choix sont
      valides.
- [ ] D6. Gérer explicitement la préservation Angel/Devil, les Key Pieces, les
      statues d'anges, le crâne marqué, les salles secrètes et les objets à
      éviter selon le build.
- [ ] D7. Tester les règles indépendamment du réseau avec des états synthétiques
      déterministes.

**Jalon D** : l'agent généraliste reçoit des objectifs et des faits corrects
sans changer de politique ni de checkpoint.

## E. Entraînement unifié

- [ ] E1. Entraîner principalement sur des runs naturelles complètes.
- [ ] E2. Mélanger combat, navigation, loot et décisions dans chaque fenêtre
      d'entraînement PPO.
- [ ] E3. Maintenir une seule replay/rollout distribution et une seule série de
      checkpoints.
- [ ] E4. Ne pas imposer `phase0a -> phase0b -> phase1a -> phase1b` comme ordre
      d'entraînement.
- [ ] E5. Conserver les phases spécialisées uniquement comme évaluations de
      régression : combat, navigation, interaction et boss.
- [ ] E6. Si les situations rares sont absentes des données, autoriser une
      petite fraction de resets contrôlés mélangés aux runs naturelles, par
      exemple 10–15 %, sans créer de modèle spécialisé.
- [ ] E7. Réduire progressivement ces resets rares lorsque la couverture des
      runs naturelles augmente.
- [ ] E8. Évaluer `gamma: 0.995` ou un traitement hiérarchique des retours pour
      les conséquences multi-salles.
- [ ] E9. Normaliser les rewards/returns si la value loss devient instable sur
      les longues runs.

## F. Évaluation et garde-fous

- [ ] F1. Dashboard par checkpoint : reward, rooms cleared, floors reached,
      boss kills, run depth, no-hit, deal protection, exploration et ressources.
- [ ] F2. Évaluations déterministes sur seeds fixes pour comparer les versions.
- [ ] F3. Évaluations sur seeds inconnus pour détecter la mémorisation.
- [ ] F4. Mesurer séparément les capacités sans entraîner séparément : combat,
      navigation, loot, économie et décisions de route.
- [ ] F5. Détecter les comportements dégénérés : camping, boucles de portes,
      farming de pickups, gaspillage de bombes, refus de combattre et suicide
      après accumulation de reward.
- [ ] F6. Une modification n'est validée que si elle améliore les runs
      générales ou corrige une régression mesurée.

## Scénarios contrôlés : rôle limité

Les anciens travaux A4 d'injection d'états ne sont pas supprimés, mais changent
de rôle :

- tests reproductibles de l'observation et des actions ;
- évaluations de décisions rares ;
- debugging des routes ;
- petite source optionnelle de diversité dans l'entraînement généraliste.

Ils ne doivent pas devenir un curriculum obligatoire par feature, ni produire
des checkpoints indépendants pour le crâne marqué, les deals, les shops ou les
salles secrètes.

## Prochaine action concrète

**B1 et B3 sont terminés.** Prochaine étape : **B2** (événements distincts
`room_entered`/`room_cleared`/`boss_killed`/`floor_entered`/`floor_cleared`/
`run_won` + cycle de vie salle/étage/run), puis **B4** (reward généraliste
borné par salle, coup final compté, signaux boss/étage/victoire) avant la
validation en jeu **B5**.
