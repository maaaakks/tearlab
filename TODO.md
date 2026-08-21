# TODO — feature/hierarchical-agent

Roadmap complet : `docs/plans/hierarchical_agent_implementation.md`.

## ✅ Terminé

- [x] A1. State v2 : run/room identity, portes+destinations, items au
      piédestal+prix, pickups structurés, rochers spéciaux, inventaire étendu
- [x] A2. Actions bombe/item actif/carte-pilule (`MultiDiscrete([9,9,2,2,2])`)
      avec masques de ressources ; one-shot côté Lua
- [x] A3. RunMemory : graphe de salles 13×9, arêtes bidirectionnelles,
      obs `map` (4 canaux) derrière `env.enable_map_memory`
- [x] Reward review : cap dégâts +0.25, kills supprimé, clear +2.0,
      no-hit +0.5 conditionné, -0.5/demi-cœur, mort/timeout -5/-5,
      time_penalty 0

## 🔜 Prochaine tâche : A4 (états de départ contrôlés)

Objectif : démarrer un épisode directement dans une situation rare (route du
Beast, deal diabolique, salle secrète...) au lieu d'attendre qu'elle arrive.
Sans ça, les segments de route sont trop rares pour être entraînés.

- [ ] A4.1. Protocol : message `{"command": "setup", "setup": {...}}` dans
      `handleMessage` (`mod/main.lua`), appliqué après reset d'épisode via une
      file "pending setup" consommée dans `onGameStart`.
- [ ] A4.2. `mod/game_control.lua` : `applySetup(setup)` idempotent, champs
      optionnels :
      - `stage` / `stage_type` : téléportation (`Game():SetStage`)
      - `items` : liste d'IDs (`player:AddCollectible(id)`)
      - `cards` / `pills` : `player:AddCard(id)` / `player:AddPill(id)`
      - `bombs` / `keys` / `coins` / `hearts` : ressources exactes
      - `spawn_enemies` : `{type, variant, count}` autour du joueur
      - `spawn_pickup` : pickup spécifique
      - `grid_entity` : rocher teinté / crâne marqué à position donnée
- [ ] A4.3. Python : champ `phase.setup` (dict) dans `config.py`, envoyé au
      premier `reset()` ; option CLI `--setup <json>` pour train/evaluate.
- [ ] A4.4. Scénarios nommés dans `configs/setups/*.yaml` :
      - `depths2_skull.yaml` : Depths II + 1 bombe + crâne marqué spawné
      - `mom_polaroid.yaml` : Polaroid possédée + The Fool en poche
      - `devil_deal.yaml` : salle diable + item piédestal + HP configurés
      - `secret_room.yaml` : mur candidat + bombe
- [ ] A4.5. Test manuel par scénario : vérifier l'état initial (HUD + dump JSON).

**Critère de validation** : épisode lancé avec `depths2_skull.yaml` → côté
Python `run.stage == 6`, `special_rocks` contient un `marked_skull`,
`num_bombs == 1`.

## Stage B — Knowledge base + planner (~2–3 semaines)

- [ ] B1. `knowledge/items.json` généré depuis items.xml du jeu
      (ids, tags, quality) — 1–2 j
- [ ] B2. Route DAG `mega_satan.yaml` (compléter `the_beast.yaml`) — 0,5 j
- [ ] B3. Planner basé règles : RunMemory + route DAG → flux de Subgoals
      (`python/planner/goals.py` déjà en place) ; choix de porte, gestion
      clefs/bombes, retours arrière — 3–5 j
- [ ] B4. Decision engine squelette : contraintes dures + affordabilité
      deals/shops (schéma `knowledge/decision_rules/deals.yaml` prêt) — 1–2 j
- [ ] B5. Intégration bout-en-bout : planner branché sur le trainer live,
      sous-buts injectés dans l'obs (goal vector), tests en jeu — 2–3 j

→ Jalon : agent qui **suit une route** de façon autonome (sans optimisation
apprise).

## Stage C — Apprentissage des décisions (~4–6 semaines + training)

- [ ] C1. Slice verticale : rocher teinté détecter → naviguer → bomber — 2–3 j
- [ ] C2. Entity-set transformer pour objets identifiés (items, prix,
      ennemis) en parallèle du CNN grille — 3–5 j
- [ ] C3. LSTM récurrent dans le contrôleur ⚠️ BPTT + collecte async =
      séquences courtes ou états cachés stockés — 2–4 j
- [ ] C4. Behavior cloning du decision engine depuis un teacher scripté — 3–5 j
- [ ] C5. Recherche stratégique sur le graphe de route + value fn apprise
      (partie recherche, incertaine) — 5–8 j
- [ ] C6. Fine-tuning goal-conditionné par segment de route, puis curriculum
      full-floor ; shaping potential-based le long du DAG — 3–5 j +
      plusieurs semaines de wall-clock training

## Divers / suivi

- [ ] Corriger les 3 tests wall-collision pré-existants (cassés en amont,
      attendent pos_x/pos_y normalisés au lieu de positions monde)
- [ ] Évaluer gamma 0.99 → 0.995 quand les épisodes deviennent multi-salles
- [ ] VecNormalize sur les rewards quand les phases full-run arriveront
