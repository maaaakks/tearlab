# TODO — Prochaine tâche : A4 (états de départ contrôlés)

Contexte : voir `docs/plans/hierarchical_agent_implementation.md` (roadmap complet).
Stages A1-A3 terminés. A4 est le dernier prérequis avant le Stage B.

## A4 — Commandes de debug dans le mod (états rares injectables)

Objectif : pouvoir démarrer un épisode directement dans une situation rare
(route du Beast, deal diabolique, salle secrète...) au lieu d'attendre qu'elle
arrive naturellement. Sans ça, les segments de route sont trop rares pour
être entraînés.

### Tâches

- [ ] A4.1. Protocol : nouveau message `{"command": "setup", "setup": {...}}`
      traité par `handleMessage` dans `mod/main.lua`, appliqué après le reset
      d'épisode (via une file "pending setup" consommée dans `onGameStart`).

- [ ] A4.2. `mod/game_control.lua` : fonction `applySetup(setup)` avec actions
      idempotentes, chacune optionnelle :
      - `stage` / `stage_type` : téléportation (`Game():SetStage`)
      - `room_type` : forcer le type de la salle courante si possible
      - `items` : liste d'IDs à donner (`itemPool:AddRoomBlacklist` non requis,
        `player:AddCollectible(id)`)
      - `cards` / `pills` : `player:AddCard(id)` / `player:AddPill(id)`
      - `bombs` / `keys` / `coins` : ressources exactes
      - `hearts` : configurer HP (containers + type)
      - `spawn_enemies` : liste `{type, variant, count}` positionnée autour du joueur
      - `spawn_pickup` : pickup spécifique (pièce/bombe/clef/cœur)
      - `grid_entity` : placer un rocher teinté/crâne marqué à une position

- [ ] A4.3. Python : champ `phase.setup` (dict) dans `config.py`, envoyé dans
      `to_game_settings()` / au premier `reset()` ; ou commande CLI
      `--setup <json>` pour `train.py` et `evaluate_live.py`.

- [ ] A4.4. Scénarios nommés réutilisables (fichiers `configs/setups/*.yaml`) :
      - `depths2_skull.yaml` : Depths II + bombe + crâne marqué spawné
      - `mom_polaroid.yaml` : Mom vaincue + Polaroid + The Fool en poche
      - `devil_deal.yaml` : salle du diable + item au piédestal + HP configurés
      - `secret_room.yaml` : mur candidat + bombe

- [ ] A4.5. Test manuel : lancer chaque scénario, vérifier l'état initial via
      le HUD et un dump JSON de l'état sérialisé.

### Critère de validation

Un épisode démarré avec `configs/setups/depths2_skull.yaml` montre, côté
Python : `run.stage == 6`, `special_rocks` contenant un `marked_skull`,
`num_bombs == 1`.
