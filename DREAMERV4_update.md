# DREAMERV4_update.md — Plan world model pour IsaacRL

*Créé le 2026-08-22. Roadmap R&D "apprentissage data-limité".*

## Objectif

Passer du model-free pur (PPO sur épisodes) à un agent qui apprend la
dynamique du jeu (world model) pour atteindre une efficience d'échantillons
type "humain avec budget limité". Motivation : à ~160 pas/s sur une seule
instance, chaque transition coûte cher ; un world model permet d'apprendre
hors-ligne ("en imagination") et de planifier court terme.

## État actuel (2026-08-22)

- PPO live + reward safety-first (`no_hit_bonus` 2.0, vitesse conditionnée au
  no-hit) — run `phase1a-4v1-speed12-h30s-safety-aux`
- Têtes auxiliaires C5 activées (`aux_coef: 0.05`) : room_type, danger,
  next_dest, reachable — supervision dense à chaque step
- Mode démos humaines en place : `configs/phase1a-human.yaml` +
  `python/record_human.py` → npz par épisode dans `data/human/`
- Baselines A/B n_steps dans `docs/baselines/2026-08-22-phase1a-4v1-nsteps512.md`

## Le plan en 3 stades

### Stade 1 — World model embryon : tête `next_state` supervisée

Une tête de plus sur le tronc partagé : prédire `état_suivant = f(état,
action)` et entraîner en MSE/cross-entropy à côté des heads C5. Le modèle ne
sert jamais à décider ; il force le tronc à encoder les conséquences des
actions (si je tire à gauche, le projectile va là).

- Coût : quelques heures. Risque : quasi nul (additif, checkpoint-compatible
  via `_adapt_state_dict`).
- **Sonde de décision** : si la loss de prédiction converge nettement sous le
  hasard, la dynamique est apprenable → go stade 2. Sinon, retravailler
  l'espace d'observation avant tout.

Cible de prédiction v1 : le canal "murs/entités" de la grille t+1 ou un
résumé (positions normalisées joueur + ennemis). Éviter de prédire la grille
complète dès le départ (stochastique, partiellement observable).

### Stade 2 — Latent dynamics model complet (DreamerV3-style RSSM)

Un RSSM compact sur observations structurées (grille + player + entities),
actions MultiDiscrete natives :

- encoder : état → latent z_t (déjà ≈ notre tronc)
- dynamique : p(z_{t+1} | h_t, z_t, a_t) récurrente
- décodeurs optionnels : reconstruction grille (léger), reward, value
- replay buffer + entraînement hors-ligne depuis nos données

Pourquoi cette famille et pas TD-MPC2 : actions discrètes natives (TD-MPC2 =
continu, actions discrètes "problème ouvert" selon ses auteurs), obs
structurées non-pixels (le tokenizer vidéo de Dreamer 4 est overkill), GPU
solo RTX 4070 suffisant pour un modèle S/M.

### Stade 3 — Imagination training

Entraîner la politique dans des rollouts imaginés par le modèle (recette
Dreamer), avec garde-fous : horizon court (5-15 steps), pénalité de
désaccord d'ensemble, mélange réel/imaginaire. Option Dreamer 4 moderne :
tokenizer + transformer causal si on passe un jour aux pixels.

## Pourquoi pas X (notes de recherche)

- **TD-MPC2** : excellent mais natif continu ; planning MPPI à chaque step ;
  adaptation MultiDiscrete = chantier ouvert.
- **Dreamer 4 officiel** : JAX (souffrance Windows), pensé flux vidéo à grande
  échelle ; repros PyTorch communautaires (lucidrains, nicklashansen) jeunes.
- **JEPA / V-JEPA 2** : paradigme intéressant (prédiction en espace latent sans
  reconstruction) mais échelle foundation-model, pas branchable sur notre env.

## Jalons / critères de succès

| Jalon | Critère |
|---|---|
| S1 | next-state loss < 50 % du baseline hasard après 100k steps |
| S2 | rollouts imaginés à 10 steps restent cohérents visuellement |
| S3 | politique imaginée ≥ politique PPO réel à budget de pas réel égal |

## Prochaines actions concrètes

1. Finir la collecte de démos humaines (≥ 50 épisodes propres)
2. Script d'imitation warm-start (behavior cloning) → `pretrain_bc.py`
3. Reprendre phase1a avec le checkpoint pré-entraîné (baseline comparatif)
4. Implémenter la tête S1 `next_state`
5. Décision S2 sur la base de la courbe de prédiction
