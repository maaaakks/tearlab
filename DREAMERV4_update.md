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

| Jalon | Critère | Résultat (2026-08-22) |
|---|---|---|
| S1 | next-state loss < 50 % du baseline hasard après 100k steps | ❌ ratio plateau ~1.07 (>1 = hasard) |
| S2 | rollouts imaginés à 10 steps restent cohérents visuellement | ❌ imagination 3x pire que la persistance |
| S3 | politique imaginée ≥ politique PPO réel à budget de pas réel égal | non testé — bloqué par S2 |

## Bilan S1+S2 (2026-08-22) — verdict : goulot = espace d'observation

Chaîne complète construite et opérationnelle : tête `next_state` live
(`next_state_coef`), recorder agent/humain npz, RSSM compact offline
(`train_rssm_offline.py`), courbes (`plot_rssm.py`), rollouts imaginés
rendus (`visualize_rssm.py`). Le pipeline tourne de bout en bout en <1h.
C'est l'actif principal de cette itération : tout re-test post-enrichissement
d'obs coûtera une heure, pas une semaine.

### Ce qui a été mesuré

- **S1 live (tête next_state sur tronc PPO)** : v1 plateaud à ratio ≈ 1.0
  car l'action n'était pas en entrée (bug corrigé v2 : one-hot action + cibles
  delta). v2 avec coef 0.05 puis 0.5 : descente nette puis **plateau à
  ~1.06-1.10 dès ~70k** (run nscoef05, 290k steps). Ratio > 1 = sous le
  hasard mathématiquement impossible pour un prédicteur conditionnel convergé
  → le tronc PPO n'encode pas la dynamique, même gratuitement.
- **S2 offline (RSSM compact, 40k transitions agent+hummain)** :
  - posterior teacher-forced : player_recon 0.0145 → ~0.002 ✓ (la position
    est extractible), mais val_grid ≈ 0.0044 vs hasard 0.0048 (marge).
  - chemin stochastique : collapse structurel du posterior (kl ≡ 0) — z est
    redondant car h contient déjà les embeds. Trois calibrations essayées
    (free-nats 3→0, kl_weight 1.0→0.05) : le collapse persiste. Ce n'est pas
    un problème de poids, c'est architecturalement décoratif à cette échelle
    d'obs.
  - **imagination prior-only 10 steps : MSE pos joueur 0.078 vs persistance
    0.026 (8 défaites / 10 épisodes)**. Le jalon S2 est raté.

### Interprétation

Deux modèles d'architectures différentes (tronc PPO + tête linéaire ; RSSM
dédié offline) échouent tous les deux à prédire la dynamique depuis ces
observations. Le problème n'est ni l'algorithme ni le training : c'est
l'information disponible en entrée. Suspects identifiés, par ordre probable :

1. `frame_stack=1` : vitesses des ennemis non observables → leur position
   future est quasi indépendante de l'état courant.
2. Pas de tailles/rayons d'entités → impossible d'anticiper un contact.
3. Grille 13×7 : discrétisation grossière des positions.

### Décision

- **Ne PAS adopter TD-MPC2** : changer de famille d'algo n'ajoute pas
  d'information absente des observations. Notre RSSM déterministe-de-facto
  est l'équivalent local suffisant quand l'obs sera riche.
- **Ne PAS abandonner le world model** : le pipeline offline est bon marché,
  il re-juge en une heure toute amélioration d'obs.
- **Le PPO+curriculum reste la ligne principale** (il tourne déjà ; c'est lui
  qui a produit les vrais gains : win rate 2v1 0.65).
- **Prochaine itération = enrichissement d'obs**, dans cet ordre :
  1. colonne rayon dans les entités (cols réservées 13–15, zéro impact wire)
  2. vitesses ennemis (nouvelles cols ou frame_stack>1)
  3. re-test : `record_agent --stochastic` → `train_rssm_offline` →
     `visualize_rssm`, critère inchangé (battre la persistance à 10 steps).

## Prochaines actions concrètes


4. Implémenter la tête S1 `next_state`
5. Décision S2 sur la base de la courbe de prédiction
