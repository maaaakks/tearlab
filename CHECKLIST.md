# Binding of Isaac RL — Project Checklist

## Phase 0: Environment Setup & Integration
- [ ] Confirm Isaac version (Afterbirth+ vs Repentance) and OS
- [ ] Enable `--luadebug` in Steam launch options
- [ ] Run `scripts/install_mod.sh` to symlink mod into Isaac's mod directory
- [ ] Launch Isaac, enable IsaacRL mod from Mods menu
- [ ] Verify mod loads (check debug console for "IsaacRL: Mod loaded")
- [ ] Test luasocket availability — check console for "luasocket loaded" vs "falling back to file IPC"
- [ ] If luasocket fails: install luasocket into Isaac's Lua path, or confirm file IPC fallback works
- [ ] Write a simple Python test client that connects, receives one state JSON, prints it
- [ ] Verify state JSON structure matches what `isaac_env.py` expects
- [ ] Verify action injection works (send a move command, confirm player moves)
- [ ] Verify reset works (send reset command, confirm new run starts)
- [ ] Test full step loop: reset → receive state → send action → receive next state
- [ ] Measure tick throughput (states/sec) to establish baseline training speed

## Phase 1a: Single Stationary Enemy (Gaper)
- [ ] Configure mod to spawn 1 Gaper on episode start
- [ ] Run `train.py --config configs/phase1a.yaml` and confirm training loop starts
- [ ] Verify TensorBoard logs are written to `logs/`
- [ ] Monitor reward curves — agent should learn to shoot toward the enemy
- [ ] Check reward component breakdowns (damage_dealt should dominate early)
- [ ] Agent achieves >50% win rate (room cleared before death)
- [ ] Agent achieves >80% win rate — Phase 1a complete
- [ ] Save Phase 1a final checkpoint

## Phase 1b: Single Projectile Enemy (Monstro)
- [ ] Switch to `configs/phase1b.yaml` (enemy_type=20)
- [ ] Resume from Phase 1a checkpoint: `--resume checkpoints/phase1a_final`
- [ ] Monitor: agent should learn dodging behavior
- [ ] Agent achieves >80% win rate vs Monstro
- [ ] Save Phase 1b checkpoint

## Phase 1c: Multiple Mixed Enemies
- [ ] Switch to `configs/phase1c.yaml` (enemy_count=4)
- [ ] Resume from Phase 1b checkpoint
- [ ] Monitor: target prioritization, kiting behavior
- [ ] Agent achieves >70% win rate vs 4 enemies
- [ ] Save Phase 1c checkpoint

## Phase 2: Room Combat + Pickups
- [ ] Add pickup spawning to `game_control.lua`
- [ ] Add pickup collection detection to `reward.py`
- [ ] Verify pickup channel in observation grid works
- [ ] Agent learns to grab hearts when low HP
- [ ] Agent achieves >70% win rate with pickups
- [ ] Save Phase 2 checkpoint

## Phase 3: Single Floor Navigation
- [ ] Extend `state_serializer.lua` with minimap/floor-level observations
- [ ] Add room discovery and door navigation to observation space
- [ ] Disable forced enemy spawning — use natural room spawns
- [ ] Add room discovery reward, boss kill reward, floor completion reward
- [ ] Extend network architecture with floor-level context
- [ ] Agent can navigate between rooms and clear a full floor
- [ ] Save Phase 3 checkpoint

## Phase 4: Multi-Floor Runs
- [ ] Extend episode length for full runs
- [ ] Add trapdoor navigation
- [ ] Consider adding LSTM/attention for long-horizon memory
- [ ] Agent completes Basement → Mom run

## Infrastructure & Quality
- [ ] Set up Xvfb for headless training (no monitor required)
- [ ] Measure and optimize training throughput
- [ ] Add frame skip tuning (find sweet spot: speed vs reaction time)
- [ ] Set up periodic evaluation during training
- [ ] Add model recording/replay capability for debugging behavior
- [ ] Investigate parallel Isaac instances for faster rollout collection

## Known Risks to Track
- [ ] luasocket availability in Isaac's sandboxed Lua
- [ ] Game speed bottleneck (30 ticks/sec real-time)
- [ ] Reward hacking (monitor component breakdowns)
- [ ] Entity detection false positives (spikes, statues detected as enemies)
- [ ] Projectile encoding — may need velocity channels, not just position

## Optimisation native GPU — 8 novembre 2026

- [x] Six optimisations indépendantes, fusions tick/décision et variante par groupe de threads.
- [x] Contrôles CPU : 34 configurations, événements et RNG identiques, tolérances flottantes déclarées.
- [x] Compilation CPU/PTX/CUBIN hors ligne, ressources statiques et banc de 18 variantes.
- [x] Qualification CUDA : 144 cas réussis ; comparaison par proposition et confirmation entrelacée réalisées. Voir [les résultats](docs/GPU_ABLATION_RESULTS.md).
- [x] Backend natif choisi pour le pilote et boucle d'apprentissage complète mesurée ; voir les résultats PPO ci-dessous.

Voir [le protocole et les commandes](docs/GPU_ABLATION_PLAN.md). Aucun test GPU lancé pendant cette passe.

## Pilote PPO natif de dix minutes — 8 novembre 2026

- [x] Backend natif branché sur PPO ; contexte de commande et bootstrap des timeouts vérifiés.
- [x] Deux récompenses comparées à poids neufs sur 25 cellules et 11 familles, avec compositions/PV/géométrie réservés.
- [x] Budget partagé : 568 s, 31 162 368 transitions par bras ; audits indépendants et meilleurs checkpoints conservés.
- [x] Boucle complète mesurée : ×7,15 contre le tensoriel initial, ×1,17 contre le premier graphe, sur ce diagnostic.
- [ ] Résoudre les échecs face à plusieurs tireurs et répliquer sur plusieurs graines avant promotion.
- [ ] Valider dans Isaac ; ce pilote est uniquement simulé et instrumenté.

Résultats et limites : [PPO_GPU_10M_RESULTS.md](docs/PPO_GPU_10M_RESULTS.md).

## PPO parallèle complet — 8 novembre 2026

- [x] Capture de la collecte réseau/action/simulation/buffers sur GPU, sans changer la tâche.
- [x] Graphe forward/backward du réseau PPO et Adam fusionné, options indépendantes.
- [x] Contrats GPU : RNG/états préservés, timeouts, trois updates, arrêt KL et guards.
- [x] Lanceur optionnel `research.train_parallel` ; backend original conservé.
- [x] Gain collecte+PPO confirmé : ×2,70 supplémentaire, ×18,54 contre le tensoriel ; débit maximal testé à 4 096 salles, sans preuve de gain de qualité RL.

Protocole : [PARALLEL_PPO_PROTOCOL.md](docs/PARALLEL_PPO_PROTOCOL.md).

## Campagne de généralisation — 8 novembre 2026

- [x] Budget expérimental 1/3 contacts, observation santé/immunité et timeouts finis explicites.
- [x] Shaping potentiel avec annulation à tout terminal ; audit CPU et qualification GPU/tensorielle.
- [x] Collecte/PPO capturés, curriculum par scènes dynamiques, examens et artefacts hors ligne.
- [x] Première graine : visée, ablation du budget à exposition égale, deux tireurs et mélanges avec rétention validés.
- [x] Audits réservés sur deux apprentissages indépendants : compositions 48,5 % / 42,7 % sans dégât, contre ancien 4,4 % ; 3 PV 80,8 % / 72,2 %, contre 8,0 %. Limites et coûts consignés.

Protocole figé avant expériences : docs/GENERALIZATION_PROTOCOL.md.


## Tirs croisés — 2026-11-09

- [x] Diagnostic : projectiles dominants sur quatre tireurs, exposition aux murs mesurée, générateur commun aux effectifs qualifié avec secours explicite.
- [x] Comparaison sur deux graines à 52 428 800 transitions/bras : pooling retenu ; prolongation ciblée séparée à 39 321 600 transitions/graine, rewards inchangées.
- [x] Poids figés puis audits de placements/tirs synchronisés/géométrie : 4T12 contrôlé 4.4/0.0 % -> 45.6/36.9 % sans dégât ; tests/intégrité et limites dans CROSSFIRE_RESULTS.md. Aucun vrai jeu.


## Maîtrise et objectifs — 2026-11-09

- [ ] Comparer objectif ancien / victoire non actualisée / potentiel conservatif, à exposition égale et deux parents.
- [ ] Traiter encerclements et grandes salles avec géométries natives simultanées, rétention et score du pire groupe.
- [ ] Auditer les modèles figés sur nouveaux facteurs réservés, documenter les plateaux et les limites.


## Essai réel manuel — 2026-11-09

- [x] Préparer meilleur checkpoint figé, projection 238 entrées et inférence CPU à vitesse normale, touche C et arrêts de contrôle.
- [x] Remplacer les anciens liens IsaacRL cassés par une copie preview isolée, sauvegarder options/saves et fournir restauration/lanceur.
- [ ] Validation en jeu par le joueur : contrôle réel, visée/esquive et écarts entre IA vanilla et simulateur. Aucun combat lancé automatiquement.


## Démonstration automatique — 2026-11-09

- [x] Préparer le démarrage automatique et onze scénarios en boucle ; premier contact = échec, cœurs protégés, arrêt C/déconnexion/retard.
- [x] Valider la syntaxe Lua et la boucle avec API simulées (attente de partie, changement de scénario, contact et arrêt).
- [ ] Confirmer le démarrage et les combats dans le vrai jeu après rechargement par le joueur.


## Correction injection preview — 2026-11-09

- [x] Initialiser la table ButtonAction du vrai injecteur avant les callbacks : les actions étaient reçues mais ignorées.
- [x] Reproduire le défaut puis vérifier déplacement diagonal, tir et retour manuel avec le vrai injecteur et les API de jeu simulées (logs avant/après).
- [ ] Confirmer mouvement et tir dans le jeu après son rechargement par le joueur.


## Nettoyage loot preview — 2026-11-09

- [x] Nettoyer tous les pickups au reset (consommables, coffres, trinkets et piédestaux) et supprimer les nouveaux pickups pendant la démo automatique.
- [x] Vérifier avec API simulées le nettoyage du loot existant, la suppression des drops et la préservation des pickups en mode manuel.
- [ ] Confirmer visuellement le nettoyage après rechargement du jeu par le joueur.


## Pilote cas difficiles / budget de vie — 2026-11-09

- [x] Préparer comparaison à exposition égale, budget 1 vs 3 contacts, objectif potentiel conservatif identique et examens sous les deux budgets.
- [x] Valider les terminaux par contact, la collecte native et les graphes GPU : 7 tests passent.
- [ ] Comparer les résultats sur les cas difficiles et la rétention ; confirmer toute amélioration sur une seed indépendante.


## Résultats pilote budgets de contacts — 2026-11-09

- [x] Exécuter deux bras à 9 830 400 décisions chacun et auditer les modèles latest figés sur les mêmes examens, deux cohortes ×128 épisodes/cellule, répétabilité exacte du parent.
- [x] Distinguer esquive et récupération : strict meilleur en no-hit/rétention, trois contacts meilleur en victoires avec dégâts.
- [ ] Répliquer sur une seed indépendante ; traiter la longue distance en grande salle puis tester récupération suivie de finition stricte à exposition égale. Aucun candidat promu en jeu.


## Focus récupération puis discipline — 2026-11-09

- [x] Préparer comparaison appariée sur deux seeds, récupération200 +no-hit400 contre strict600, parent strict commun, Adam réinitialisé dans les deux bras.
- [x] Vérifier distribution ciblée et smoke GPU des deux phases/bras ; modèle live inchangé.
- [ ] Mesurer les gains sur longue distance640×360 et la rétention, auditer les modèles figés et consigner les échecs.
