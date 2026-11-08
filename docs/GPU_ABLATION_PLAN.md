# IsaacRL — variantes natives prêtes pour les comparaisons GPU

État au 8 novembre 2026 : implémentation et contrôles CPU terminés ; compilation
CUDA terminée hors ligne. **Aucun noyau GPU, benchmark GPU, entraînement ni jeu
n'a été lancé pendant cette passe.** Les nouveaux backends sont expérimentaux
et ne remplacent pas les lanceurs existants.

## Ce qui est implémenté

| Proposition | Option / variante | Travail effectué |
|---|---|---|
| Mouvement fusionné | `movement` | Deux sous-étapes, friction, murs et vitesse intermédiaire dans un noyau |
| Tirs ennemis fusionnés | `attacks` | Compteurs, émissions, choix de slots et mouvement des poursuivants |
| Buffers et contacts directs | `projectiles` | Émission des larmes, contacts relatifs, dégâts, immunité et disparition sans tenseurs temporaires de paires |
| Fins d'épisode sans snapshots | `terminal_mask` | Bloque les écritures des salles terminées ; fallback tensoriel indépendant et masque natif |
| Resets ciblés | `sparse_reset` | Géométrie et recherche de candidats réservées aux salles à reset ; effets de bord globaux conservés |
| Observations directes | `direct_observation` | Écrit les 236 composantes directement, avec sorties contiguës réparties sur les threads CUDA |
| Fusion d'un tick | `fused_tick` | Réunit mouvement, attaques et contacts dans un lancement |
| Fusion des trois ticks | `fused_decision` | Une décision entière par thread, temporaires locaux, RNG prélevé en amont |
| Répartition par groupe | `fused_decision_warp` | Un groupe de 32 threads par salle ; projectiles et contacts répartis entre threads |

Les six premières propositions ont chacune leur interrupteur indépendant.
Les fusions ont des dépendances explicites et refusent les combinaisons invalides.
La disposition des tenseurs publics reste compatible avec le simulateur ; aucune
conversion générale du stockage des états n'est revendiquée.

Les primitives numériques C++ sont partagées entre la DLL CPU et les noyaux CUDA.
La version CUDA utilise le Driver API, sans extension C++ PyTorch, installation
de Triton ou modification des dépendances de l'environnement.

## Comparaisons préparées

Le banc propose **18 variantes** : graphe existant, contrôle avec orchestration
par étapes, six ajouts individuels, combinaison sans fusion, six retraits
individuels, fusion d'un tick, fusion d'une décision et version par groupe.

- **Ajout individuel** : proposition contre `staged_control`.
- **Retrait individuel** : `unfused_all` contre `without_<proposition>`.
- **Fusion** : `fused_tick` contre `unfused_all`, puis `fused_decision` contre
  `fused_tick`, sans changer les autres options.
- **Répartition** : `fused_decision_warp` contre `fused_decision`.
- **Gain total** : chaque variante contre le graphe V1 existant.

Les gains des options ne doivent pas être additionnés. Certaines se recouvrent
ou déplacent le goulot d'étranglement. Le contrôle par étapes évite d'attribuer à
un noyau un changement venant seulement de l'orchestration.

Le plan couvre les tireurs et les 13 compositions mixtes, plusieurs tailles de
lots et graines. Il distingue resets naturels, resets fréquents forcés par un
timeout court, et absence d'auto-reset. La densité effective des terminaisons
est enregistrée. L'ordre des variantes est mélangé de façon reproductible.

## Mesures produites lors du futur lancement GPU

Pour chaque cas :

- temps mural médian/p95 et échantillons bruts ;
- temps écoulé entre événements CUDA médian/p95 — ce n'est pas une mesure du
  seul calcul actif, car le flux peut aussi attendre ;
- décisions de salle/s et ticks de salle **émis**/s, trois ticks par décision,
  y compris les ticks où une salle déjà terminée n'avance plus ;
- coût de préparation/capture séparé ;
- allocations persistantes supplémentaires et pic d'allocations PyTorch,
  qui ne représentent pas toute la VRAM du pilote ;
- compte des terminaisons et leur fréquence ;
- empreintes des sources, du binaire natif et configurations ;
- écarts de fidélité par champ et état final du générateur aléatoire ;
- rapports de gain pour les comparaisons addition/retrait/fusion/répartition.

L'option `--profile` exporte aussi les traces Chrome et les tableaux des noyaux.
Le profiler s'exécute après les chronométrages, pas dans leur fenêtre.
Tous les chronométrages utilisent des sorties qui possèdent leur mémoire ;
le banc n'utilise pas `step_view` pour favoriser artificiellement une variante.

## Barrière de fidélité

Chaque cas est comparé à `StrictNativeShooterEnv` ou `StrictCombatSuiteEnv` à
la **taille de lot réellement mesurée**, avant le chronométrage. La comparaison
porte sur tous les tenseurs d'état de la référence, toutes les sorties et
tous les champs de diagnostic.

Les événements discrets, PV, compteurs et RNG doivent correspondre exactement.
Les tolérances flottantes par défaut sont `atol=3e-4`, `rtol=1e-5`, et les écarts
maximaux par champ sont conservés. Pour exiger une égalité numérique stricte,
utiliser `--atol 0 --rtol 0`. Aucun seuil n'est relevé automatiquement.

Une variante qui échoue est marquée `failed_not_promoted` et n'est pas
chronométrée. Les autres peuvent continuer ; la commande termine avec un code
d'erreur si un cas a échoué. Le fichier JSON est sauvegardé au fur et à mesure.

Ce protocole vérifie la fidélité au simulateur existant. Il ne certifie pas une
fidélité supplémentaire au jeu Isaac.

## Vérifications déjà exécutées, sans GPU

**9 tests CPU réussis ; 1 test GPU explicitement ignoré.** La matrice principale
contient 34 configurations (17 variantes natives ou tensorielle × 2 scénarios),
32 décisions par configuration sur 16 salles. Elle vérifie les sorties et états
à chaque décision, avec resets répétés.

Les événements discrets et états RNG correspondent exactement sur cette matrice.
L'écart flottant absolu maximal observé est `0.000152587890625`. Le contrôle
tensoriel et le masquage terminal seul restent comparés avec tolérance zéro.
Les noyaux natifs ne sont donc **pas annoncés identiques bit à bit**.

Les tests ciblent aussi : clear au premier tick, absence de kill après une
blessure terminale, buffers remplis et contacts ex aequo, reset vide/partiel/
complet, effets de bord sur la vitesse des poursuivants, quatre variantes de
résolution des murs, délais de commandes, mode diagnostic et conservation des
observations entre deux appels. Les mauvais types, tailles et configurations
du pointeur natif sont rejetés.

Pendant les tests CPU, l'initialisation CUDA est remplacée par une fonction qui
échoue immédiatement si elle est appelée. `cpu-validation.json` contient les
écarts par configuration et le champ `gpu_executed: false`.

## Ressources statiques des noyaux

Compilation hors ligne pour `sm_89`, adaptée à la RTX 4070 connue. Le rapport
ptxas annonce les registres suivants :

| Noyau | Registres |
|---|---:|
| Mouvement | 34 |
| Attaques | 41 |
| Projectiles, thread | 59 |
| Projectiles, groupe | 44 |
| Tick fusionné | 66 |
| Décision fusionnée, thread | 64 |
| Décision fusionnée, groupe | 113 |
| Observation | 18 |
| Reset | 54 |

Aucun spill n'est annoncé par cette compilation ; le reset utilise une pile
locale de 32 octets. Ce sont des données de compilation, **pas des mesures
d'occupation ni de vitesse GPU**. Le pilote charge le PTX et peut produire un
code différent lors de sa compilation JIT. La version par groupe doit donc
être mesurée avant d'être préférée à celle par thread.

## Commandes dans le dépôt IsaacRL

Générer seulement le plan, sans initialiser CUDA :

```powershell
.\.venv\Scripts\python.exe -B -m research.gpu_ablation --output data/gpu-ablation/plan.json
```

Refaire les contrôles CPU et compiler les noyaux hors ligne :

```powershell
.\.venv\Scripts\python.exe -B -m unittest research.test_staged_simulator.CPUQualification -v
.\.venv\Scripts\python.exe -B -m research.native_runtime --kind both --resources
```

**À exécuter seulement lorsque le GPU sera disponible**, premier passage court :

```powershell
.\.venv\Scripts\python.exe -B -m research.gpu_ablation --run-gpu --only graph_v1 staged_control movement attacks projectiles terminal_mask sparse_reset direct_observation --lanes 128 --seeds 29 --steps 16 --repeats 3 --verify-steps 32 --output data/gpu-ablation/smoke.json
```

Comparer ensuite les combinaisons et les deux répartitions :

```powershell
.\.venv\Scripts\python.exe -B -m research.gpu_ablation --run-gpu --only graph_v1 unfused_all fused_tick fused_decision fused_decision_warp --lanes 512 8192 --output data/gpu-ablation/fusion.json
```

Sans `--only`, le banc exécute toute la matrice. Ajouter `--profile` pour les
traces ; les fichiers portent la variante, le scénario, le mode de reset,
la taille du lot et la graine.

Le test GPU dédié, qui vérifie notamment un lot de 33 salles pour couvrir un
groupe partiel, exige aussi un opt-in :

```powershell
$env:RUN_ISAAC_GPU_TESTS='1'
.\.venv\Scripts\python.exe -B -m unittest research.test_staged_simulator.GPUQualification -v
Remove-Item Env:RUN_ISAAC_GPU_TESTS
```

Les GPU tests sont désactivés par défaut même si CUDA est installé.
Les sorties et le code sont prêts ; les colonnes de gains de cette deuxième
passe restent volontairement vides jusqu'aux mesures autorisées.
