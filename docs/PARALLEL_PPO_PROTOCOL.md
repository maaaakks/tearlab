# Boucle PPO parallèle GPU — 8 novembre 2026

Le simulateur natif traitait déjà 1 024 salles en parallèle. Cette étape capture
aussi la collecte complète (32 décisions, réseau, échantillonnage, physique,
resets, récompense, écriture des transitions) dans un graphe CUDA. Un second
graphe couvre le réseau pendant le forward/backward PPO. Adam fusionné est
optionnel. Les statistiques différées constituent une ablation séparée.

Le modèle, les 45 actions, les 236 observations, le contexte de commande, les
récompenses, le mélange de scènes, les timeouts, le bootstrap, la normalisation
des avantages et les deux époques PPO restent ceux du pilote. KL contrôlé avant
chaque mise à jour ; arrêt à 0,02, gradients limités à 0,5. Aucune modification
de précision, TF32 désactivé. Pas de démonstration, de teacher ou de reward nouvelle.

## Mesures

- Politiques fraîches jetables ; aucun checkpoint historique chargé ou remplacé.
- Qualification avant chronométrage : même initialisation, scènes et graines,
  avec 400e décision forcée pour vérifier les vrais timeouts et leur bootstrap.
- Événements discrets, actions et trois RNG exacts. Flottants : atol 3e-4,
  rtol 1e-5 ; poids après Adam : atol 2e-5, rtol 1e-4, seuils fixés avant mesure.
- Une chauffe, puis répétitions entrelacées avec rotation de l'ordre des variantes.
- Synchronisation CUDA aux bornes de collecte et d'optimisation ; médiane du
  temps total réel, sans profiler. Capture et initialisation mesurées séparément.
- Lots : même horizon 32, minibatch 4 096, deux époques. Un plus grand nombre de
  salles change le lot PPO et le nombre d'updates par interaction : son débit
  ne prouve pas un meilleur apprentissage à durée égale.

## Contrats et limites

Les tests vérifient trois updates successives identiques au backend de référence,
la préservation de l'état/RNG à la capture, les vrais timeouts, un refus de
modification de configuration ou de stockage des poids, l'arrêt KL sans step
d'optimiseur puis la reprise, et la compatibilité du state_dict avec PilotPolicy.
Adam fusionné peut introduire un faible écart numérique ; cela ne garantit pas
des trajectoires d'apprentissage identiques à long terme.

Graphes limités aux formes fixes et au réseau PilotPolicy actuel. Les seuils de
contrôle restent côté CPU ; ils ne sont pas supprimés pour gagner du débit.
`load_state_dict` standard copie les poids en place. Remplacer leur stockage
nécessite une nouvelle capture. Même stream et paramètres physiques fixes.
Le backend original demeure accessible ; aucun lancement automatique d'Isaac.

## Lancer depuis IsaacRL

```powershell
# Diagnostic du lanceur, sans apprentissage long ni sauvegarde d'agent
.\.venv\Scripts\python.exe -B -m research.train_parallel --smoke --output data/parallel-smoke

# Pilote borné, même protocole de deux rewards que précédemment
.\.venv\Scripts\python.exe -B -m research.train_parallel --seconds 600 --envs 1024 --output data/parallel-pilot

# Ablation et mesures de la boucle complète
.\.venv\Scripts\python.exe -B -m research.benchmark_parallel --run-gpu --variants native_reference rollout_graph network_graph fused_adam deferred_metrics recommended --lanes 1024 --rounds 11 --output data/parallel-ablation.json
```

Le dossier de sortie doit être neuf pour un pilote ; son parent doit exister
pour un benchmark. Le cache natif peut être indiqué avec `--cache` au pilote.
Interrupteurs du lanceur : `--no-rollout-graph`, `--no-network-graph`,
`--no-fused-adam`, `--deferred-metrics`. Le backend recommandé active les trois
premières optimisations ; les métriques différées restent désactivées.
Checkpoints, sidecars et artefacts W&B hors ligne suivent le protocole initial.
