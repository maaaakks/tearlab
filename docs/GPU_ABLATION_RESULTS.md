# IsaacRL — résultats des optimisations natives GPU

Mesures du 8 novembre 2026 sur la RTX 4070. Le graphe V1 est la référence : les gains ci-dessous sont **supplémentaires** à la première optimisation.

## Confirmation entrelacée

Autre graine (29), sept séries de 128 décisions, ordre des backends mélangé à chaque série, même historique de commandes. Les captures et le profiler sont exclus des temps. Les sorties possèdent leur mémoire. Une décision correspond à trois ticks natifs.

| Scénario | Salles | Graphe V1 | Fusion/thread | Fusion/groupe | Gain du meilleur |
|---|---:|---:|---:|---:|---:|
| shooters | 512 | 1.6280 ms | 0.1889 ms | 0.1808 ms | ×9.01 |
| shooters | 8,192 | 2.5160 ms | 0.3640 ms | 0.3683 ms | ×6.91 |
| mixed | 512 | 1.9461 ms | 0.2530 ms | 0.1907 ms | ×10.21 |
| mixed | 8,192 | 3.3011 ms | 0.4618 ms | 0.5066 ms | ×7.15 |

À 8 192 salles, la fusion par thread est légèrement meilleure dans cette confirmation. À 512 salles, la variante par groupe est préférable, surtout dans le mélange. La différence tireurs/thread-groupe à 8 192 est petite : ne pas en déduire une supériorité universelle.

Ces chiffres mesurent la simulation seule : aucun réseau, entraînement, transfert IPC ou vrai jeu Isaac dans la fenêtre de mesure. Les scénarios restent ceux du simulateur actuel.

## Apport de chaque proposition

Ablation à 8 192 salles mixtes, resets naturels, graine 481 : trois répétitions de 32 décisions. La colonne « seule » compare au contrôle par étapes. La dernière colonne mesure combien la combinaison **sans fusion** ralentit quand on retire cette proposition. Ces facteurs ne s’additionnent pas.

| Proposition | Seule, ms | Gain seule | Sans cette proposition, ms | Gain dans la combinaison |
|---|---:|---:|---:|---:|
| Mouvement | 4.5240 | ×1.08 | 1.0880 | ×1.53 |
| Tirs ennemis | 3.9876 | ×1.23 | 1.6955 | ×2.38 |
| Projectiles et contacts | 3.3675 | ×1.45 | 2.5225 | ×3.54 |
| Masquage terminal | 4.6636 | ×1.05 | 1.0640 | ×1.50 |
| Resets ciblés | 4.2069 | ×1.16 | 1.6362 | ×2.30 |
| Observations directes | 4.7321 | ×1.03 | 0.8625 | ×1.21 |

Contrôle par étapes : 4.8854 ms ; combinaison sans fusion : 0.7116 ms. Projectiles, tirs ennemis et resets ciblés sont les contributions les plus fortes dans la combinaison.

La fusion complète apporte encore un gain plus modeste après ces optimisations : dans ce cas, 0.7116 → 0.6593 ms. Les résultats par scénario/taille/reset sont dans `ablation.csv` et tous les échantillons bruts sont dans `ablation.json`.

## Fidélité et couverture

- Qualification dédiée : 18 variantes × 2 scénarios, lots de 33 salles, 32 décisions, resets fréquents ; aucun échec.
- Matrice : **144 cas réussis, zéro échec**, 18 variantes × 2 scénarios × 2 tailles de lot × 2 modes de reset. 20,054,016 décisions de salle comparées, hors chronométrages.
- Confirmation : qualification initiale de 64 décisions, puis 904 décisions par salle (8 de chauffe + 7 × 128). Sur les 896 décisions chronométrées, toutes les terminaisons sont comparées à chaque pas ; les états complets, sorties finales et RNG sont comparés après chaque série.
- Les événements discrets, PV et RNG correspondent exactement sur cette couverture.
- Écart maximal mesuré sur les champs d’état : 0 ; observations : 5.96046448e-08 ; récompenses : 0.
- Seuils inchangés : atol=3e-4, rtol=1e-5 pour les flottants ; tolérance zéro pour les champs discrets et le RNG. Aucun seuil n’a été relevé pour obtenir un succès.
- Ceci qualifie le remplacement numérique du simulateur existant, sans ajouter de certification de fidélité à toutes les mécaniques du jeu Isaac.

## Profilage et coûts restants

| Scénario, 8 192 salles | Noyaux/décision V1 | Noyaux/décision fusionnée |
|---|---:|---:|
| shooters | 1230 | 54 |
| mixed | 1450 | 64 |

La physique fusionnée est maintenant le premier noyau en temps GPU dans les traces. Dans le mélange, viennent ensuite le reset ciblé, les tirages aléatoires et les observations. Les tirages restent faits pour tous les emplacements afin de conserver le RNG historique. Le regroupement/copie des sorties reste aussi un coût réel.

Le backend natif conserve plus de buffers persistants : environ 42,5 Mio d’allocations PyTorch supplémentaires pour une instance mixte à 8 192 salles dans cette matrice, contre 16,0 Mio pour le graphe V1. Ce n’est pas une mesure de toute la VRAM du pilote.

Les traces sont prises hors chronométrage. Leurs durées détaillées sont sensibles à la surcharge du profiler ; les gains sont calculés avec les mesures sans profiler.

## Reproduire et utiliser

Les backends sont testés et prêts à être sélectionnés explicitement. Les lanceurs d’entraînement et les checkpoints existants restent inchangés. Pour les grands lots, choisir d’abord `fused_decision` ; pour les petits lots mixtes, comparer aussi `fused_decision_warp`.

Depuis la racine IsaacRL :

```powershell
.\.venv\Scripts\python.exe -B -m research.gpu_ablation --run-gpu --lanes 512 8192 --seeds 481 --steps 32 --repeats 3 --verify-steps 32 --output data/gpu-ablation/matrix.json
.\.venv\Scripts\python.exe -B -m research.confirm_gpu_simulator --run-gpu --profile --output data/gpu-ablation/confirmation.json
```

La deuxième commande entrelace les mesures de la référence et des deux fusions, avec une autre graine. Les empreintes des sources et des artefacts sont conservées dans les JSON. Les résultats de l’ablation courte varient avec l’ordre et l’état de la machine ; la confirmation entrelacée est la référence pour les chiffres principaux.

Fichiers : `ablation.json`, `ablation.csv`, `confirmation.json`, `summary.json`, `kernel-summary.json` et traces Chrome `*.trace.json`.
