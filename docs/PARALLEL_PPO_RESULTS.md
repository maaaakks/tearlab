# IsaacRL - parallélisation et boucle PPO complète

Mesures du 8 novembre 2026, RTX 4070, PyTorch 2.13.0+cu130.

**Implémenté et mesuré : ×2.70 supplémentaires sur collecte + PPO à
1 024 salles. Dans la même confirmation : ×18.54 face au tensoriel
initial, ×3.05 face au premier graphe.** Ces facteurs concernent
une mise à jour complète, contrairement aux anciens facteurs de simulation seule.
Ce diagnostic ne relance pas le pilote de dix minutes et ne mesure aucun progrès RL.

## Confirmation des quatre générations

Même modèle initial, scénarios, graine, horizon 32, deux époques, minibatch 4 096.
Qualification puis chauffe ; 11 répétitions entrelacées avec rotation de l'ordre.
La médiane porte sur le total collecte + optimisation, sans profiler.

| Version | Temps par mise à jour | Décisions/s |
|---|---:|---:|
| Tensoriel initial | 2185.1 ms | 14,996 |
| Premier graphe de simulation | 360.0 ms | 91,024 |
| Noyaux natifs, PPO précédent | 317.7 ms | 103,126 |
| Collecte + réseau PPO capturés, Adam fusionné | 117.9 ms | 278,040 |

Sur le backend actuel : collecte médiane 30.5 ms,
optimisation 83.8 ms.
Les médianes des parties ne s'additionnent pas nécessairement à la médiane du total.
Initialisation/capture : 0.512 s dans cette série,
contre 0.039 s pour le natif précédent.
Les économies amortissent cette capture en quelques updates.

## Apport indépendant des optimisations

Série distincte de neuf répétitions, même lot de 1 024. Les facteurs ne s'ajoutent pas.

| Variante | Temps total | Gain face au natif de cette série |
|---|---:|---:|
| native_reference | 283.3 ms | ×1.00 |
| control | 287.8 ms | ×0.98 |
| rollout_graph | 178.3 ms | ×1.59 |
| network_graph | 223.8 ms | ×1.27 |
| fused_adam | 278.6 ms | ×1.02 |
| deferred_metrics | 290.7 ms | ×0.97 |
| all | 105.4 ms | ×2.69 |

`control` est la nouvelle classe sans optimisation ; son coût correspond à la référence.
`rollout_graph` capture les 32 décisions, y compris réseau, action, resets et buffers.
`network_graph` capture le forward/backward du réseau pendant PPO.
`all` ajoute Adam fusionné et les métriques différées aux deux captures.
Les métriques différées seules ne démontrent pas de gain et restent désactivées par défaut.

Confirmation de la combinaison, 15 répétitions distinctes :

| Variante | Temps total |
|---|---:|
| native_reference | 287.3 ms |
| graphs_only | 108.6 ms |
| recommended | 101.0 ms |
| all | 102.7 ms |

Les deux graphes fournissent l'essentiel du gain. Adam fusionné reste optionnel et son
apport est plus petit ; aucune promesse de gain indépendant universel.

## Davantage de salles : optimum de débit mesuré à 4 096

Horizon et minibatch inchangés ; neuf répétitions pour les grands lots.
La ligne 1 024 vient de la confirmation ci-dessus, dans une autre série.

| Salles parallèles | Temps par update | Décisions/s |
|---|---:|---:|
| 1024 | 117.9 ms | 278,040 |
| 2048 | 179.9 ms | 364,362 |
| 4096 | 340.7 ms | 384,768 |
| 8192 | 770.7 ms | 340,148 |

8 192 salles n'améliorent plus le débit ; les gradients dominent. **1 024 reste le
défaut du lanceur** afin de conserver le lot du protocole RL précédent. 4 096 est
une option de débit à tester séparément pour son efficacité d'apprentissage.
Augmenter le lot modifie le nombre d'updates par interaction, même à deux époques.
Le pic d'allocation PyTorch de la série 8 192 est
745.3 Mio ; ce n'est pas toute la VRAM du processus/système.

## Fidélité et vérifications

- Toutes les qualifications publiées ont passé, avant leurs chronométrages.
- Chaque salle est mise à la décision 398 avant qualification : vrais timeouts,
  bootstrap sur observation terminale, contacts et autoresets sont exercés.
- Actions, événements discrets et RNG exacts. Dans la confirmation native/nouveau,
  transitions et états physiques ont un écart maximal nul après la collecte ;
  différence de poids après Adam fusionné : 5,96e-8 à 1 024 salles.
- À 8 192 : différence de poids 3,86e-6, dans les seuils préfixés. Pas de
  revendication d'identité bit à bit des entraînements longs après Adam fusionné.
- Neuf tests CPU/GPU passent : états/RNG à la capture, trois updates successives
  exactes sans Adam fusionné, bootstrap timeout, arrêt KL sans optimizer step,
  reprise après cet arrêt, garde sur stockage des poids/configuration/minibatch,
  compatibilité state_dict, plus les cinq contrats du pilote.
- Lanceur `research.train_parallel --smoke` exécuté avec succès, qualification
  numérique initiale comprise. Le backend retenu et les sources sont archivés.
- PyTorch émet des avertissements de stream AccumulateGrad pendant la capture ;
  aucun échec sur la suite finale et les séries qualifiées. Ils sont conservés
  dans les logs, sans être supprimés artificiellement.

Les benchmarks excluent évaluations, sauvegardes et démarrage. Ils ne permettent
pas de multiplier automatiquement le débit total d'un futur pilote par ces facteurs.
Aucun training long, aucun jeu Isaac lancé, aucun modèle historique remplacé.

## Ce que nous apprend RL_training

Lecture seule d'un autre projet local de comparaison (`RL_training`) : son PPO utilise 2 048 arènes,
un horizon 128 et des minibatches 16 384. Isaac parallélisait déjà 1 024 salles :
il ne manquait pas une architecture générale de simulation parallèle.
La principale lacune mesurée était l'orchestration Python autour du simulateur.

Le rapport `RAPPORT_DECOLLAGE_V2_002.md` décrit 21 495 808 décisions en 390,9 s et
100 % sur 1 000 essais réservés, au départ arrêté avec deux secondes de contrôle.
Il distingue 61,1 % au départ mobile et 0 % sur dix secondes. Ce n'est pas une
raison de minimiser le résultat : il montre l'intérêt d'une compétence précise,
d'un parent conservé, d'une récompense corrigée et d'examens séparés. Les tâches,
réseaux, horizons et epochs diffèrent : comparer directement les débits ne classe
pas les deux simulateurs ou les deux problèmes.

Prochain protocole Isaac conseillé, à comparer à durée ET interactions égales :

1. Isoler visée/tir, puis un tireur, puis deux tireurs avant les mélanges.
   Exiger réussite sur plusieurs graines et retenir les compétences précédentes.
2. Tester séparément un budget de vie calibré au jeu face au terminal au premier
   contact. Conserver le clear sans dégât comme examen indépendant, pas comme
   seule tâche. Mesurer tir, impacts utiles, dégâts reçus, timeouts et durée.
3. Revoir le signal d'apprentissage : la comparaison actuelle ne démontre pas
   de supériorité du bonus HP. Vérifier les récompenses sur trajectoires de
   réussite/échec/temporisation avant d'ajouter du shaping. Ne pas tout changer
   simultanément ; reward puis curriculum doivent rester des ablations lisibles.
4. Varier positions, distances et phases de tir ; réserver compositions et
   géométries entières. Les poursuivants seuls ne prouvent pas la maîtrise des
   tireurs. Un gros réseau supplémentaire n'est pas l'axe prioritaire démontré.

Référence externe consultée : [RocketGPU](https://github.com/thecosmicskye/RocketGPU),
simulation native de nombreux environnements sur GPU ; les benchmarks publiés
portent sur la physique et un autre matériel, pas sur notre PPO Isaac.

## Livrables

- `parallel_ppo.py`, `train_parallel.py`, `benchmark_parallel.py`, `test_parallel_ppo.py`.
- `PROTOCOLE.md` : options, commandes et invariants.
- `confirmation-1024.json` : comparaison principale ; `ablation-1024.json` et
  `combined-ablation.json` : contributions ; `scaling*.json` : nombre de salles.
- `summary.json`, `tests.log`, `launcher-smoke-final/report.json` et sources du diagnostic.

Code committé localement dans IsaacRL ; aucun push.
