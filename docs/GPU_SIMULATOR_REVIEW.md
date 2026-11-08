# IsaacRL — première optimisation GPU, 8 novembre 2026

Le nouveau backend accélère les simulateurs existants en réutilisant un graphe
CUDA préparé une fois. Il est installé séparément dans le dépôt IsaacRL, dans
`research/gpu_simulator.py`. Les lanceurs d'entraînement ne sont pas modifiés.

Le GPU a été utilisé après confirmation de sa disponibilité. Aucun entraînement,
changement de checkpoint, lancement d'Isaac ou modification du mod n'a été effectué.

## Résultats sur la RTX 4070

PyTorch 2.13.0+cu130, un thread CPU, float32. Référence : simulateurs V2 **avec
la correction existante des fins d'épisode**, `StrictNativeShooterEnv` et
`StrictCombatSuiteEnv`. Les deux chemins effectuent trois ticks par décision et
produisent observations, récompenses, terminaisons et diagnostics.

| Scénario | Salles simultanées | Référence, ms/décision de lot | Nouveau, ms/décision de lot | Gain | Nouveau, décisions de salle/s |
|---|---:|---:|---:|---:|---:|
| 4 tireurs | 128 | 43,761 | 1,637 | ×26,73 | 78 190 |
| 4 tireurs | 512 | 43,108 | 2,751 | ×15,67 | 186 117 |
| 4 tireurs | 2 048 | 42,349 | 3,864 | ×10,96 | 529 977 |
| 4 tireurs | 8 192 | 43,659 | 5,104 | ×8,55 | 1 604 964 |
| Mixtes, 13 compositions | 128 | 51,332 | 3,660 | ×14,02 | 34 970 |
| Mixtes, 13 compositions | 512 | 51,704 | 4,137 | ×12,50 | 123 749 |
| Mixtes, 13 compositions | 2 048 | 50,035 | 5,240 | ×9,55 | 390 814 |
| Mixtes, 13 compositions | 8 192 | 49,288 | 6,577 | ×7,49 | 1 245 522 |

Chaque valeur est la médiane de trois répétitions de 32 décisions, après
préchauffage. L'ordre des deux backends alterne entre répétitions. Le chronomètre
mesure le temps mural avec synchronisation CUDA aux bornes, hors profiler et hors
préparation du graphe. Les sorties du nouveau backend possèdent leur propre
mémoire : leur conservation dans un replay est incluse dans le coût mesuré.

Les échantillons bruts, temps de préparation et empreintes SHA-256 des sources
sont dans `comparison.json`. Ce sont des mesures locales de simulation seule,
sans inférence de politique, optimisation du réseau ni IPC. Le facteur de gain
ne doit pas être appliqué directement au temps d'un entraînement complet.

## Coûts identifiés et changements

Le profilage de cinq décisions de 512 salles révèle 6 130 appels CPU
`cudaLaunchKernel` pour les tireurs avec correction terminale, soit environ
1 226 par décision. Les opérations élémentaires, les indexations et les copies
se succèdent depuis Python. La préparation des tirs ennemis, les snapshots et
restaurations des salles terminées, les resets et la construction des
observations contribuent à ce coût.

Le nouveau backend :

- capture la séquence complète et réutilise ses buffers, avec un lancement de
  graphe par décision ;
- réinjecte les états calculés dans des emplacements GPU stables pour que chaque
  replay reparte bien de l'état précédent ;
- préalloue la constante de spawn qui empêchait la capture ;
- regroupe les sorties par type, avec deux copies de groupes plutôt qu'une
  allocation et un lancement pour chaque sortie ;
- conserve les générateurs et les tirages exacts de la référence, y compris
  pour les salles qui ne resetent pas ;
- conserve le gel des états après terminaison à l'intérieur d'une décision.

La deuxième itération a remplacé les clones individuels des sorties du premier
prototype par deux groupes. Les gains annoncés dans le tableau concernent cette
version finale.

**Cette passe supprime surtout le coût de pilotage depuis Python. Elle ne fusionne
pas encore les noyaux physiques en quelques gros noyaux CUDA.** Les nombreuses
opérations subsistent à l'intérieur du graphe. Le profilage ajoute lui-même une
forte surcharge ; ses durées ne remplacent pas le benchmark hors profiler.

Fichiers de diagnostic : `baseline-profile.json` et `stage-profile.json`.

## Vérifications exécutées

28 tests réussis : 10 nouveaux tests, 5 tests existants de frontières terminales,
13 tests existants de physique V2.

Les nouveaux tests comparent exactement, sans tolérance numérique, les sorties
et tous les tenseurs d'état de la référence et du nouveau backend. Ils couvrent
les tireurs, les 13 compositions mixtes, les resets automatiques répétés,
les resets explicites partiels, le mode sans auto-reset et le mode diagnostic
où les blessures ne terminent pas l'épisode. Les états des générateurs aléatoires
sont également comparés. Les trajectoires GPU principales couvrent 50 décisions
sur 64 salles avec timeouts à 7 décisions pour provoquer de nombreux resets.

Deux scénarios ciblés vérifient un clear au premier tick et une blessure qui
doit empêcher un kill ultérieur. Un autre vérifie qu'une observation conservée
reste intacte après la décision suivante. Les changements de configuration,
les mauvais types d'actions et les remplacements des tenseurs d'état sont rejetés.

Cette égalité concerne le simulateur existant et les situations testées. Elle
n'ajoute aucune certification de fidélité au vrai jeu Isaac.

## Utilisation dans le dépôt

```python
from research.gpu_simulator import GraphReadyNativeEnv, CudaGraphSimulator

env = GraphReadyNativeEnv(8192, "cuda", seed=42, enemies=4)
# Régler ici env.p, env.native, env.config et env.keep_terminal.
sim = CudaGraphSimulator(env)
obs, reward, done, info = sim.step(actions)  # int64 CUDA, forme [8192]
```

Pour le mélange, utiliser `GraphReadyCombatSuiteEnv(ids, "cuda", seed, native)`.
Les paramètres et la taille du lot sont figés à la capture. La lecture des états
se fait via `sim.env`. Pour modifier un état, utiliser `copy_` ou une écriture en
place. `sim.reset(mask)` permet les resets manuels, y compris si le graphe a été
capturé avec `keep_terminal=True`. Un nouveau graphe est nécessaire si les
paramètres physiques ou le mode terminal changent. Utiliser le flux CUDA qui a
créé le simulateur.

`step_view(actions)` évite les copies des sorties, mais celles-ci sont écrasées
à la décision suivante. Le benchmark utilise toujours `step`, sans cette
restriction de durée de vie.

Depuis la racine du dépôt :

```powershell
.\.venv\Scripts\python.exe -B -m unittest research.test_gpu_simulator research.test_terminal_physics_review -v
$env:PYTHONPATH = (Join-Path (Get-Location) 'python')
.\.venv\Scripts\python.exe -B -m unittest python.test_physics_v2 -v
.\.venv\Scripts\python.exe -B -m research.benchmark_gpu_simulator --output data/gpu-simulator-comparison.json
```

## Suite utile

Mesurer la boucle d'apprentissage complète avant de modifier ses lanceurs. Le
backend est prêt pour une intégration explicite ; il n'est pas activé par défaut.
Ensuite, les principaux candidats à une deuxième passe sont la fusion des
noyaux de physique, les copies terminales et le reset géométrique du mélange
(64 candidats par ennemi, calculés pour tout le lot à chaque reset appelé).
Toute économie de tirages aléatoires devra être versionnée, puisqu'elle peut
changer les trajectoires à seed identique.

Le graphe conserve des allocations intermédiaires pour les réutiliser. Cette
passe ne revendique donc pas une réduction de VRAM. Une mesure dédiée sera utile
avant de viser des lots nettement plus grands ou de partager le GPU avec un
modèle important.
