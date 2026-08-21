# Binding of AI

A reinforcement learning agent that learns to play *The Binding of Isaac: Repentance* through curriculum-based training. A Lua mod inside the game communicates with a Python PPO trainer over TCP, streaming observations and receiving actions at 30 Hz.

This project also serves as a case study for a YouTube documentary series — development stories live in `docs/story/`.

## Architecture

```
Isaac (Lua Mod, 30 Hz)  ──TCP:9999──  Python (PPO via Stable-Baselines3)
```

The Lua mod owns the game clock. It serializes the room into an 8-channel grid + 14 player features, sends it over TCP as JSON, and polls non-blockingly for actions. Python's `IsaacEnv` (a Gymnasium wrapper) blocks on each state, computes rewards from state diffs, and feeds everything into SB3's PPO.

Lua never waits for Python — if Python is slow, the last action is re-applied. Python syncs across episode boundaries via `episode_id`. See [docs/archive/system_architecture.md](docs/archive/system_architecture.md) for a full diagram.

## Quick Start

### 1. Install the mod

```powershell
.\scripts\install_mod.ps1
```

This symlinks `mod/` into Isaac's mods directory. Use `-Method Copy` for a hard copy instead.

### 2. Set up Python

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r python\requirements.txt
```

### 3. Prepare Isaac

- Add `--luadebug` to Steam launch options
- Close Isaac once and enable background training:

```powershell
.\scripts\configure_background_training.ps1
```

- Launch the game, enable **IsaacRL** from the Mods menu
- Verify the debug console shows `IsaacRL: Mod loaded`

### 4. Commandes d'entraînement — copier/coller

Lancez ces commandes depuis la racine du projet, avec Isaac ouvert et le mod
**IsaacRL** activé :

```powershell
cd C:\Users\maxbr\Desktop\IsaacRL\binding-of-ai
```

#### Repartir de zéro (nouveau modèle)

```powershell
.\.venv\Scripts\python.exe python\train.py --config configs\phase1a.yaml
```

Cette commande crée une nouvelle session d'entraînement. Elle ne remplace pas
les checkpoints des sessions précédentes.

#### Reprendre le dernier checkpoint

```powershell
.\.venv\Scripts\python.exe python\train.py --config configs\phase1a.yaml --resume latest
```

Cette commande nécessite qu'au moins un checkpoint ait déjà été enregistré.
Si `checkpoints\live` est vide, utilisez d'abord **Repartir de zéro**. Un
checkpoint périodique sera ensuite créé toutes les 10 minutes ; arrêter
proprement l'entraînement avec `Ctrl+C` crée également un checkpoint.

#### Reprendre un checkpoint précis

```powershell
.\.venv\Scripts\python.exe python\train.py --config configs\phase1a.yaml --resume "checkpoints\live\<run_id>\periodic_000012345.pt"
```

Remplacez le chemin d'exemple par celui du fichier `.pt` voulu dans
`checkpoints\live\<run_id>\`. La reprise restaure le modèle, l'optimiseur, les
compteurs d'entraînement et l'état aléatoire sauvegardé.

#### 💎 Premier run généraliste (full-game, multi-salles) — priorité actuelle

Le jalon B (TODO) est prêt côté Python (tests verts). Pour lancer la toute
première **vraie run naturelle** — où l'agent nettoie des salles, traverse des
portes, change d'étage, ramasse du loot et utilise des bombes **dans le même
épisode** :

```powershell
.\scripts\launch_full_run.ps1
```

Équivalent sans script :

```powershell
.\.venv\Scripts\python.exe python\train.py --config configs\full_game.yaml
```

Ce que fait `configs\full_game.yaml` :
- `full_run: true` → une salle nettoyée **ne met plus fin à l'épisode** ; seule
  la mort, la victoire, un abandon ou le timeout global clôt le run ;
- layouts/enemies/ennemis **naturels** (rien d'injecté) ;
- mémoire de carte (`map`) + encodeur d'objets (`entities`) dans l'observation ;
- timeout global de sécurité (~45–90 min de jeu) au lieu d'une durée de salle ;
- `gamma: 0.995` pour les conséquences multi-salles ;
- run W&B nommé `phaseB-full-run-natural` (tags `phaseB`, `full-run`, `natural`,
  `map-memory`).

Rappels pour que la run tourne réellement de bout en bout :
1. Isaac ouvert, mod **IsaacRL** activé (`--luadebug` + options.ini édité **jeu
   fermé**).
2. Lancer la commande ci-dessus **une fois Isaac prêt**.
3. Vérifier dans la console que les runs durent des centaines de steps et que
   l'`episode_id` reste stable à travers les clears de salle.
4. `Ctrl+C` pendant l'entraînement sauvegarde un checkpoint `interrupted`.

### 5. Parallel Launcher / TUI

For multi-instance runs, launch the worker manager first:

```powershell
.venv\Scripts\python python\launcher.py tui --workers 8 --batch-size 4
```

The TUI monitors worker windows and TCP readiness, and lets you launch batches,
send start sequences to selected workers, or terminate specific sandboxes.

For the headless worker flow, use:

```powershell
.venv\Scripts\python python\launcher.py launch --workers 8 --batch-size 4
```

### 6. Monitor

```powershell
tensorboard --logdir logs
```

Or enable [Weights & Biases](#weights--biases) in your config for richer experiment tracking.

## Observation & Action Space

**Observations** (`Dict`):

| Component | Shape | Description |
|-----------|-------|-------------|
| `grid` | `(8, 7, 13)` | 8-channel spatial grid — walls, obstacles, pits, player, enemies (normalized HP), projectiles, pickups, doors |
| `player` | `(14,)` | HP (red/soul/black), speed, damage, range, fire rate, shot speed, luck, bombs, keys, coins, active item, charge |

**Actions** (`MultiDiscrete([9, 5])`):

- **Move (0-8)**: idle, 4 cardinal, 4 diagonal
- **Shoot (0-4)**: none, up, down, left, right

## Reward Shaping

Rewards are computed from state diffs each tick:

| Component | Default | Trigger |
|-----------|---------|---------|
| `damage_dealt` | +1.0 | Per HP of damage dealt to enemies |
| `enemy_killed` | +5.0 | Per enemy eliminated |
| `damage_taken` | -10.0 | Each time the player loses HP |
| `room_cleared` | +20.0 | All enemies in the room are dead |
| `death` | -50.0 | Player dies |
| `time_penalty` | -0.1 | Every tick (encourages speed) |
| `survival_bonus` | 0.0 | Per tick while enemies remain (optional) |

All weights are configurable per phase via YAML.

## Training Curriculum

Training follows a curriculum of increasing difficulty (see [CHECKLIST.md](CHECKLIST.md)):

| Phase | Config | Goal |
|-------|--------|------|
| **1a** | `phase1a.yaml` | Kill Gapers — learn to aim and shoot |
| **1b** | `phase1b.yaml` | Kill Monstro — learn to dodge projectiles |
| **1c** | `phase1c.yaml` | Multiple mixed enemies — prioritization, kiting |
| **2** | `phase2.yaml` | Room combat with pickups |
| **3** | `phase3.yaml` | Single floor navigation |
| **4** | `phase4.yaml` | Multi-floor runs (Basement to Mom) |

Each phase resumes from the previous checkpoint to transfer learned behavior.

## Configuration

Configs are YAML files with five sections:

```yaml
env:
  frame_skip: 1              # Send state every Nth game tick
  max_episode_steps: 3000    # Episode timeout

reward:
  damage_dealt: 1.0
  enemy_killed: 5.0
  death: -50.0
  # ... all weights configurable

train:
  learning_rate: 0.0003
  total_timesteps: 1000000
  save_interval: 50000       # Checkpoint frequency

phase:
  enemy_type: 10             # Isaac EntityType (10=Gaper)
  enemy_count: 4
  spawn_enemies: true

wandb:
  enabled: false
  project: "binding-of-ai"
```

See [configs/phase1a.yaml](configs/phase1a.yaml) for a complete example.

## Checkpoints

Live checkpoints have one stable repository-local location:

```text
checkpoints/live/<YYYYMMDD_HHMMSS_run-name>/
├── periodic_000012345.pt
├── interrupted_000014000.pt
├── latest.json
├── metrics.jsonl
└── evaluations/
```

- A complete checkpoint is saved every `train.save_interval_seconds` (600
  seconds / 10 minutes in `phase1a.yaml`).
- `Ctrl+C`, crashes, and normal completion each save a final checkpoint.
- `checkpoints/live/latest.json` points to the most recently saved checkpoint.
- Resume with `--resume latest`, a run folder, or a specific `.pt` path.
- Existing run folders are never reused by a new from-scratch run, so equal
  step numbers cannot overwrite an older experiment.
- Every PPO update is appended to `metrics.jsonl`, keeping losses, KL, clip
  fraction, explained variance, feature health, exploration, throughput, and
  rolling gameplay outcomes available after the terminal closes.

### Deterministic evaluation

Stop the trainer, leave Isaac running with the mod enabled, then run:

```powershell
.\.venv\Scripts\python.exe python\evaluate_live.py latest --episodes 20
```

The evaluator disables sampling and reports mean/std reward, win/death/timeout
rates, episode length, kills, damage taken, and action diversity. Its JSON report
is saved beside the checkpoint under `evaluations/`.

During training, the live HUD and console also expose PPO health metrics:
KL divergence, clip fraction, explained variance, entropy, gradient norm,
rolling win rate, action entropy/idle rates, and dead-feature fraction. Phase 1a
uses entropy decay (`0.02` to `0.005`) to explore broadly early and stabilize
later, plus `target_kl: 0.03` to stop overly destructive policy updates.
The 22-value player vector is scaled to comparable numeric ranges before the
MLP, preventing large raw values such as tear range or consumable counts from
drowning out normalized positions and nearest-entity deltas.

## Weights & Biases

Enable experiment tracking by setting `wandb.enabled: true` in your config YAML. First-time setup:

```powershell
wandb login
```

Logs all hyperparameters, PPO metrics, and model checkpoints to your W&B dashboard.

## Speed Diagnostics

The environment tracks timing metrics to help tune the speed multiplier:

- **Frame drops**: Detected via `episode_tick` gaps between Lua and Python. If Python can't keep up with the game, frames are skipped.
- **Step latency**: Time Python spends waiting for each state. Low = game is the bottleneck. High = Python is faster than the game.
- Both are logged per episode: `frames_dropped=0 avg_latency=33.2ms`

## Project Structure

```
binding-of-ai/
├── mod/                    # Lua mod (runs inside Isaac)
│   ├── main.lua            # Episode lifecycle, frame skip
│   ├── state_serializer.lua# 8-ch grid + 14 player features
│   ├── action_injector.lua # Input interception
│   ├── game_control.lua    # Enemy spawning, resets
│   ├── tcp_server.lua      # TCP server (JSON lines)
│   └── config.lua          # Mod defaults
├── python/                 # Python training code
│   ├── train.py            # Main training loop
│   ├── isaac_env.py        # Gymnasium environment
│   ├── reward.py           # Reward shaping
│   ├── network.py          # CNN+MLP feature extractor
│   ├── config.py           # Config dataclasses
│   ├── evaluate.py         # Model evaluation
│   └── utils.py            # Checkpoint utilities
├── configs/                # YAML configs per curriculum phase
├── scripts/                # PowerShell helpers
│   ├── install_mod.ps1     # Mod installation
│   └── launch_training.ps1 # Training launcher
├── checkpoints/            # Saved models (auto-created)
├── logs/                   # TensorBoard events (auto-created)
└── docs/
    ├── archive/            # Architecture diagrams, old plans
    └── story/              # Documentary case-study writeups
```

## Requirements

- Windows 10+ with Steam
- The Binding of Isaac: Repentance
- Python 3.9+
- `--luadebug` Steam launch option
