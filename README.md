# IsaacRL — GPU combat research

Experimental reinforcement learning for **The Binding of Isaac**, using an
empirically calibrated combat simulator, batched GPU physics and PPO. A Lua mod
can run a frozen policy against vanilla enemies in an owned copy of the game.

This is a research prototype: **two enemy archetypes, at most four enemies,
rectangular rooms**. It is not a complete Isaac simulator or an agent that plays
a full run. Simulator scores and qualitative in-game observations are reported
separately. No game executable, assets, decompiled code, saves or trained weights
are distributed here.

## Current result

A controlled experiment compared 200 updates with a three-contact budget then
400 no-hit updates against 600 strict updates. Same parent, scenes, rewards and
19,660,800 new decisions per arm; two optimization seeds.

| Large room, four distant shooters; no-hit wins | Strict | Recovery → strict |
|---|---:|---:|
| Seed 1, final checkpoint | 4.3% | 24.3% |
| Seed 2, final checkpoint | 14.5% | 21.5% |

The selected candidate scored 23.4%, versus its parent's 3.9%, on fresh random
layouts of known factors. Retention averaged 89.1% versus 88.7%. Only one seed
passed the development retention guard. These are **simulator results**, not
measured win rates in Isaac or proof of generalization to unseen enemy families.

See [results](docs/FOCUS_RECOVERY_RESULTS.md) and
[protocol](docs/FOCUS_RECOVERY_PROTOCOL.md). The earlier roughly 42× headline
concerned simulation; it must not be presented as end-to-end training speed.

## Install and run CPU checks

Python 3.11 or newer; use a virtual environment. Install PyTorch for your
platform first, then the other requirements:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B -m unittest research.test_terminal_physics_review research.test_mastery research.test_focus_recovery -v
```

The default suite uses CPU reference physics and skips GPU tests. No game,
checkpoint, CUDA toolkit or paid service is needed for those tests. A fresh
environment install has not yet been verified; the prepared checkout is tested
with the existing research environment. [Environment details](docs/REPRODUCIBILITY.md).

## GPU training

Requires CUDA-enabled PyTorch, NVIDIA CUDA Toolkit (`nvcc`), and a compatible
GPU. Native CPU compilation on Windows also requires an MSVC developer shell.
The native builder defaults to architecture 89 (the tested RTX 4070).

```powershell
.\.venv\Scripts\python.exe -B research/focus_recovery.py --parent PATH_TO_COMPATIBLE_CHECKPOINT.pt --output artifacts/focus-run
```

A compatible checkpoint contains `state_dict` for `CrossPolicy('pool')`.
Published scores came from a trained lineage; random weights cannot reproduce
them with the continuation budget above. Checkpoints are not bundled. W&B runs
offline; training never requires logging into a cloud account. No training is
launched by importing the modules or running the default CPU tests.

## Live preview

See [live preview setup](docs/LIVE_PREVIEW.md). It is opt-in and requires an
owned copy of Isaac Repentance and compatible LuaSocket support. Nothing installs
itself, edits saves or enables the debug console automatically.

## Layout

| Path | Purpose |
|---|---|
| `research/` | GPU simulator, PPO and curriculum experiments |
| `python/` | Physics, policies and frozen-policy TCP bridge |
| `mod/` | Standalone optional live preview |
| `data/physics_v2/calibration_r5.json` | Small empirical physics configuration |
| `docs/` | Protocols, results and limitations |
| `artifacts/` | Ignored local outputs and native build cache |

## Repository status

This is the local **`research/gpu-curriculum-public` preparation branch**, curated
from an active research checkout. Historical experiments and raw logs are kept in
the original checkout; `PUBLIC_SNAPSHOT.json` records source hashes and revision.
Nothing has been pushed by this cleanup. Do not merge the branch into `main`
until the preparation checklist is reviewed.

The project license is **not yet specified**; no license is invented for the
existing project code. See [third-party notices](THIRD_PARTY_NOTICES.md) and the
[publication checklist](docs/PUBLICATION.md).
