# Training Log

This file captures durable lessons from training runs so we can reuse what we learn.

## Architecture Version Reference

| Version | Commit | CNN Channels | Pooling | CNN Output | Player Features | Player MLP | Bottleneck | Best Pickups |
|---------|--------|-------------|---------|------------|-----------------|------------|------------|-------------|
| V1 | `c0dcd08` | 8→32→64→64 | None | 5,824 | 14 | 14→64→64 | 5,888→256 | **~30** |
| V2 | `11c5345` | 8→32→64→128 | None | 11,648 | 22 | 22→128→128 | 11,776→256 | untested (too large) |
| V2b | `7c028c0` | 8→32→64→128 | AvgPool(1) | 128 | 22 | 22→128→128 | 256→256 | ~0 (spatial info destroyed) |
| V3 | `20ebed0` | 8→16→32→64 | MaxPool(2) | 1,152 | 22 | 22→64→64 | 1,216→256 | ~14 (unstable) |
| V4 | `99d4442` | 8→16→32→32 | None | 2,912 | 22 | 22→64→64 | 2,976→256 | ~16 |

## 2026-03-10 - Phase0b nav coin smoke frame-skip sweep
- Takeaway: `frame_skip=3` is the current sweet spot for learning speed and stability in this task setup.
- RL term: this behavior is commonly called `reward hacking` or `specification gaming` (the agent learns a shortcut that maximizes the specified reward efficiently).
- Evidence:
  - fs1 baseline (`phase0b-nav-coin-smoke`): https://wandb.ai/sepzie1/binding-of-ai/runs/8d74tvnm
  - fs3 (`phase0b-nav-coin-smoke-fs3`): https://wandb.ai/sepzie1/binding-of-ai/runs/0eean4d5
  - fs5 (`phase0b-nav-coin-smoke-fs5`): https://wandb.ai/sepzie1/binding-of-ai/runs/08h2ljsi
- Observed pattern:
  - fs3 reaches short-episode/high-reward behavior earlier than fs1 and fs5.
  - fs5 improves too, but converges later in W&B step-space and has slower env-step throughput at equal timesteps.
- Confidence: medium-high.
- Follow-up: run longer training budgets in future sweeps to confirm this ranking persists at larger timestep counts.

## 2026-03-12 - PPO hyperparameter tuning (phase0c multicoin)

- **Takeaway:** Default SB3 PPO settings were far too aggressive for our setup. Tuning `target_kl`, `clip_range`, and `learning_rate` dramatically improved training stability and reward.
- Evidence:
  - Before tuning (default-ish settings, lr=3e-4, n_epochs=10): clip_fraction ~0.5, approx_kl 0.10-0.15, reward oscillated wildly between runs. Run: https://wandb.ai/sepzie1/binding-of-ai/runs/kob9gwks
  - After tuning (lr=5e-5, clip_range=0.1, target_kl=0.03, n_epochs=4, batch_size=256, n_steps=2048): clip_fraction 0.20-0.40, approx_kl 0.02-0.05, explained_variance 0.75-0.85, smooth reward curve 50→100. Run: https://wandb.ai/sepzie1/binding-of-ai/runs/d8meaq3c
- Key metrics to watch:
  - `clip_fraction`: healthy range 0.1-0.2, above 0.3 means policy is changing too aggressively
  - `approx_kl`: healthy range 0.01-0.03, above 0.05 means updates are too large
  - `explained_variance`: should trend toward 0.8-1.0, below 0.5 means value function is struggling
  - `target_kl` is the most surgical lever — it early-stops epoch updates when KL exceeds threshold, preventing overshoot
- Confidence: high.
- Follow-up: clip_fraction is still 0.2-0.4 (ideally <0.2). Could try further reducing lr or increasing batch size if it becomes a bottleneck again.

## 2026-03-12 - Single-coin reward design limitations

- **Takeaway:** A large binary reward (+50 for coin pickup) with tiny time penalty (-0.03/step) creates a nearly binary reward signal. The model learns to reach the coin but has little gradient to optimize *speed* of collection.
- Solution: switched to multicoin respawn setup (coins respawn on pickup, +5.0 per coin, -0.05/step time penalty, 500-step timeout). This gives continuous reward signal proportional to collection efficiency.
- Confidence: high — reward mean went from ~60 (single coin) to ~100 (multicoin, ~20 coins/episode).

## 2026-03-12 - Observation resolution and wall-sticking

- **Takeaway:** The 7x13 grid observation is too coarse for precise navigation near room edges. Isaac and pickups exist at continuous positions but are quantized to discrete grid cells. When a coin is slightly above Isaac near the bottom wall, the model can't distinguish the small offset and gets stuck.
- Observed: Isaac frequently gets stuck running into walls when coins are just barely offset from its position, especially near room boundaries.
- Candidate fixes (untested):
  1. Add continuous player position + nearest pickup position to the player vector (cheap, targeted)
  2. Double grid resolution to 14x26 (expensive, general improvement)
- Confidence: medium — the wall-sticking pattern is consistent, but haven't confirmed it's purely an observation issue.
- Follow-up: try adding continuous positions to player vector first as the cheaper experiment.

## 2026-03-12 - Resuming from checkpoints with different hyperparameters

- **Takeaway:** When resuming training from a checkpoint with different config, several SB3 internals need manual fixup:
  1. `clip_range` must be wrapped in `constant_fn()` — SB3 expects a callable schedule, not a raw float.
  2. `n_steps` change requires recreating the rollout buffer — `PPO.load()` allocates the buffer at load time with the old size.
  3. Old checkpoint hyperparameters (value function quality, etc.) don't poison the new run — after ~100k+ steps, the model fully adapts to new settings.
- Confidence: high — hit both bugs in practice and confirmed fixes.

## 2026-03-16 - Bigger NN regression and wall collision penalty tuning

- **Context:** After adding distance vectors to the model input, explained variance was dipping to 0.2-0.5 and learning had stalled. We doubled the bottleneck (256 → 512) and scaled up the CNN (64 → 128 filters) and player MLP (64 → 128). Also added a wall collision penalty to prepare for obstacles.
- **Run `yhdfihe8` (`phase0d-obstacles-bigger-nn`):** Wall collision penalty was too large — it swallowed the coin reward. Mid-training, Isaac had learned to run parallel to walls while avoiding coins (interpreting lack of wall penalty as reward). By end of training, it learned to stand completely still until timeout.
- **Run `eaait2xd` (`phase0d-obstacles-double-reward-for-700-steps`):** Reduced collision penalty from 0.05 to 0.01 and doubled coin reward to 10.0 (compensating for longer 700-step episodes). Agent learned to collect 15+ coins per episode over 3M steps, but wall penalty was completely ignored.
- **Run `wu0qeijm` (`phase0d-obstacles-wall-penalty`):** Increased collision penalty back to 0.05. No improvement in wall avoidance. Training plateaued — pickups flat at ~18, reward flat at ~120.
- **Training stability issues in wu0qeijm:** clip_fraction 0.28-0.32 (healthy: <0.2), approx_kl 0.007-0.011 (rising), explained_variance 0.2-0.6 (noisy), value_loss 5-7. The optimizer was churning without finding improvement.
- **Qualitative regression:** Agent movement was erratic and unstable compared to the smoother coin-to-coin navigation achieved with the old smaller architecture.
- Confidence: high — consistent across multiple runs.

## 2026-03-16 - CNN bottleneck layer too large (architectural fix)

- **Takeaway:** The CNN had no spatial pooling — three conv layers with `padding=1` preserved the full 7×13 grid, producing a flattened output of 128 × 7 × 13 = **11,648 features**. Combined with the 128-dim player MLP, the bottleneck `fc` layer was 11,776 → 512, containing ~6M parameters. This caused two problems:
  1. Player features (position, distances) were only 1.1% of the bottleneck input, drowning out the spatial/distance info critical for smooth navigation.
  2. The 6M-param layer was undertrained with PPO's batch_size=256, leading to poor value estimates and erratic policy updates.
- **Fix:** Added `AdaptiveAvgPool2d(1)` after the last conv layer, reducing CNN output from 11,648 to 128. The bottleneck layer is now 256 → 512 (~131K params), and player features are 50% of the input.
- **Tradeoff:** Global average pooling discards spatial layout from the CNN (where walls/obstacles are). Wall/obstacle proximity is currently only represented in the grid, not the player vector. If wall avoidance regresses, nearest-wall distance should be added to the player features.
- **Expected outcome:** More stable training (lower clip_fraction, better explained_variance), smoother agent behavior, and player features actually influencing the policy. Requires training from scratch — old checkpoints are incompatible.
- Confidence: medium-high — architecturally sound, but untested.

## 2026-03-17 - Post-pooling network fails to learn coin collection

- **Run `zgrjgzvv` (`phase0d-obstacles-avgpool-fix-no-collision`):** Trained the new pooled architecture from scratch with no wall penalty. Over 1.2M steps, the model failed to meaningfully improve past the first 100K. Pickups_collected oscillated between 28-38 with no upward trend, ep_rew_mean plateaued at 120-150.
- **Key metric:** Explained variance was noisy (0.2-0.8), never stabilizing. Clip fraction 0.24-0.34 (still above healthy range). The critic couldn't reliably predict state-dependent returns.
- **Diagnosis:** The remaining reward signal (time penalty -0.05/step + sparse coin pickup +10) lacks dense, state-dependent variation. Time penalty is constant regardless of behavior (-35/episode). Coin pickups are the only variable signal but are sparse and delayed. The bigger network (300K params vs old 100K) has a flatter loss landscape and needs stronger gradients to find structure.
- **Fix attempted:** Added potential-based pickup approach reward shaping (`pickup_approach_scale`) — a per-step reward proportional to change in distance to nearest pickup.
- Confidence: medium — reward shaping is well-established in RL literature, but the scale (1.0) may need tuning.

## 2026-03-17 - AdaptiveAvgPool2d(1) destroys spatial info, approach shaping bug

- **Run [`5amcfl6b`](https://wandb.ai/sepzie1/binding-of-ai/runs/5amcfl6b) (`phase0d-pickup-approach-shaping`):** Approach shaping at scale 10.0 with the global-avg-pooled architecture. 100K steps, no learning at all. Explained variance ≈ 0 (scale 0–0.0008). Policy entropy at theoretical maximum (~2.195 vs max 2.197) — effectively uniform random.
- **Root cause 1 — architecture:** `AdaptiveAvgPool2d(1)` collapses the 7×13 grid to 1×1 per channel, destroying all spatial information. The critic can't distinguish states because it can't see *where* things are on the grid. Player vector features alone weren't enough to compensate.
- **Root cause 2 — reward bug:** On the step a coin is collected and respawns, `nearest_pickup_dx/dy` jumps to the new coin's location. The approach shaping computed a large negative delta (~-6.8 at scale 10), nearly canceling the +10 collection reward.
- **Fix:** Replaced `AdaptiveAvgPool2d(1)` with `MaxPool2d(2)` (7×13 → 3×6, preserves spatial layout). Reduced CNN last layer to 64 channels. Bottleneck: 1,280→512 (656K params). Fixed approach shaping to skip on collection steps.
- Confidence: high on the diagnosis, medium on the fix — needs training validation.

## 2026-03-18 - Bigger network (V3 MaxPool) is less stable than original

- **Run [`mgspdtj4`](https://wandb.ai/sepzie1/binding-of-ai/runs/mgspdtj4):** 5M steps with MaxPool2d(2) architecture (64ch, 1,280→512 bottleneck) + approach shaping + bug fix. Model learned for ~500K steps (ep_rew 0→100, explained_variance 0→0.6), then became unstable. Explained variance repeatedly collapsed from 0.6 to 0.2. Clip fraction 0.25-0.35 (well above healthy 0.1-0.2). Approx_kl rising steadily from 0.002 to 0.01. Pickups oscillated 4-14, never improved past 500K.
- **Diagnosis:** The larger network (300K+ params) with the same hyperparameters tuned for the smaller network caused oversized policy updates. Each gradient step changed more parameters, producing higher clip fractions and unstable critic learning. The bigger network was strictly worse than the original V1 which collected 30+ coins.
- **Fix:** Designed V4 architecture with fewer CNN channels (8→16→32→32, no pooling, flatten). Bottleneck: 2,976→256. Policy/value heads back to original size (128→64). Learning rate raised from 5e-5 to 1e-4.
- **Important note:** V4 is NOT the same as V1. V1 had wider CNN channels (8→32→64→64, CNN output 5,824) with 14 player features. V4 has narrower channels (8→16→32→32, CNN output 2,912) with 22 player features (added distance vectors). V4 has half the CNN spatial capacity of V1.
- **Lesson:** Bigger networks aren't automatically better in RL. The original architecture's stability was more valuable than extra capacity. The real bottleneck fix was reducing CNN channels, not adding pooling.
- Confidence: high — clear regression with evidence across 5M steps.

## 2026-03-19 - V4 architecture with approach shaping: learning but slow

- **Run [`man2a6p8`](https://wandb.ai/sepzie1/binding-of-ai/runs/man2a6p8):** 5M steps with V4 architecture (8→16→32→32 channels, no pooling, 2,976→256 bottleneck) + approach shaping (scale 10.0) + wall collision penalty (-0.05). lr=5e-5, pickup_collected=10.0.
- **Results:** Model learned steadily but slowly. Pickups collected rose from ~4 to 14-16 by 5M steps. Ep_rew_mean climbed from -20 to ~120. Pickup rate near 1.0 (almost always picks up at least one coin). Wall collision steps ~300-400 (noisy, no clear downward trend).
- **Training health:** Explained variance noisy 0.3-0.6 (never stabilized above 0.6). Clip fraction 0.2-0.35 (still above ideal). Approx_kl rising from 0.004 to 0.01. Entropy loss rising from -2.1 to -1.7 (policy exploring more). Value loss rising with reward scale (3→7).
- **Comparison:** The original V1 architecture (no approach shaping, no distance vectors, wider CNN: 8→32→64→64 = 5,824 CNN features, 14 player features) reached ~30 pickups in similar or less training time. V4 has half the CNN capacity (2,912 vs 5,824 features) plus 8 extra player features that were supposed to compensate. Multiple variables changed simultaneously: CNN capacity halved, distance vectors added, approach shaping added, wall collision added. Impossible to attribute the regression to any single change.
- **Next steps:** Fixed wall collision false positives (momentum during direction changes, diagonal movement). Bumped pickup_approach_scale 10→20 and wall_collision_penalty -0.05→-1.0. Reduced lr remains at 5e-5. Running new training to see if stronger signals help.
- **Suggested ablation:** Run V1-exact architecture (8→32→64→64, 14 player features, no shaping, no wall penalty) to re-establish the baseline, then add changes one at a time.
- Confidence: high on the data, uncertain on the path forward — too many simultaneous changes to diagnose.

## 2026-03-21 - V1 baseline ablation: ~30 hours, ~36 pickups, patience matters

- **Context:** Ran V1 architecture (8→32→64→64 CNN, 14 player features) as an ablation baseline with masked distance vectors, no approach shaping, no wall penalty. pickup_collected=5.0, 500-step episodes, lr=5e-5.
- **Runs:** [`kcxnmmds`](https://wandb.ai/sepzie1/binding-of-ai/runs/kcxnmmds) → [`qhk2v9kj`](https://wandb.ai/sepzie1/binding-of-ai/runs/qhk2v9kj) → [`web1zl67`](https://wandb.ai/sepzie1/binding-of-ai/runs/web1zl67) → [`6sfr9qbs`](https://wandb.ai/sepzie1/binding-of-ai/runs/6sfr9qbs) (sequential resumes, ~30 hours total). Eval run: [`pxbg826y`](https://wandb.ai/sepzie1/binding-of-ai/runs/pxbg826y).
- **Result:** Model converged at ~36 pickups/episode average. This is the best performance achieved so far and confirms V1 as the strongest architecture tested.
- **Key lesson — patience in evaluation:** Previously judged runs in 300K–1M step windows, which sometimes missed ongoing growth. Letting the model train until it truly plateaued for >1M steps revealed continued improvement that shorter windows would have missed. The model was still improving at points where earlier runs were cut short.
- **Takeaway:** When evaluating RL training, don't cut runs short based on short-window stalls. Look for a true plateau sustained over 1M+ steps before concluding the model has converged. Growth can be slow and noisy — what looks like a plateau in a 500K window may be the middle of a long upward trend.
- Confidence: high — 30 hours of training, clear convergence.
- Follow-up: now adding wall collision penalty (-0.05) and unmasking distance vectors to see if they can push beyond 36 pickups.

## 2026-08-21 - Checkpoint resume when an observation branch is added (entities)

- **Bug:** Resuming a pre-entities checkpoint with the new entity-set branch crashed in `_load_checkpoint`: `RuntimeError: size mismatch for features.fc.0.weight` (`[256, 5888]` vs `[256, 5952]`). The entities encoder appends 64 features to the fused input, growing the first projection layer's input dim. `load_state_dict(strict=False)` only drops *missing* keys — it cannot bridge a *size* mismatch, and the optimizer would have hit the same wall next (param-set mismatch).
- **Fix (in `live_trainer.py`):** `_adapt_state_dict` copies the overlapping columns of any layer whose last dim grew under section (pre-existing streams are concatenated first, the new stream last), leaves the new columns at fresh init, and raises on incompatible shapes. As a heuristic rule, both the policy `fc.0` and value head share the same input, so both grow by the same delta. The optimizer state is only restored when the parameter sets match exactly (`_architecture_matches`); otherwise it is skipped with a warning and Adam moments reset.
- **Lesson:** "Additive" observation extensions (new encoder branches) silently change the *shape* of fuse layers, not just the key set. Treat "(strict=False) fresh weights for new branches" narrowly — it only covers brand-new params, so size-mismatch adaptation must be explicit and tested. Also, if you transfer weights on a sliced concat you cannot transfer the optimizer's moments (the param set is different), so reset them.
- Confidence: high — regression test reproduces the exact save-then-add-entities-resume path and passes.
- Follow-up: this only grows columns for append-style concat; a reordering of streams would need param-name remapping instead.

## 2026-08-21 - Success-conditioned urgency and unified aim supervision

- **Evidence:** The first live `phase1a` run reached roughly 67k steps with improved movement entropy but no durable clear rate. The 20-episode reward average fell to about -7.61 while shoot entropy remained about 0.97. The per-step time cost was only -0.0005 (at most -0.075 over 150 steps), so it was not the main regression. Death and timeout both paid -5, damage progress could recover +1, and the cumulative aim penalty made longer unsuccessful episodes more negative.
- **Takeaway:** Urgency must be conditional on success. Remove the continuous time and aim penalties, pay a decaying speed bonus only on room clear, make early death worse than late death, and scale timeout cost by enemy HP left. This preserves the ordering `fast clear > slow clear > progress at timeout > survival > late death > early death`.
- **Implementation:** The same PPO policy receives a small auxiliary cross-entropy loss on its existing shoot head. Nearest-enemy direction supplies one of eight direction labels; no visible enemy supplies a lower-weight idle label. There is no separate aiming model, training phase, or transfer checkpoint.
- Confidence: medium — the incentive bug is confirmed from reward arithmetic and live metrics; the replacement still needs a clean run from zero to validate clear rate and aim accuracy.
- Follow-up: compare the new run at 50k/100k real steps using clear rate, aim-visible accuracy, shoot entropy and reward, not reward alone.

## 2026-08-21 - CUDA and single-instance speedhack

- **Evidence:** Single-worker live `phase1a` sampled about 8.7 transitions/s, so a 1,024-step rollout delayed a policy refresh by roughly two minutes. The current 1v1 episodes average about 70 transitions.
- **Configuration:** `phase1a.yaml` uses one Isaac and `n_steps: 512`. With a 4x game speed this targets a policy refresh every 12-15 seconds without reducing the PPO batch to a single short episode.
- **Hardware:** The local venv now uses `torch 2.13.0+cu130`; CUDA was verified on the RTX 4070. GPU acceleration improves PPO minibatch updates, while game simulation/IPC remain the principal throughput limit.
- **Speed policy:** Start at 4x and measure game tick rate, `steps_per_second`, KL and clip fraction. The earlier 10x experiment caused stale action windows and did not yield a proportional end-to-end throughput.
- Confidence: medium — CUDA is verified and unit tests pass; the first x4 run still needs live throughput validation.

## 2026-08-21 - In-mod speedhack: render artifacts above ~2x are cosmetic only

- **Evidence:** The in-mod speedhack (nested `Game():Update()` calls per rendered frame, replacing the Cheat Engine speedhack) produces visual glitches at multipliers above ~2x. The agent's behavior and metrics stay normal — episodes, rewards and actions are unaffected.
- **Root cause:** Rendering sees only every Nth simulation state once multiple logic ticks run between frames. Render-coupled effects (damage flashes, transition fades, screen shake, sprite animation timing) assume a 1:1 frame-to-tick relationship and desync past ~2x. This is inherent to in-process fast-forward; the CE speedhack exhibited the same class of artifact.
- **Why training is unaffected:** Observations are serialized from game memory (`state_serializer.lua`: positions, grid, HP), not from pixels. A glitched frame on screen corresponds to consistent underlying game state.
- **Policy:** Run x4 for unattended training and ignore artifacts. Drop to x2 or x1 when spectating or recording footage.
- Also fixed: the speedhack loop previously kept calling `Game():Update()` after an episode reset fired mid-nested-tick; it now breaks when `isResetting()` or the player is gone.
- Confidence: high on cosmetic-only impact (agent behavior confirmed normal); medium on the exact ~2x threshold.

## 2026-08-22 - n_steps must cover several full episodes

- **Evidence:** With `n_steps: 512` and a 300-step episode horizon (30s rooms), each PPO batch contained ~1.7 episodes. Terminal signals (timeout/death/clear) arrived in irregular waves; the 4v1 run peaked at -9.4 rolling reward @50k then regressed to -11.2 @100k.
- **Change:** `n_steps: 2048`, `batch_size: 256` (~7 episodes per batch).
- **Result @100k:** no post-50k regression (-9.49 held vs -11.17), same wall-clock throughput (~160 steps/s). Win rate not yet improved; zero KL early-stops means lr headroom remains.
- Confidence: medium — single seed, noisy metric; confirm at 200-300k before generalizing.
- Follow-up: rule of thumb going forward: keep `n_steps` >= 3-5x mean episode length.

## 2026-08-22 - World-model probe (S1): a next-state head must see the action

- **Evidence:** The S1 `next_state` summary head (v1, `next_state_coef: 0.05`, state-only input) converged in ~20k steps then plateaued exactly at the chance baseline: `next_state_ratio` ≈ 1.0 from 40k to 260k steps (run `phase1a-4v1-speed12-h30s-safety-aux-nextstate`). Ratio 1.0 = MSE of predicting the batch mean.
- **Root cause:** v1 predicted `summary(s_{t+1}) = f(s_t)` — the taken action was not part of the input. Without action conditioning the best achievable prediction IS the transition marginal mean, so the plateau was the theoretical optimum of the mis-specified problem. The doc (`DREAMERV4_update.md`) specified f(état, action); the implementation dropped the action.
- **Fix (S1 v2):** one-hot MultiDiscrete action concatenated to the pi features at the head input; targets changed to DELTA `summary(s_{t+1}) - summary(s_t)` (removes the trivial persistence shortcut of absolute positions); per-component losses added (`next_state_loss_player/proximity/enemies`) to localize any future plateau.
- Confidence: high on the diagnosis (ratio pinned at exactly the chance value + information-theoretic argument), low on v2 outcome — needs a fresh 100k run.
- Follow-up: if v2 still plateaus above ~0.7 by ~100k, suspect observation coverage (enemy velocities unobservable, frame_stack=1) before abandoning the world-model track.

# 2026-11-08 — Comparaisons natives GPU préparées, mesures différées

- Le protocole sépare les ajouts individuels, retraits individuels et fusions afin de ne pas attribuer à une proposition un changement d'orchestration. Le graphe existant reste la référence de débit.
- Contrôles CPU : 34 configurations, événements discrets et état RNG identiques ; écart flottant absolu maximal 0,000152587890625. Les seuils futurs CUDA sont explicites, sans promotion automatique en cas d'échec.
- CPU/PTX/CUBIN compilés hors ligne ; aucune exécution GPU, aucun entraînement ni changement de checkpoint. La compilation indique 64 registres pour la décision par thread, 113 par groupe ; ce n'est pas une mesure de vitesse.
- Confiance : contrats CPU vérifiés, fonctionnement et gains CUDA encore à qualifier. Prochaine étape : court diagnostic GPU puis matrice complète autorisée, pas de long entraînement avant ces mesures.
- Sources et preuves : `docs/GPU_ABLATION_PLAN.md`, `docs/evidence/gpu-native-v2/`, `research/gpu_ablation.py`.
