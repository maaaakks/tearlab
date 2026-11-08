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
- [ ] Choisir les optimisations à promouvoir et mesurer la boucle d'apprentissage complète.

Voir [le protocole et les commandes](docs/GPU_ABLATION_PLAN.md). Aucun test GPU lancé pendant cette passe.
