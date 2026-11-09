# Latest TearLab checkpoint

Only `latest.pt` is distributed, with its SHA256 sidecar `latest.meta.json`.
It is the final checkpoint of the most recently completed training run:
`replay50-seed291773`, strict no-hit consolidation with50% retained scenarios.
The bytes are unchanged from the original training checkpoint, including its
optimizer state and source hashes. No intermediate or ancestor weights are bundled.

- Architecture: pooling `CrossPolicy`,238 observations, previous-action token,
 45 joint movement/shooting actions. Checkpoint schema:`mastery-v1`.
- Final phase:200 updates,6,553,600 new decisions; this is a continuation of a
 trained lineage, not a model trained from scratch with that budget.
- Frozen simulator audit on new layouts of known factors:25.3% no-hit wins in
 the large-room/four-distant-shooter case;89.1% average retention;79.0% for the
 weakest retained family. This is the latest model, not a claim that it beats
 every earlier candidate on every metric.
- It has not been validated in the real game. The installed game agent is unchanged.
- No game files, screenshots, saves or trajectory recordings are in the checkpoint.
- Project-owned weights are dedicated under [CC0-1.0](../LICENSE).

CPU inference:

```powershell
.\.venv\Scripts\python.exe -B python/play_frozen_live.py --checkpoint models/latest.pt --output artifacts/live-session --auto
```

The bridge validates the SHA256 sidecar before loading. Loading weights does not
install or activate the mod. Original historical experiments started from other
parents, so this single latest file does not reproduce every published lineage.
