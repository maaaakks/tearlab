# Optional frozen-agent preview

The files in `mod/` are a standalone preview. Copy them into a separate Isaac mod
directory only when you explicitly want to run the demo. Do not overwrite an
existing IsaacRL installation or saves. This preparation checkout does not
install or activate the mod.

Requirements: an owned Isaac Repentance installation, Lua mod support and a
working `require('socket')` in that Lua runtime. LuaSocket is not bundled. The
original play test used Repentance1.7.9b; other versions are unverified.

Start the CPU policy bridge separately:

```powershell
.\.venv\Scripts\python.exe -B python/play_frozen_live.py --checkpoint models/latest.pt --output artifacts/live-session --auto
```

The bridge checks the checkpoint schema; live-compatible weights use
`crossfire-v1` with pooling policy or the bundled `mastery-v1` checkpoint.
The bundled latest weights load directly, without converting their original bytes.
They have not been validated in Isaac; the current game installation is unchanged.

The game and bridge communicate over localhost TCP10009. After entering a run,
the auto demo cycles11 scenarios with Horfs, Gapers and mixtures. First contact,
victory or40seconds ends the scenario. Hearts are protected; each contact still
counts as failure. Pickups are cleared at reset and suppressed during the loop.

`C` stops/resumes the demo. A `STOP` file in the session output directory stops
the bridge. Disconnect, stale actions and capacity guards release agent input.
No navigation, item usage, bombs or full-run competence is provided.
