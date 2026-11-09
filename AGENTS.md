# Working on this research snapshot

Use a project virtual environment. Keep training opt-in and W&B offline unless
the user requests otherwise. Never install the mod, edit game options/saves or
publish/push as part of tests. Commit logical changes and document measured
results separately from hypotheses and live-game behavior.

Run the CPU tests documented in README. Native GPU tests require explicit
`RUN_ISAAC_GPU_TESTS=1` and a writable `ISAAC_NATIVE_CACHE`. Do not introduce
machine-specific paths. Keep only the curated `models/latest.pt` checkpoint;
do not commit intermediate weights or generated sessions.
