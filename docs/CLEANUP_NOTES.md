# What changed in this preparation branch

The active research checkout was not cleaned in place: it contains substantial
pre-existing uncommitted moves, deletions and dependencies. This branch uses a
separate Git worktree and a curated current tree; the original state is preserved.

- Include the import dependency closure for the current GPU curriculum and tests.
- Include the small calibration JSON, with source trace hashes but relative paths.
- Replace hard-coded personal repository paths with paths relative to the module.
- Make the training parent checkpoint explicit and put the native cache under artifacts.
- Use the standalone optional preview instead of the old mixed training mod entry.
- Preserve scientific result plots, but omit raw sessions, binaries, checkpoints,
  generated evidence copies, abandoned experiments and desktop tooling.
- Keep existing history and third-party attribution. No global license is assigned.

Checks performed locally with the existing project environment:11 CPU tests pass,
2 GPU tests skipped by default; training CLI help works; all22 Python modules
parse. Known credential signatures were not found in the prepared tree or114
reachable local commits. This is a limited scan, not a guarantee that all kinds
of secrets or personal data are absent. No fresh-environment install or new GPU
compilation has been claimed for this preparation branch.

The original branch remains the experimental training track. This branch is for
reviewing the public project layout; it is not automatically pushed or merged.
