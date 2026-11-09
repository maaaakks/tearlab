# TearLab publication

- [x] Isolate cleanup in a dedicated worktree/branch; active checkout preserved.
- [x] Include the current simulator dependency closure and small calibration.
- [x] Replace machine-specific repository paths in current Python modules.
- [x] Replace stale README with scope, commands, measured results and limits.
- [x] Keep only the latest checkpoint and its hash; exclude intermediate runs,
  game files, saves, sessions and generated build artifacts.
- [x] Preserve upstream attribution for the JSON helper.
- [x] Apply CC0-1.0 to project-owned code/docs/weights at the owner's request;
  retain third-party notices and historical rights.
- [x] Review the prepared tree diff, credit the upstream and scan retained history
  for known credential signatures before publishing.
- [x] Verify installation and CPU checks in a fresh GitHub Actions environment.
- [ ] Verify GPU compilation in a fresh environment.
- [x] Bundle the latest weights and loading instructions. Earlier experiment
  parent weights are deliberately not bundled; do not claim exact lineage reproduction.
- [x] Prepare the TearLab identity, README banner and upstream fork attribution.
- [x] Replace em dashes with ordinary hyphens in the current published tree.

Publication target: `maaaakks/tearlab`, a public fork of `Sepzie/binding-of-ai`.
First push completed on 2026-11-09; the initial CPU workflow passed.
The active research checkout remains separate from this curated release tree.
The first release is based on the prepared snapshot and bundles the latest
consolidation checkpoint; it does not publish every local experimental file.

The source repository was already public when inspected. Local uncommitted
changes are not thereby on GitHub. The cleaned tree preserves existing history;
secret scanning is best effort and is not a guarantee about every possible
credential or historical personal datum. No history rewrite is performed.
