# Local publication preparation

- [x] Isolate cleanup in a dedicated worktree/branch; active checkout preserved.
- [x] Include the current simulator dependency closure and small calibration.
- [x] Replace machine-specific repository paths in current Python modules.
- [x] Replace stale README with scope, commands, measured results and limits.
- [x] Keep only the latest checkpoint and its hash; exclude intermediate runs,
  game files, saves, sessions and generated build artifacts.
- [x] Preserve upstream attribution for the JSON helper.
- [x] Apply CC0-1.0 to project-owned code/docs/weights at the owner's request;
  retain third-party notices and historical rights.
- [ ] Review the prepared tree diff and the retained history before publishing.
- [ ] Verify installation in a fresh Python environment and GPU compilation there.
- [x] Bundle the latest weights and loading instructions. Earlier experiment
  parent weights are deliberately not bundled; do not claim exact lineage reproduction.
- [ ] Review/push/merge separately. This cleanup does not publish anything.

The source repository was already public when inspected. Local uncommitted
changes are not thereby on GitHub. The cleaned tree preserves existing history;
secret scanning is best effort and is not a guarantee about every possible
credential or historical personal datum. No history rewrite is performed.
