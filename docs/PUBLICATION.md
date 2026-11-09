# Local publication preparation

- [x] Isolate cleanup in a dedicated worktree/branch; active checkout preserved.
- [x] Include the current simulator dependency closure and small calibration.
- [x] Replace machine-specific repository paths in current Python modules.
- [x] Replace stale README with scope, commands, measured results and limits.
- [x] Exclude weights, game files, saves, sessions and generated build artifacts.
- [x] Preserve upstream attribution for the JSON helper.
- [ ] Choose a project license with the owner; none is assigned automatically.
- [ ] Review the prepared tree diff and the retained history before publishing.
- [ ] Verify installation in a fresh Python environment and GPU compilation there.
- [ ] Provide documented, downloadable weights before claiming exact experiment reproduction.
- [ ] Review/push/merge separately. This cleanup does not publish anything.

The source repository was already public when inspected. Local uncommitted
changes are not thereby on GitHub. The cleaned tree preserves existing history;
secret scanning is best effort and is not a guarantee about every possible
credential or historical personal datum. No history rewrite is performed.
