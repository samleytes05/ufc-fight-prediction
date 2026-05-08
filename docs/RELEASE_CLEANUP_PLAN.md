# Phase 4 Cleanup Plan

## Scope
Prepare the repository for public-release cleanup by organizing documentation, standardizing Phase 3 doc locations, refreshing top-level guides, and documenting what should later be kept, archived, or removed.

This plan is non-destructive by design:
- no model logic changes
- no feature logic changes
- no strategy threshold changes
- no scraper behavior changes
- no private-information removal yet
- no deletion of non-junk project files in this step

## Files To Rename
- `outputs/strategy/PHASE3_RESULTS.md`
  - rename to `docs/strategy/BETTING_STRATEGY_RESULTS.md`
- `outputs/strategy/FINAL_PHASE3_STATUS.md`
  - rename to `docs/strategy/BETTING_STRATEGY_STATUS.md`
- `outputs/strategy/PHASE3_CLOSEOUT.md`
  - rename to `docs/strategy/BETTING_STRATEGY_CLOSEOUT.md`
- `outputs/strategy/PHASE3_FILE_INVENTORY.md`
  - rename to `docs/release_prep/RELEASE_FILE_INVENTORY.md`

## Files To Move
- move Phase 3 narrative docs from `outputs/strategy/` into `docs/strategy/`
- keep generated CSVs, plots, and live-tracking artifacts under `outputs/strategy/`

## Files To Keep
- core code under `src/`
- top-level reference docs:
  - `README.md`
  - `PIPELINE_OVERVIEW.md`
  - `MODELING_OVERVIEW.md`
  - `EXPERIMENT_LOG.md`
  - `STRATEGY_SPEC.md`
- all final strategy outputs in `outputs/strategy/reports/`, `outputs/strategy/equity_curves/`, and `outputs/strategy/live_tracking/`

## Files To Archive
- intermediate Phase 3 experiment artifacts already identified in the Phase 3 file inventory
- older redundant diagnostics and plots, but only as documented guidance for a later cleanup pass

## Files To Delete Only If Clearly Generated Junk
- `__pycache__/`
- `*.pyc`

No other deletions should occur in this step.

## Documentation Updates Needed
- update `STRATEGY_SPEC.md` title to `# Betting Strategy Specification`
- update `STRATEGY_SPEC.md` cross-references to the new Phase 3 doc locations
- refresh `README.md` so it clearly explains Phases 1-4 and the current live shadow workflow
- append a Phase 4 start entry to `EXPERIMENT_LOG.md`
- create `docs/release_prep/RELEASE_CHANGELOG.md`
- create `docs/release_prep/PUBLIC_RELEASE_TODO.md`

## Safety Notes
- do not delete data, outputs, or model artifacts yet
- do not remove private or local references yet
- do not overwrite moved files if destination content exists; merge carefully instead
- preserve generated evidence files in place under `outputs/strategy/`
