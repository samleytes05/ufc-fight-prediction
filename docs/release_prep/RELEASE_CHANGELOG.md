# Phase 4 Changelog

## Files Renamed
- `outputs/strategy/PHASE3_RESULTS.md` -> `docs/strategy/BETTING_STRATEGY_RESULTS.md`
- `outputs/strategy/FINAL_PHASE3_STATUS.md` -> `docs/strategy/BETTING_STRATEGY_STATUS.md`
- `outputs/strategy/PHASE3_CLOSEOUT.md` -> `docs/strategy/BETTING_STRATEGY_CLOSEOUT.md`
- `outputs/strategy/PHASE3_FILE_INVENTORY.md` -> `docs/release_prep/RELEASE_FILE_INVENTORY.md`

## Files Moved
- moved strategy narrative docs from `outputs/strategy/` into `docs/strategy/`
- moved the strategy file inventory into `docs/release_prep/`

## Files Created
- `docs/RELEASE_CLEANUP_PLAN.md`
- `docs/release_prep/RELEASE_CHANGELOG.md`
- `docs/release_prep/PUBLIC_RELEASE_TODO.md`
- `docs/phase1_pipeline/.gitkeep`
- `docs/phase2_modeling/.gitkeep`
- `.gitignore`

## Files Deleted
- no non-junk project files deleted in this step
- only generated cache files should be removed during the junk cleanup step

## Files Left Untouched Intentionally
- model code under `src/model.py` and `src/modeling_*.py`
- feature logic under `src/features.py`
- strategy thresholds and backtest logic under `src/strategy/`
- data files and outputs beyond the doc moves
- private information and local-path cleanup

## Private/Public Cleanup Still Pending
- remove private and local paths from public-facing docs and outputs
- decide which data files can be published
- replace full data with sample data where needed
- review scraper-source terms before publishing scraped artifacts
- remove personal or private live tracking details before public release
