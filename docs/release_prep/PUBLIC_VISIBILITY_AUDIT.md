# Public Visibility Audit

## Summary
- Total tracked files: 54
- Total ignored files/folders currently present: 15
- Risk status: PASS

This audit uses `git ls-files` as the source of truth for what would be public if the repository is pushed as-is.

## Public Files by Category

### Root Docs/Config
- `.gitignore`
- `AGENTS.md`
- `EXPERIMENT_LOG.md`
- `MODELING_OVERVIEW.md`
- `PIPELINE_OVERVIEW.md`
- `README.md`
- `STRATEGY_SPEC.md`
- `requirements.txt`

### Src Code
- `src/features.py`
- `src/model.py`
- `src/modeling_baseline.py`
- `src/modeling_prediction_behavior.py`
- `src/modeling_pruning.py`
- `src/modeling_refinement.py`
- `src/modeling_tuning.py`
- `src/scrapers/__init__.py`
- `src/scrapers/backfill_historical_dataset.py`
- `src/scrapers/build_live_inputs.py`
- `src/scrapers/common.py`
- `src/scrapers/derive_live_features.py`
- `src/scrapers/fetch_completed_results.py`
- `src/scrapers/fetch_current_odds.py`
- `src/scrapers/fetch_fighter_attributes.py`
- `src/scrapers/fetch_fighter_history.py`
- `src/scrapers/fetch_historical_odds.py`
- `src/scrapers/fetch_upcoming_fights.py`
- `src/strategy/backtest_strategies.py`
- `src/strategy/bet_sizing_experiments.py`
- `src/strategy/build_betting_dataset.py`
- `src/strategy/diagnostics.py`
- `src/strategy/evaluate_strategies.py`
- `src/strategy/final_strategy_validation.py`
- `src/strategy/live_shadow_evidence.py`
- `src/strategy/live_shadow_tracking.py`
- `src/strategy/strategy_grid_search.py`
- `src/train.py`

### Tests
- `tests/test_scraper_retry.py`

### Docs
- `docs/RELEASE_CLEANUP_PLAN.md`
- `docs/phase1_pipeline/.gitkeep`
- `docs/phase2_modeling/.gitkeep`
- `docs/release_prep/PUBLIC_RELEASE_TODO.md`
- `docs/release_prep/RELEASE_CHANGELOG.md`
- `docs/release_prep/RELEASE_FILE_INVENTORY.md`
- `docs/release_prep/PUBLIC_VISIBILITY_AUDIT.md`
- `docs/strategy/BETTING_STRATEGY_CLOSEOUT.md`
- `docs/strategy/BETTING_STRATEGY_RESULTS.md`
- `docs/strategy/BETTING_STRATEGY_STATUS.md`

### Sample Outputs
- `sample_outputs/strategy/bet_sizing_summary_sample.csv`
- `sample_outputs/strategy/edge_bucket_analysis_sample.csv`
- `sample_outputs/strategy/final_strategy_summary_sample.csv`
- `sample_outputs/strategy/strategy_results_sample.csv`

### Data
- `data/future_fights_template.csv`

### Archive
- `archive/legacy_tools/merge_master.py`
- `archive/legacy_tools/merge_master_physical_odds.py`
- `archive/phase1_audits/compare_historical_datasets.py`

### Notebooks
- None tracked

### Other
- None

## Strategy Visibility Check
- `src/strategy/`: PASS
- `docs/strategy/`: PASS
- `STRATEGY_SPEC.md`: PASS
- `sample_outputs/strategy/`: PASS
- `README.md`: PASS

Overall result: PASS. All requested strategy/public-facing artifacts are tracked and publicly visible.

## Risk Flags
- No tracked files currently match the requested sensitive/private filename patterns.
- No tracked `.env`, `.pem`, `.key`, `.pkl`, or `.joblib` files were found.
- No tracked public data/output CSVs remain under `data/` except the small safe template file `data/future_fights_template.csv`.
- No tracked files currently contain local absolute path references matching `C:\Users\`, `/Users/`, `/mnt/data/`, `nonAdmin`, or `Desktop`.

## Large Files
- None. No tracked files exceed 5 MB.

## Local Path / Secret String Scan

### Local Path Findings
- None found in tracked files.

### Secret-Like String Findings
- Reviewed text hits for `TOKEN` in scraper code. These are ordinary variable names and parsing logic, not credentials.
- No credential-like tracked content was identified from the requested string scan.

## Ignored Files and Folders Check
- Confirmed ignored and not tracked:
  - `data/betting_ready.csv`
  - `data/completed_results_scraped.csv`
  - `data/current_odds_raw.csv`
  - `data/current_odds_scraped.csv`
  - `data/fighter_attributes_scraped.csv`
  - `data/fighter_recent_history_scraped.csv`
  - `data/future_card_shadow_report.csv`
  - `data/future_fight_predictions.csv`
  - `data/historical_backfill/`
  - `data/legacy_betting/`
  - `data/live_tracking.csv`
  - `data/upcoming_fights.csv`
  - `data/upcoming_fights_scraped.csv`
  - `data/walk_forward_prediction_export.csv`
  - `outputs/`

- `.gitignore` now protects:
  - `data/`
  - `data/legacy_betting/`
  - `outputs/`
  - `*.csv`
  - while allowing:
  - `data/README.md`
  - `data/*template*.csv`
  - `sample_outputs/strategy/*.csv`

## Recommended Actions Before Sharing
- `KEEP` strategy code, strategy docs, root project docs, and `sample_outputs/strategy/*.csv` as public repo artifacts.
- `KEEP` `data/future_fights_template.csv` as the only tracked `data/` CSV because it is a lightweight template.
- `KEEP PRIVATE` the ignored `data/` CSVs and `outputs/` artifacts unless you later replace them with explicit public samples.
- `OPTIONAL` stage and commit this cleanup before pushing so GitHub reflects the reduced public file set.
