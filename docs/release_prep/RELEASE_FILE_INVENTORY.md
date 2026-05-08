# Release File Inventory

## KEEP
- core scripts:
  - `src/strategy/build_betting_dataset.py`
  - `src/strategy/backtest_strategies.py`
  - `src/strategy/strategy_grid_search.py`
  - `src/strategy/final_strategy_validation.py`
  - `src/strategy/bet_sizing_experiments.py`
  - `src/strategy/live_shadow_tracking.py`
  - `src/strategy/live_shadow_evidence.py`
- final docs:
  - `STRATEGY_SPEC.md`
  - `README.md`
  - `docs/strategy/BETTING_STRATEGY_STATUS.md`
  - `docs/strategy/BETTING_STRATEGY_RESULTS.md`
  - `docs/strategy/BETTING_STRATEGY_CLOSEOUT.md`
  - `docs/release_prep/RELEASE_FILE_INVENTORY.md`
- final reports:
  - `outputs/strategy/reports/final_strategy_summary.csv`
  - `outputs/strategy/reports/bet_sizing_summary.csv`
  - `outputs/strategy/live_tracking/LIVE_SHADOW_EVIDENCE.md`
  - `outputs/strategy/live_tracking/live_performance_summary.csv`
  - `outputs/strategy/live_tracking/live_clv_summary.csv`
  - `outputs/strategy/live_tracking/live_backtest_comparison.csv`

## ARCHIVE
- intermediate experiment outputs:
  - `outputs/strategy/strategy_results.csv`
  - `outputs/strategy/strategy_bet_log.csv`
  - `outputs/strategy/fold_results.csv`
  - `outputs/strategy/reports/strategy_grid_results.csv`
  - `outputs/strategy/reports/strategy_grid_top.csv`
- old diagnostic CSVs:
  - `outputs/strategy/reports/agreement_summary.csv`
  - `outputs/strategy/reports/confidence_bucket_summary.csv`
  - `outputs/strategy/reports/edge_bucket_summary.csv`
  - `outputs/strategy/reports/price_bucket_summary.csv`
  - `outputs/strategy/reports/fold_stability_summary.csv`
- redundant plots:
  - old edge / heatmap diagnostic PNGs once final docs are frozen
  - duplicate equity curves that are superseded by final accepted summaries

## DO NOT PUBLISH
- full raw data
- live bankroll tracking details beyond the curated evidence summaries
- local paths
- private outputs
- `.env`
- cookies / session files
- large scraped datasets unless licensing is confirmed
