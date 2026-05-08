# Betting Strategy Status

## Locked Model
- Logistic Regression + Platt calibration

## Locked Strategy
- V2_Core
- edge_A > 0.04
- p_model_A >= 0.65

## Sizing Profiles
- Flat $100 benchmark
- 1% bankroll conservative
- Kelly capped 2% aggressive

## Preferred Sizing Decision Path
- safer production choice: 1% bankroll
- aggressive option: Kelly capped 2%
- keep flat $100 as the benchmark reference line

## Current Deployment Mode
- live shadow tracking
- no automatic real-money execution

## Live Shadow Workflow Commands
- pre-card refresh: `vmathv\Scripts\python.exe -m src.scrapers.build_live_inputs --bankroll 10000`
- post-card settlement: `vmathv\Scripts\python.exe -m src.scrapers.build_live_inputs --include-completed-results --settle-live-tracking --bankroll 10000`
- evidence report: `vmathv\Scripts\python.exe src\strategy\live_shadow_evidence.py`

## Post-Card Workflow
1. Refresh completed fight results
2. Settle live_bet_tracking.csv with actual outcomes
3. Update realized profit vs expected profit
4. Record CLV fields: odds_at_pick, closing_odds, line_movement, beat_closing_line_flag

## Remaining Final Steps
1. Run shadow mode over future events
2. Record recommendations and results after each event
3. Compare live ROI vs backtest ROI
4. Track CLV / closing-line proxy if available
5. Decide final production sizing after sufficient live sample
6. Freeze final strategy documentation

## Live Shadow Evidence Tracking
- evidence report path: `outputs/strategy/live_tracking/LIVE_SHADOW_EVIDENCE.md`
- pending bets path: `outputs/strategy/live_tracking/pending_live_bets.csv`
- settled bets path: `outputs/strategy/live_tracking/settled_live_bets.csv`
- performance summary path: `outputs/strategy/live_tracking/live_performance_summary.csv`
- CLV summary path: `outputs/strategy/live_tracking/live_clv_summary.csv`
- evidence report command: `vmathv\Scripts\python.exe src\strategy\live_shadow_evidence.py`
