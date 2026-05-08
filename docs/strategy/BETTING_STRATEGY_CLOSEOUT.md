# Phase 3 Closeout

## Objective
Use locked model probabilities to identify positive-EV betting opportunities.

## Final Accepted Strategy
V2_Core:
- `edge_A > 0.04`
- `p_model_A >= 0.65`

## Historical Validation
- odds-covered fights: `3786`
- accepted strategy bets: `262`
- ROI: `11.21%`
- win rate: `60.31%`
- max drawdown: `1117.29`
- profitable folds: `80%`

## Sizing Decision
- benchmark: flat `$100`
- safer live option: `1%` bankroll
- aggressive option: Kelly capped `2%`

## Live Shadow System
- recommendation refresh: pre-card shadow report update through the normal live build flow
- bet tracking: append-only live recommendation log in `outputs/strategy/live_tracking/live_bet_tracking.csv`
- settlement: post-card results update fills realized outcome and profit columns
- CLV tracking: stores `odds_at_pick`, `closing_odds`, `line_movement`, and `beat_closing_line_flag`
- evidence report: summarizes pending bets, settled bets, ROI, CLV, and event-level performance

## Current Limitation
Live evidence is not yet meaningful because future events must occur over time.

## Next Phase
Phase 4:
- prune repo
- remove private/raw artifacts
- clean docs
- create public sample data
- prepare public release
