# Betting Strategy Results


## Experiment 1: Initial Phase 3 Strategy Suite
Date: 2026-04-24

### Strategy Definition
- edge threshold: strategy batch from 0.00 to 0.05+
- confidence threshold: none, 0.60, and 0.70 variants
- filters: +150 A-side underdog cap, optional RF agreement filter
- stake: flat 100

### Data Coverage
- total fights: 4247
- fights with odds: 3786
- % coverage: 89.1%

### Results
- ROI: 0.1696
- total profit: 1051.62
- total bets: 62
- win rate: 0.6774
- max drawdown: -356.96

### Observations
- what worked: highest ROI strategy in this batch was `s4_edge_gt_0p05_p_ge_0p70`
- what failed: lower-edge or low-coverage variants may still be noisy and need fold review
- anomalies: historical odds coverage is limited to the portion matched through the legacy odds file

### Next Adjustments
- test edge bucket refinements and confidence filters against fold stability
- consider fractional Kelly only after the flat-stake baseline is accepted
## Experiment 2: Final Strategy Validation
Date: 2026-04-24

- strategy tested: `V2_Core`
- ROI: 0.1121
- bets: 262
- drawdown: 1117.29
- fold stability: 80.0% profitable folds
- decision: ACCEPT

- strategy tested: `V2_Fav`
- ROI: 0.1213
- bets: 182
- drawdown: 597.96
- fold stability: 100.0% profitable folds
- decision: REJECT

## Experiment 3: Bet Sizing Extension
Date: 2026-04-26

- locked entry rule: `V2_Core`
- benchmark sizing: `flat_100`
- conservative sizing: `pct_bankroll_1pct`
- aggressive sizing: `kelly_capped_2pct`

### Results
- flat `$100`: final bankroll `12937.18`, total profit `2937.18`
- `1%` bankroll: final bankroll `13257.01`, total profit `3257.01`
- Kelly capped `2%`: final bankroll `17166.75`, total profit `7166.75`
- full Kelly: rejected for bankroll blow-up and extreme stake spikes
- half Kelly: rejected for extreme stake spikes and unstable drawdown

### Decision
- safer production choice: `1% bankroll`
- aggressive option: Kelly capped `2%`
- flat `$100` remains the benchmark for reporting continuity

## Experiment 4: Live Shadow Tracking Integration
Date: 2026-04-26

- integrated `src/strategy/live_shadow_tracking.py` into the normal live refresh command
- shadow outputs now include V2_Core flags, benchmark/conservative/aggressive stake columns, and expected profit columns
- live tracking output:
  - `outputs/strategy/live_tracking/live_bet_tracking.csv`
  - `outputs/strategy/live_tracking/latest_live_recommendations.csv`
- post-card settlement support added:
  - refresh completed results
  - settle realized profits
  - compare actual vs expected profit
  - track CLV with `odds_at_pick`, `closing_odds`, `line_movement`, and `beat_closing_line_flag`

### Current Mode
- live shadow tracking only
- no automatic real-money execution
- final live sizing decision deferred until 10-20 shadow-tracked events are completed

## Phase 3 Closeout

- final accepted strategy: `V2_Core`
- accepted rule:
  - `edge_A > 0.04`
  - `p_model_A >= 0.65`
- rejected / secondary strategy:
  - `V2_Fav` rejected only because sample size `< 200`
- historical backtest result:
  - `262` bets
  - ROI `11.21%`
  - win rate `60.31%`
  - `80%` profitable folds
- sizing result:
  - flat `$100` benchmark
  - `1%` bankroll conservative
  - Kelly capped `2%` aggressive
- live status:
  - shadow tracking implemented
  - no automatic betting
  - live evidence currently too early to judge
