# Betting Strategy Specification

## Purpose
This document defines the official Version 1 deployable betting system for the project. It locks the model, feature set, calibration, staking rule, and reporting outputs so future changes can be measured against a stable baseline.

This project is for research and educational purposes only.
It does not provide betting or financial advice.
Historical performance does not guarantee future results.
No automatic wagering or real-money execution is implemented.

## Version 1 System
- Model: Logistic Regression
- Calibration: Sigmoid / Platt scaling
- Feature set: Current baseline differential feature set with physical differences enabled (`include_physical = True`)
- Betting rule: V2_Core
  - `edge_A > 0.04`
  - `p_model_A >= 0.65`
- Shadow sizing profiles:
  - Flat `$100` benchmark
  - `1%` bankroll conservative
  - Kelly capped at `2%` bankroll aggressive

## Decision Rule
For each future fight:
1. Score Fighter A win probability with the calibrated Logistic Regression model.
2. Convert Fighter A market odds into implied probability.
3. Compute `edge_A = p_model_A - implied_prob_A`.
4. Bet only if:
   - `edge_A > 0.04`
   - `p_model_A >= 0.65`

## Supporting Evidence

### Final Validation
- Historical strategy validation completed.
- Accepted live rule: `V2_Core`
- Rule: `edge_A > 0.04` and `p_model_A >= 0.65`
- Historical validation bets: `262`
- Win rate: `0.603`
- Total profit: `+$2937.18`
- ROI: `+0.112`
- Profitable folds: `80%`
- Max drawdown: `-$1117.29`

### Sizing Comparison
- Flat `$100`: final bankroll `12937.18`, profit `+$2937.18`
- `1%` bankroll: final bankroll `13257.01`, profit `+$3257.01`
- Kelly capped `2%`: final bankroll `17166.75`, profit `+$7166.75`
- Practical production preference:
  - `1%` bankroll = safer default
  - Kelly capped `2%` = aggressive option

### Model Lock
- Logistic Regression + Platt calibration remains the locked scorer for Phase 3 deployment.
- Do not reopen feature or model selection during live shadow mode.
- Do not change model or feature logic unless new data is added and the full benchmark stack is rerun.

Conclusion:
- The current Logistic Regression system remains the locked production scorer, and `V2_Core` is the locked live betting rule.

## Shadow-Mode Reporting
Future-fight shadow reports should include:
- `fighter_A`
- `fighter_B`
- `p_model_A`
- `implied_prob_A`
- `edge_A`
- `odds_A`
- `v2_core_bet_flag`
- `stake_flat_100`
- `stake_bankroll_1pct`
- `stake_kelly_capped_2pct`
- `expected_profit_flat_100`
- `expected_profit_bankroll_1pct`
- `expected_profit_kelly_capped_2pct`
- `sizing_profile`
- `model_confidence_tier`

Implementation path:
- Main prediction export: `data/future_fight_predictions.csv`
- Shadow strategy export: `data/future_card_shadow_report.csv`
- Live tracking log: `outputs/strategy/live_tracking/live_bet_tracking.csv`
- Current live recommendations: `outputs/strategy/live_tracking/latest_live_recommendations.csv`
- Phase 3 results doc: `docs/strategy/BETTING_STRATEGY_RESULTS.md`
- Phase 3 status doc: `docs/strategy/BETTING_STRATEGY_STATUS.md`
- Phase 3 closeout doc: `docs/strategy/BETTING_STRATEGY_CLOSEOUT.md`

## Public Artifacts
- Strategy logic: `src/strategy/`
- Strategy documentation: `docs/strategy/`
- Sample outputs: `sample_outputs/strategy/`

Full outputs and live tracking are excluded from the public-facing repo contents for size and privacy reasons.

## Live Mode
- recommendations only
- no automated real-money execution

## Evaluation Metrics
- ROI
- win rate
- max drawdown
- profitable fold rate
- CLV / closing-line proxy

## Operational Notes
- Do not use odds as model features.
- Keep evaluation strictly chronological.
- Profit and ROI are the primary deployment metrics, not just AUC or accuracy.
- After each completed UFC card:
  - refresh completed results
  - settle live tracking
  - fill realized profit columns
  - update CLV fields: `odds_at_pick`, `closing_odds`, `line_movement`, `beat_closing_line_flag`
- Re-benchmark against this spec before changing model class, feature set, or strategy rule.
- No model or feature changes should be made unless new data is added and the strategy is revalidated.
