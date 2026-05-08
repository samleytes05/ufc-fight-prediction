# UFC Modeling Project

## Project Goal
Build a self-owned UFC fight prediction pipeline, train calibrated fight outcome models, use model probabilities to test positive-EV betting strategies, and transition to live shadow tracking without automatic wagering.

This project is for research and educational purposes only.
It does not provide betting or financial advice.
Historical performance does not guarantee future results.
No automatic wagering or real-money execution is implemented.

## Phase 1: Data Pipeline
Phase 1 established the self-owned data pipeline used for both training and live feature generation.

It includes:
- historical scraping
- feature generation
- live input preparation
- a self-owned training dataset
- no reliance on external CSVs for model features

Core references:
- `PIPELINE_OVERVIEW.md`
- `src/features.py`
- `src/scrapers/build_live_inputs.py`

## Phase 2: Modeling
Phase 2 locked the modeling stack using strict walk-forward validation.

Key outcomes:
- walk-forward validation is the primary benchmark
- tuned Logistic Regression + Platt calibration was selected as the production scorer
- Random Forest remains the benchmark / diagnostic model
- the feature space is locked
- no more tuning should be done unless new data is added

Core references:
- `MODELING_OVERVIEW.md`
- `src/train.py`
- `src/model.py`
- `src/modeling_*.py`

## Phase 3: Betting Strategy
Phase 3 used locked model probabilities and market odds to validate betting rules without changing the model.

Key outcomes:
- odds are used only after predictions
- `V2_Core` was accepted:
  - `edge_A > 0.04`
  - `p_model_A >= 0.65`
- sizing profiles:
  - flat `$100` benchmark
  - `1%` bankroll conservative
  - Kelly capped `2%` aggressive
- live shadow tracking is implemented
- no automatic real-money betting is implemented

Core references:
- `STRATEGY_SPEC.md`
- `docs/strategy/BETTING_STRATEGY_RESULTS.md`
- `docs/strategy/BETTING_STRATEGY_STATUS.md`
- `docs/strategy/BETTING_STRATEGY_CLOSEOUT.md`

## Betting Strategy (Phase 3)
Final strategy:
- `V2_Core`
- `edge_A > 0.04`
- `p_model_A >= 0.65`

Sizing:
- flat `$100` benchmark
- `1%` bankroll recommended
- Kelly capped at `2%`

Historical performance:
- about `11%` ROI
- about `60%` win rate
- `262` bets
- strong fold stability

Note:
- Full raw outputs are not included in the public-facing repo layout.
- Clean representative examples live under `sample_outputs/strategy/`.

## Main Scripts
Pipeline:
- `src/features.py`
- `src/scrapers/build_live_inputs.py`

Modeling:
- `src/train.py`
- `src/model.py`
- `src/modeling_*.py`

Strategy:
- `src/strategy/build_betting_dataset.py`
- `src/strategy/backtest_strategies.py`
- `src/strategy/strategy_grid_search.py`
- `src/strategy/final_strategy_validation.py`
- `src/strategy/bet_sizing_experiments.py`
- `src/strategy/live_shadow_tracking.py`
- `src/strategy/live_shadow_evidence.py`

## Commands
Rebuild historical features:

```powershell
vmathv\Scripts\python.exe src\features.py
```

Run modeling:

```powershell
vmathv\Scripts\python.exe src\train.py
```

Live pre-card refresh:

```powershell
vmathv\Scripts\python.exe -m src.scrapers.build_live_inputs --bankroll 10000
```

Post-card settlement:

```powershell
vmathv\Scripts\python.exe -m src.scrapers.build_live_inputs --include-completed-results --settle-live-tracking --bankroll 10000
```

Live evidence report:

```powershell
vmathv\Scripts\python.exe src\strategy\live_shadow_evidence.py
```

## Current Status
- Phase 1 complete
- Phase 2 complete
- Phase 3 complete
- Phase 4 in progress: public release cleanup
