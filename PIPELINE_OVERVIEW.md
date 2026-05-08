# Pipeline Overview

## Purpose
This project is a UFC fight prediction system with a self-owned historical pipeline, a validated modeling stage, and a live-input workflow that will feed Phase 3 strategy work.

The system has three main parts:
- historical backfill pipeline
- feature engineering / modeling pipeline
- live prediction pipeline

They connect in this order:
- scrapers build historical raw data
- `src/features.py` rebuilds the modeling dataset from that raw data
- modeling runners under `src/modeling_*.py` evaluate and document model behavior
- live scrapers build current pre-fight inputs for future predictions

## High-Level Architecture
### 1. Historical Backfill Pipeline
This pipeline scrapes historical UFC data from public sources and stores it in a self-owned archive under `data/historical_backfill/`.

It is responsible for:
- completed fight results
- fighter-by-fight stat totals
- event dates and chronology
- fighter profile enrichment
- historical physical features

### 2. Feature Engineering And Modeling Pipeline
This pipeline converts the historical raw archive into the final modeling table, then evaluates models on strict chronological splits.

It is responsible for:
- cleaning raw fight totals
- building fighter histories
- computing career and recent-form aggregates
- generating matchup differential features
- computing pre-fight ELO
- writing the training-ready dataset

### 3. Live Prediction Pipeline
This pipeline scrapes current upcoming fights and live enrichment data, then builds prediction-ready rows for future matchups.

It is responsible for:
- upcoming card matchups
- current odds
- fighter attributes
- recent fighter history
- live pre-fight feature derivation

## Data Flow
### Historical Flow
1. UFCStats historical events are scraped into raw results and fight-level stat totals.
2. Fighter profile data is scraped and normalized.
3. Historical physical features are derived by joining profiles to fight dates.
4. The feature pipeline rebuilds the final modeling dataset.
5. Output: `data/historical_backfill/ufc_rebuilt_features_scraped.csv`

### Live Flow
1. Upcoming fights are scraped from current event sources.
2. Current odds are scraped and matched to upcoming matchups.
3. Fighter attributes and recent fighter history are scraped.
4. Live features are derived into a prediction-ready table.
5. Output: `data/upcoming_fights.csv`

## Key Datasets
### Training / Historical
- `data/historical_backfill/ufc_fight_results.csv`
  - one row per fight
  - contains event, bout, outcome, method, round, and time
  - defines target orientation and fight chronology

- `data/historical_backfill/ufc_fight_stats.csv`
  - one row per fighter per fight
  - contains fight-level totals, not per-round rows
  - used to compute all rolling and career features

- `data/historical_backfill/historical_event_catalog_scraped.csv`
  - event-level metadata with event dates and locations
  - provides the chronological backbone

- `data/historical_backfill/historical_fighter_profiles_scraped.csv`
  - fighter profile enrichment from public profile sources
  - includes DOB, height, reach, stance, and related identity fields where available

- `data/historical_backfill/historical_physical_features_scraped.csv`
  - fight-level physical enrichment derived from profiles + event dates
  - contains age, height, reach, and physical-difference features

- `data/historical_backfill/ufc_rebuilt_features_scraped.csv`
  - final training-ready modeling dataset
  - used by the modeling runners and the live derivation workflow

### Live
- `data/upcoming_fights.csv`
  - final live prediction input table
  - contains upcoming matchup rows with as many pre-fight features as can be derived live

- `data/current_odds_scraped.csv`
  - consensus or cleaned current odds for upcoming fights
  - used for live edge and strategy evaluation

- `data/fighter_attributes_scraped.csv`
  - current fighter profile attributes used for live enrichment

- `data/fighter_recent_history_scraped.csv`
  - recent completed fights for fighters on the current live slate
  - supports recent-form and observed history features

## Feature Pipeline
The feature pipeline is implemented in `src/features.py`.

It works from fight-level fighter totals, not per-round rows.

That is sufficient because the model only needs:
- fight result and chronology
- striking, takedown, submission, control, and KD totals
- fight duration / rounds fought for normalization

Main feature groups:
- career aggregates
  - cumulative historical totals and rates before each fight
- last-3 aggregates
  - recent-form summaries from the fighter’s last three fights
- differential features
  - matchup features defined as Fighter A minus Fighter B
- physical-difference features
  - `age_diff`, `height_diff`, `reach_diff`
- ELO features
  - pre-fight ratings and ELO-based opponent-strength context

Important design rules:
- no future leakage
- strict chronological ordering
- pre-fight state only
- no per-round dependency in the default historical backfill

## ELO System
The ELO system is part of the feature pipeline.

Rules:
- all fighters start at `1500`
- each fight uses pre-fight ratings only
- ratings update only after the fight result
- chronological order is required for correctness

The ELO chain has been audited for:
- correct initialization
- correct winner/loser direction
- no chronology backtracking
- no future leakage into prior fights

## Independence Design
Training is now self-owned:
- historical results come from the scraper pipeline
- historical stats come from the scraper pipeline
- fighter detail enrichment comes from the scraper pipeline
- the rebuilt training dataset is derived from self-owned files only

Training does not rely on downloaded legacy CSVs anymore.

Important exception:
- betting / EV analysis still has a separate historical odds dependency
- legacy betting files are isolated under `data/legacy_betting/`
- this does not affect training-data independence

## How To Run
### Historical Backfill
```powershell
vmathv\Scripts\python.exe -m src.scrapers.backfill_historical_dataset --max-events 1000 --skip-odds
```

### Rebuild Features
```powershell
vmathv\Scripts\python.exe src\features.py
```

### Model Baseline
```powershell
vmathv\Scripts\python.exe src\train.py
```

### Modeling Tuning
```powershell
vmathv\Scripts\python.exe src\modeling_tuning.py
```

### Live Pipeline
```powershell
vmathv\Scripts\python.exe -m src.scrapers.build_live_inputs
```

## Known Limitations
- historical physical data is incomplete for some older fighters, especially reach and DOB-derived age
- historical odds coverage is still low, so betting-side independence is not complete
- scraper reliability depends on public site structure remaining stable
- current public-source dependencies are mainly UFCStats and Best Fight Odds

## Durable Modeling Outputs
- `outputs/modeling/baseline/`
- `outputs/modeling/baseline_experiments/`
- `outputs/modeling/pruning/`
- `outputs/modeling/refinement/`
- `outputs/modeling/tuning/`
- `outputs/modeling/prediction_behavior/`

## Project Phases
- Phase 1: data pipeline
  - completed for training independence
  - historical archive, feature rebuild, and ELO are self-owned and validated

- Phase 2: modeling and decision-quality validation
  - completed
  - baseline comparison, pruning, refinement, tuning/calibration, and prediction-behavior analysis are archived under `outputs/modeling/`
  - final modeling decision is documented in `MODELING_OVERVIEW.md`

- Phase 3: strategy / betting system
  - next phase
  - use the locked production and benchmark models for odds-driven strategy work
