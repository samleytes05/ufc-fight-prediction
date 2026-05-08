# AGENTS.md

## Project Overview
This project builds a machine learning model to predict UFC fight outcomes.

The pipeline:
1. Data preparation and feature engineering in `src/`
2. Differential feature construction
3. Model training with time-based splits

---

## Key Files

- src/features.py  
  → Feature engineering logic (target location for improvements)

- src/model.py  
  → Model definitions

- src/train.py  
  → Training pipeline

---

## Data

Primary self-owned training data is in:
- data/historical_backfill/ufc_fight_results.csv
- data/historical_backfill/ufc_fight_stats.csv
- data/historical_backfill/ufc_fighter_details.csv

Primary self-owned rebuilt training dataset is:
- data/historical_backfill/ufc_rebuilt_features_scraped.csv

Legacy imported CSVs may exist only under legacy/reference locations.

---

## Modeling Rules

- Use differential features (A vs B)
- Avoid using raw A/B features unless necessary
- No data leakage (no future fights)
- Respect time-based splits

---

## Preferred Workflow

When making changes:
1. Modify or create logic in `src/`
2. Prefer script-based, reproducible workflows over notebooks
3. Keep functions modular and readable

---

## Feature Engineering Guidelines

Focus on:
- Efficiency metrics (accuracy, ratios)
- Per-round normalization
- Last 3 fights (recent form)
- Differential features

Avoid:
- Using future data
- Duplicating existing features

---

## Commands

Environment is already set up with:
- virtual environment: vmathv

Run training:
python src/train.py

---

## Permissions

Allowed without asking:
- Read files
- Edit Python files in src/
- Add new features

Ask before:
- Installing new packages
- Deleting files
- Major refactors

---

## Goal

Improve model performance through better feature engineering and clean pipeline structure.
