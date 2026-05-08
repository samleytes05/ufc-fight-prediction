# Modeling Overview

## Purpose
This document closes the modeling stage of the project. It summarizes the validated dataset, the chronological evaluation framework, the model-selection decisions, and the evidence files preserved under `outputs/modeling/`.

## Dataset And Validation
- Modeling dataset: `data/historical_backfill/ufc_rebuilt_features_scraped.csv`
- Historical result metadata: `data/historical_backfill/ufc_fight_results.csv`
- Event chronology metadata: `data/historical_backfill/historical_event_catalog_scraped.csv`
- Validation method: strict expanding-window walk-forward splits, documented in:
  - `outputs/modeling/baseline_experiments/model_comparison_report.md`
  - `outputs/modeling/pruning/model_comparison_report_2.md`
  - `outputs/modeling/tuning/model_tuning_report.md`

## Baseline Model Results
- Full baseline experiment outputs:
  - `outputs/modeling/baseline_experiments`
- Metrics table:
  - `outputs/modeling/baseline_experiments/metrics_summary.csv`
- Main report:
  - `outputs/modeling/baseline_experiments/model_comparison_report.md`
- Key result:
  - the early strongest raw baseline was `random_forest + all_features`

## Pruning Results
- Pruning outputs:
  - `outputs/modeling/pruning`
- Feature subset definitions:
  - `outputs/modeling/pruning/feature_sets_used.md`
- Metrics table:
  - `outputs/modeling/pruning/metrics_summary.csv`
- Main report:
  - `outputs/modeling/pruning/model_comparison_report_2.md`
- Locked baselines coming out of pruning:
  - primary tree benchmark: `random_forest + pruned_combined`
  - secondary linear benchmark: `logistic_regression + pruned_combined`

## Dual-Model Refinement Results
- Refinement outputs:
  - `outputs/modeling/refinement`
- Main report:
  - `outputs/modeling/refinement/model_refinement_report_2.md`
- Candidate feature outcomes:
  - `outputs/modeling/refinement/candidate_feature_results.csv`
- Key finding:
  - no added feature improved the locked two-model baseline strongly enough to replace `pruned_combined`

## Tuning And Calibration Results
- Tuning outputs:
  - `outputs/modeling/tuning`
- Main report:
  - `outputs/modeling/tuning/model_tuning_report.md`
- Best parameter summary:
  - `outputs/modeling/tuning/best_params.json`
- Key decisions supported by those files:
  - production model: tuned logistic regression + Platt calibration
  - benchmark model: random forest + pruned_combined
  - RF still remained the stronger raw tree benchmark on uncalibrated probability quality, but calibrated tuned logistic became the preferred production scorer

## Prediction Behavior Analysis
- Behavior-analysis outputs:
  - `outputs/modeling/prediction_behavior`
- Main report:
  - `outputs/modeling/prediction_behavior/prediction_behavior_report.md`
- Core merged table:
  - `outputs/modeling/prediction_behavior/prediction_behavior_table.csv`
- Agreement/disagreement evidence:
  - `outputs/modeling/prediction_behavior/agreement_summary.csv`
  - `outputs/modeling/prediction_behavior/disagreement_summary.csv`
- Decision-rule evidence:
  - `outputs/modeling/prediction_behavior/decision_rule_summary.csv`

Key supported findings:
- logistic / RF agreement rate: `84.3%`
  - source: `outputs/modeling/prediction_behavior/agreement_summary.csv`
- agreement accuracy: `63.4%`
  - source: `outputs/modeling/prediction_behavior/agreement_summary.csv`
- disagreement accuracy: logistic `53.5%`, RF `46.5%`
  - source: `outputs/modeling/prediction_behavior/disagreement_summary.csv`
- logistic confidence `>= 60%` rule: `51.4%` coverage, `68.4%` accuracy, `log_loss 0.618`
  - source: `outputs/modeling/prediction_behavior/decision_rule_summary.csv`
- agree + logistic confidence `>= 70%` rule: `9.0%` coverage, `79.1%` accuracy
  - source: `outputs/modeling/prediction_behavior/decision_rule_summary.csv`

## Final Modeling Decisions
- Production model:
  - tuned logistic regression + Platt calibration
  - supported by `outputs/modeling/tuning/model_tuning_report.md` and `outputs/modeling/tuning/best_params.json`
- Secondary benchmark model:
  - random forest + pruned_combined
  - supported by `outputs/modeling/pruning/model_comparison_report_2.md` and `outputs/modeling/prediction_behavior/prediction_behavior_report.md`
- RF should not override logistic when models disagree
  - supported by `outputs/modeling/prediction_behavior/disagreement_summary.csv`
- model agreement is useful as a confidence filter
  - supported by `outputs/modeling/prediction_behavior/agreement_summary.csv` and `outputs/modeling/prediction_behavior/decision_rule_summary.csv`
- logistic confidence is useful for selective decision rules
  - supported by `outputs/modeling/prediction_behavior/decision_rule_summary.csv`

## Phase Transition
Phase 2 is complete enough to move to Phase 3. The modeling stage now has:
- a validated dataset and documented walk-forward evaluation history
- a locked production scorer
- a locked benchmark model
- a documented prediction-behavior profile to guide downstream strategy work

Phase 3 should build on these locked modeling artifacts rather than reopen feature or validation changes unless new evidence demands it.
