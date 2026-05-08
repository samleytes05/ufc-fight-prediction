# Experiment Log

This log tracks meaningful model iterations, not every minor experiment.
Update it per session or when a change materially affects the feature set, evaluation method, calibration approach, or how we interpret model quality.

Current end-of-modeling summary:
- `MODELING_OVERVIEW.md`

## Logging Guidelines
- Do not log every small tweak, failed micro-test, or intermediate scratch run.
- Do log new feature groups, major feature changes, evaluation-method changes, calibration changes, and any result that materially changes interpretation.
- Group related experiments into one entry when they belong to the same development step.
- Keep entries short, decision-focused, and comparable across sessions.

## Experiment 1: Notebook Pipeline Migration

Date:
- 2026-04-19

Summary:
- Converted the notebook-based preparation and feature pipeline into reusable Python functions in `src/features.py`.
- Replaced notebook dependency in training with a code-driven feature build path.

Changes:
- Moved data loading, cleaning, stat parsing, fight aggregation, opponent merge, long-format history building, rolling features, differential features, and ELO logic into `src/features.py`.
- Updated `src/train.py` to call the feature pipeline directly.
- Added leakage checks for first-fight history behavior.

Evaluation Method:
- End-to-end dataset build from raw CSVs.
- Single chronological train/test split.

Results:
- Dataset shape: `8473 x 43`
- Logistic Regression: accuracy `0.5929`, ROC AUC `0.6516`, log loss `0.6683`
- XGBoost: accuracy `0.6024`, ROC AUC `0.6165`, log loss `0.6809`

Impact:
- Improved reproducibility and modularity.
- Preserved notebook logic closely.
- No meaningful modeling improvement yet; this was mainly an infrastructure upgrade.

Interpretation:
- The project no longer depends on manual notebook execution for core feature generation.
- The baseline became easier to audit, extend, and validate for leakage.

Decision:
- Keep.
- Code-based feature generation becomes the new baseline workflow.

---

## Experiment 2: Differential Feature Expansion And Pruning

Date:
- 2026-04-20

Summary:
- Added new matchup-style, opponent-adjusted, defensive, and ratio-based features, then pruned them using importance and walk-forward validation.
- Reduced the feature set to a leaner baseline by removing weak or redundant columns.

Changes:
- Added damage-efficiency, net-striking, takedown-path, opponent-adjusted damage, and normalized experience features.
- Added absorbed-based and opponent-adjusted defensive features, including offense-vs-defense ratios and adjusted striking features.
- Ran focused pruning on newly added defensive/opponent-adjusted features.
- Set the default baseline to the pruned `best_of_new_set` feature configuration.

Evaluation Method:
- Single chronological split for initial screening.
- 5-fold walk-forward cross-validation for stability checks.
- Feature usefulness review using Logistic Regression coefficients, XGBoost importance, permutation importance, and correlation analysis.

Results:
- Final baseline dataset shape: `8473 x 38`
- Active feature count: `35`
- Walk-forward Logistic Regression: accuracy `0.5938`, ROC AUC `0.6344`, log loss `0.6769`, Brier `0.2411`
- Walk-forward XGBoost: accuracy `0.5950`, ROC AUC `0.6133`, log loss `0.6810`, Brier `0.2419`

Impact:
- Feature set became smaller, cleaner, and easier to interpret.
- Opponent-adjusted striking and defensive ratio features added useful signal.
- Some earlier gains from larger feature sets did not hold up under walk-forward validation.

Interpretation:
- The model benefits from selective matchup-aware features, but too many overlapping defensive features add noise.
- Walk-forward validation showed that some single-split improvements were mostly split noise rather than robust gains.

Decision:
- Modify and keep.
- The pruned 35-feature set becomes the feature baseline.

---

## Experiment 3: Walk-Forward Evaluation Upgrade

Date:
- 2026-04-20

Summary:
- Replaced the single train/test evaluation as the primary benchmark with expanding-window walk-forward cross-validation.
- Added fold-by-fold reporting and average metrics.

Changes:
- Implemented expanding-window walk-forward folds in `src/train.py`.
- Preserved chronological ordering and separated training from future evaluation.
- Added comparison between walk-forward averages and the old single-split reference.

Evaluation Method:
- 5-fold walk-forward cross-validation with strict time ordering.
- Single chronological split retained only as a reference point.

Results:
- Walk-forward metrics were consistently weaker than single-split metrics in ROC AUC and log loss.
- This materially changed how feature improvements were interpreted.

Impact:
- Reduced risk of over-interpreting one favorable split.
- Made the baseline more realistic for forward-looking prediction use.
- Lowered apparent model quality compared with earlier split-based impressions.

Interpretation:
- The model is directionally useful, but performance is less stable than a single split suggested.
- Future development should prioritize walk-forward results over holdout-only gains.

Decision:
- Keep.
- Walk-forward evaluation becomes the primary development benchmark.

---

## Experiment 4: Probability Calibration

Date:
- 2026-04-20

Summary:
- Added time-safe probability calibration inside each walk-forward fold.
- Compared uncalibrated, sigmoid, and isotonic calibration for both Logistic Regression and XGBoost.

Changes:
- Added inner chronological calibration splits within each walk-forward fold.
- Tested sigmoid / Platt scaling and isotonic calibration without future leakage.
- Added Brier score and calibration-bin reporting.

Evaluation Method:
- 5-fold outer walk-forward evaluation.
- Within each outer fold, a later slice of the training window was reserved for calibration only.

Results:
- Logistic Regression uncalibrated: accuracy `0.5886`, ROC AUC `0.6266`, log loss `0.6950`, Brier `0.2481`
- Logistic Regression sigmoid: accuracy `0.5990`, ROC AUC `0.6266`, log loss `0.6630`, Brier `0.2353`
- Logistic Regression isotonic: accuracy `0.5943`, ROC AUC `0.6242`, log loss `0.7410`, Brier `0.2351`
- XGBoost uncalibrated: accuracy `0.5915`, ROC AUC `0.6000`, log loss `0.6987`, Brier `0.2485`
- XGBoost sigmoid: accuracy `0.5837`, ROC AUC `0.6000`, log loss `0.6706`, Brier `0.2389`

Impact:
- Sigmoid calibration materially improved probability quality for both models.
- Logistic Regression benefited the most and preserved ranking quality.
- Isotonic was unstable and hurt log loss despite some Brier improvement.

Interpretation:
- The model’s raw probabilities were too aggressive.
- Sigmoid calibration corrected much of the overconfidence without harming ROC AUC.

Decision:
- Keep.
- Logistic Regression + sigmoid calibration becomes the default final probability path.

---

## Experiment 5: Practical Decision-Use Evaluation

Date:
- 2026-04-20

Summary:
- Added walk-forward prediction export, threshold analysis, and upset review to judge whether the model is useful beyond ranking.
- Focus shifted from pure classification metrics to selective decision quality.

Changes:
- Exported per-fold raw probability, calibrated probability, and actual result.
- Added threshold analysis for calibrated probabilities at `0.55`, `0.60`, `0.65`, and `0.70`.
- Added upset analysis for cases where calibrated probability was below `0.50` but Fighter A still won.

Evaluation Method:
- 5-fold walk-forward evaluation using the calibrated Logistic Regression output as the primary probability path.
- Threshold analysis applied to the combined out-of-fold calibrated predictions.

Results:
- Final probability model: accuracy `0.5990`, ROC AUC `0.6266`, log loss `0.6630`, Brier `0.2353`
- Threshold `>= 0.55`: `2728` signals, win rate `0.627`, Brier `0.2297`
- Threshold `>= 0.60`: `1591` signals, win rate `0.681`, Brier `0.2169`
- Threshold `>= 0.65`: `578` signals, win rate `0.713`, Brier `0.2038`
- Threshold `>= 0.70`: `107` signals, win rate `0.776`, Brier `0.1757`

Impact:
- Higher-confidence subsets looked materially better than the full prediction set.
- The model showed more practical value when used selectively rather than indiscriminately.
- Upset misses still exposed weak spots in matchup interpretation.

Interpretation:
- The current system is more useful as a selective confidence tool than as a pure “bet every fight” probability engine.
- The calibration and threshold behavior are promising, but not enough yet to call the system fully decision-grade.

Decision:
- Keep.
- Use calibrated Logistic Regression probabilities as the primary output and interpret higher-threshold signals as the most actionable subset.

---

## Experiment 6: Physical Difference Features Default-On

Date:
- 2026-04-20

Summary:
- Integrated `age_diff`, `height_diff`, and `reach_diff` as default pre-fight matchup features.
- Re-ran calibrated walk-forward evaluation and historical EV analysis to verify whether the physical features improved the production-style baseline.

Changes:
- Set physical-difference features on by default in `src/features.py` and `src/train.py`.
- Kept odds excluded from model inputs.
- Re-ran the calibrated Logistic Regression walk-forward pipeline and the historical odds-based EV analysis.

Evaluation Method:
- 5-fold expanding-window walk-forward evaluation with the same time-safe sigmoid calibration path as the prior baseline.
- Historical EV evaluation on the merged odds subset using out-of-fold calibrated probabilities.

Results:
- Dataset shape changed from `8473 x 38` to `8473 x 41`
- Active feature count changed from `35` to `38`
- Before physical features, walk-forward Logistic Regression: accuracy `0.5990`, ROC AUC `0.6266`, log loss `0.6630`, Brier `0.2353`
- After physical features, walk-forward Logistic Regression: accuracy `0.6160`, ROC AUC `0.6503`, log loss `0.6522`, Brier `0.2301`
- Before physical features, walk-forward XGBoost: accuracy `0.5837`, ROC AUC `0.6000`, log loss `0.6706`, Brier `0.2389`
- After physical features, walk-forward XGBoost: accuracy `0.6113`, ROC AUC `0.6190`, log loss `0.6632`, Brier `0.2353`
- Historical EV, all bets, `edge > 0.00`: ROI improved from `-0.024` to `-0.007`
- Historical EV, all bets, best threshold moved from `edge > 0.00` / ROI `-0.024` to `edge > 0.02` / ROI `-0.005`

Impact:
- Physical matchup features produced a meaningful lift in both ranking quality and probability quality.
- The EV layer improved materially, especially by reducing losses on edge-filtered bets.
- Overall profitability remained slightly negative when betting all qualifying signals.

Interpretation:
- Age, height, and reach differences add real pre-fight matchup signal rather than just cosmetic context.
- The model remains more reliable as a selective confidence tool than as a fully profitable odds-beating system.
- Positive-edge favorites now look materially stronger than positive-edge underdogs, which still drag down ROI.
- This is now the default baseline configuration: Logistic Regression + sigmoid calibration with `include_physical = True`.

Decision:
- Keep.
- Physical difference features become part of the default baseline.

---

## Experiment 7: Self-Owned Live Data Pipeline Phase 1

Date:
- 2026-04-21

Summary:
- Built the first self-owned live data pipeline under `src/scrapers` so upcoming prediction inputs can be generated from scrapers rather than manually prepared CSVs.
- The live build now produces matchup, odds, fighter-attribute, and recent-history outputs plus an enriched `data/upcoming_fights.csv`.

Changes:
- Added modular scraper/orchestration modules under `src/scrapers`:
  - `common.py`
  - `fetch_upcoming_fights.py`
  - `fetch_current_odds.py`
  - `fetch_fighter_attributes.py`
  - `fetch_fighter_history.py`
  - `derive_live_features.py`
  - `build_live_inputs.py`
- Added normalized fighter-name matching, matchup keys, odds conversion helpers, and UFCStats fighter-directory URL resolution.
- Added raw-vs-clean output separation for live inputs.

Evaluation Method:
- Live smoke-test run of `python -m src.scrapers.build_live_inputs`.
- Coverage checks on generated live outputs.

Results:
- Upcoming matchups scraped: `74`
- Consensus odds rows: `28`
- Fighter attributes scraped: `148`
- Fighter recent-history rows: `1087`
- Final `upcoming_fights.csv` odds coverage: `31.1%`
- Final `upcoming_fights.csv` `A_days_since_last_fight` coverage: `97.3%`

Impact:
- Live predictions and shadow reports are no longer dependent on manually prepared matchup files.
- The project now owns the current-card input pipeline for fighter names, card metadata, physicals, and recent history.
- Odds coverage is still partial and completed-results scraping remains weaker than the rest of the live stack.

Interpretation:
- The live prediction workflow is now operational on self-owned scrape inputs, but still needs odds-source expansion and completed-results hardening before it can be treated as fully production-grade.

Decision:
- Keep and extend.
- The self-owned live data pipeline becomes the default path for future-card inputs.

---

## Experiment 8: Historical Backfill Contract Comparison

Date:
- 2026-04-21

Summary:
- Added a historical backfill runner and comparison layer aimed at replacing the imported raw baselines:
  - `ufc_fight_results.csv`
  - `ufc_fight_stats.csv`
  - `ufc_fighter_details.csv`
  - `ufc-master.csv`
- The new report evaluates raw-file parity, assembled-master parity, and rebuilt training-feature parity.

Changes:
- Added `src/scrapers/backfill_historical_dataset.py` to scrape historical UFCStats events into `data/historical_backfill/`.
- Added `src/scrapers/compare_historical_datasets.py` to compare imported vs scraped outputs.
- Added `data/historical_backfill/ufc_master_scraped.csv` as a first assembled contract target.
- Added explicit retire / do-not-retire decisions per imported baseline file.

Evaluation Method:
- Sampled historical scrape with `--max-events 10`.
- Direct comparison at three levels:
  - raw-file comparison
  - assembled `ufc-master` comparison
  - training-ready feature comparison

Results:
- Scraped raw outputs:
  - `ufc_fight_results.csv`: `(129, 11)`
  - `ufc_fight_stats.csv`: `(258, 19)`
  - `ufc_fighter_details.csv`: `(4487, 4)`
- Scraped assembled output:
  - `ufc_master_scraped.csv`: `(129, 118)`
- Rebuilt training-ready scraped features:
  - `ufc_rebuilt_features_scraped.csv`: `(125, 38)`
- Imported vs scraped retirement decisions:
  - `ufc_fight_results.csv`: `no`
  - `ufc_fight_stats.csv`: `no`
  - `ufc_fighter_details.csv`: `yes`
  - `ufc-master.csv`: `no`

Impact:
- The project now has a concrete readiness report instead of a vague “backfill is partial” status.
- `ufc_fighter_details.csv` is close enough to shadow-replace safely.
- Results, stats, and `ufc-master` are still far from replacement-ready because coverage is low and the stats contract is not yet faithfully rebuilt.

Interpretation:
- Historical fighter directory scraping is strong.
- Historical fight-results scraping is directionally correct but still a small sample relative to the imported baseline.
- Historical fight-stats scraping is currently closer to fight-level totals than a full per-round raw replacement, which is why contract overlap is effectively zero.
- The assembled `ufc_master_scraped.csv` is useful as a comparison target, but many contract fields remain null or only partially reconstructed.

Decision:
- Keep and continue.
- Retire only `ufc_fighter_details.csv` when convenient; keep the other imported baselines as the benchmark until backfill coverage and schema fidelity improve materially.

---

## Experiment 9: Minimum Legacy Contract Audit And Stats Contract Narrowing

Date:
- 2026-04-21

Summary:
- Reframed the four imported baseline CSVs as one combined legacy contract instead of four equally important parity targets.
- Narrowed the historical stats target to the unique fields actually required by the current feature pipeline, then fixed the sampled backfill to read UFCStats per-round rows instead of fight-total rows.

Changes:
- Extended `compare_historical_datasets.py` with:
  - combined legacy contract audit
  - minimum self-owned data contract
  - required-vs-redundant column reporting
  - retirement decisions based on unique required information rather than raw parity alone
- Updated `backfill_historical_dataset.py` to parse per-round fighter rows from UFCStats fight-detail tables.
- Fixed stats-key normalization so imported vs scraped round rows compare on consistent matchup/fighter/round keys.
- Fixed ambiguous UFCStats takedown-column parsing by reading the duplicated headers positionally.

Evaluation Method:
- Sampled historical backfill on `10` events.
- Contract comparison across raw files, assembled master, and rebuilt training features.

Results:
- `ufc_fighter_details.csv` remains the only imported file that is effectively replaceable now.
- `ufc_fight_stats.csv` now contains the required feature-driving columns with real overlap:
  - sampled overlap rows: `210`
  - sampled scraped overlap rate: `81.4%`
- The main blocker is no longer missing stats columns; it is incomplete historical coverage.
- `ufc-master.csv` remains non-replaceable because odds + physical coverage are still partial and many contract fields are not yet reconstructed historically.

Impact:
- The project now has a clear distinction between:
  - required information
  - redundant information
  - non-material legacy clutter
- Historical scraping effort can stay focused on the fields that actually drive model rebuilding.

Interpretation:
- `ufc_fight_results.csv` and `ufc_fight_stats.csv` are still materially needed, but mainly for coverage rather than schema uncertainty.
- `ufc_fighter_details.csv` is operationally useful but not a blocker for rebuilding the current feature set.
- `ufc-master.csv` is mostly redundant except for historical odds and physical matchup inputs.

Decision:
- Keep and continue.
- Prioritize expanding historical scrape coverage and historical physical/odds enrichment instead of chasing field-for-field parity on redundant legacy columns.

---

## Experiment 10: Historical Physicals And Date Backbone Independence

Date:
- 2026-04-21

Summary:
- Added self-owned historical physical enrichment and event-date validation to the backfill pipeline.
- The historical rebuilt feature dataset can now include the same physical-difference features as the current training baseline without depending on `ufc-master.csv`.

Changes:
- Added historical fighter-profile enrichment with DOB, height, reach, and stance.
- Added derived historical physical matchup rows with:
  - `A_age`, `B_age`
  - `A_height_cms`, `B_height_cms`
  - `A_reach_cms`, `B_reach_cms`
  - `age_diff`, `height_diff`, `reach_diff`
- Added historical event catalog and date validation reporting.
- Added first-pass historical odds archive/search scraping and merged whatever coverage was available into `ufc_master_scraped.csv`.
- Rebuilt the scraped feature dataset with physical features enabled from self-owned data.

Evaluation Method:
- Resume-safe sampled historical backfill on `10` events.
- Historical comparison report against imported baseline contracts.

Results:
- `historical_physical_features_scraped.csv`: `129` rows
- `historical_event_catalog_scraped.csv`: `10` events
- `ufc_master_scraped.csv`: `258 x 118`
- `ufc_rebuilt_features_scraped.csv`: `251 x 41`
- Historical date validation:
  - missing dates: `0`
  - negative fighter gaps: `0`
  - large fighter gaps: `0`
- Scraped master physical coverage:
  - age: `100%`
  - height: `100%`
  - reach: `99.6%+`
- Scraped master odds coverage:
  - `R_odds` / `B_odds`: about `4.3%`

Impact:
- Historical physical-difference features are no longer blocked by `ufc-master.csv`.
- Event dates are now validated and usable as a chronological backbone.
- The main remaining blocker for full independence shifted decisively to historical odds coverage, not physicals or dates.

Interpretation:
- The self-owned pipeline can now rebuild the training feature set with physical diffs from scraped data.
- Imported `ufc-master.csv` is now mostly needed for historical odds coverage rather than for physical enrichment.

Decision:
- Keep and continue.
- Next dependency-removal work should focus on historical odds scaling and larger resume-safe historical coverage.

---

## Current Best Model
- Model type: Logistic Regression
- Calibration method: Sigmoid / Platt scaling
- Baseline configuration: `include_physical = True`
- Key walk-forward metrics: accuracy `0.6160`, ROC AUC `0.6503`, log loss `0.6522`, Brier `0.2301`
- Why this is the current baseline:
  - Best calibrated probability quality among tested options
  - Improved materially after adding physical matchup-difference features
  - Maintains ranking quality while improving log loss and Brier
  - More stable than isotonic calibration

## Current Feature Set
- High-level description:
  - Pruned differential feature set with career and recent-form signals
  - Includes striking efficiency, takedown efficiency, absorbed-based defense, opponent-adjusted striking, finishing rates, experience, momentum, ELO context, and physical matchup differences
- What types of features are included:
  - Career and last-3 differential stats
  - Opponent-adjusted striking and defensive ratios
  - Select matchup features
  - ELO and opponent-quality context
  - Age, height, and reach differences

## Practical Use Insights
- Calibrated probabilities become more reliable at higher thresholds.
- The model is most useful when used selectively rather than across every fight.
- Signals at `0.60+` and especially `0.65+` look more actionable than the full set of predictions.
- In EV testing, positive-edge favorites behave much better than positive-edge underdogs.

## Known Weaknesses
- Overconfidence still appears in some probability regions even after calibration.
- Upset misses cluster around cases with strong striking-adjustment features and strong opponent-quality context.
- Walk-forward results are meaningfully weaker than favorable single-split results.
- All-bets EV remains slightly negative even after the physical-feature improvement.
- Positive-edge underdogs still drive most of the betting losses.

## Next Steps
- Expand historical backfill beyond sampled events with resumable scraping and coverage tracking.
- Expand historical odds coverage from scraped sources so `ufc-master.csv` is no longer the only source for those unique fields.
- Improve completed-results scraping and broader live odds coverage so the live stack is fully self-owned.
- Use the generated historical comparison report as the gating document before retiring any remaining imported baseline files.

---

## Phase Completion And Release Prep Summary

Date:
- 2026-04-27

Summary:
- Phase 1 completed: self-owned data pipeline
- Phase 2 completed: calibrated model selection
- Phase 3 completed: betting strategy validation and live shadow tracking
- Phase 4 started: repo organization and public release prep

Changes:
- Reorganized Phase 3 narrative docs into `docs/strategy/`
- Added release-prep documents under `docs/release_prep/`
- Refreshed top-level project docs so the repository now reads as a four-phase system

Impact:
- The repo is easier to understand before pruning and public cleanup.
- Phase boundaries, locked decisions, and remaining public-release tasks are now explicit.

Decision:
- Keep.
- Continue with Phase 4 pruning and private-information review after this organizational pass.
