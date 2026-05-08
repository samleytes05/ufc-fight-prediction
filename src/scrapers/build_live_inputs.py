from __future__ import annotations

"""Orchestrate live UFC scrapes into raw and enriched upcoming-fight CSVs."""

import argparse
from pathlib import Path

import pandas as pd

from .common import DATA_DIR, log
from .derive_live_features import build_live_feature_table, validate_live_feature_table
from .fetch_completed_results import scrape_completed_results
from .fetch_current_odds import (
    DEFAULT_CONSENSUS_OUTPUT_PATH,
    DEFAULT_RAW_OUTPUT_PATH,
    scrape_current_odds,
)
from .fetch_fighter_attributes import scrape_fighter_attributes
from .fetch_fighter_history import scrape_fighter_history
from .fetch_upcoming_fights import scrape_upcoming_matchups
from src.strategy.live_shadow_tracking import run_live_shadow_update


DEFAULT_UPCOMING_OUTPUT = DATA_DIR / "upcoming_fights_scraped.csv"
DEFAULT_RESULTS_OUTPUT = DATA_DIR / "completed_results_scraped.csv"
DEFAULT_ATTRIBUTES_OUTPUT = DATA_DIR / "fighter_attributes_scraped.csv"
DEFAULT_HISTORY_OUTPUT = DATA_DIR / "fighter_recent_history_scraped.csv"
DEFAULT_READY_OUTPUT = DATA_DIR / "upcoming_fights.csv"


def build_live_prediction_input(
    upcoming_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    attributes_df: pd.DataFrame,
    history_df: pd.DataFrame,
    output_path: str | Path = DEFAULT_READY_OUTPUT,
) -> pd.DataFrame:
    """Merge scraped live sources into the final enriched upcoming-fight input table."""
    live_df = build_live_feature_table(
        upcoming_df=upcoming_df,
        odds_df=odds_df,
        attributes_df=attributes_df,
        history_df=history_df,
        output_path=output_path,
    )
    validate_live_feature_table(live_df)
    return live_df


def refresh_live_data(
    upcoming_output: str | Path = DEFAULT_UPCOMING_OUTPUT,
    odds_raw_output: str | Path = DEFAULT_RAW_OUTPUT_PATH,
    odds_consensus_output: str | Path = DEFAULT_CONSENSUS_OUTPUT_PATH,
    results_output: str | Path = DEFAULT_RESULTS_OUTPUT,
    attributes_output: str | Path = DEFAULT_ATTRIBUTES_OUTPUT,
    history_output: str | Path = DEFAULT_HISTORY_OUTPUT,
    ready_output: str | Path = DEFAULT_READY_OUTPUT,
    limit_completed_events: int = 25,
    max_fights_per_fighter: int = 12,
    include_completed_results: bool = False,
    bankroll: float = 10_000.0,
    update_shadow_tracking: bool = True,
    settle_live_tracking: bool = False,
) -> dict[str, pd.DataFrame]:
    """Refresh raw live scrapes plus the enriched upcoming_fights.csv build."""
    log("starting live data refresh")
    upcoming_df = scrape_upcoming_matchups(output_path=upcoming_output)
    raw_odds_df, consensus_odds_df = scrape_current_odds(
        raw_output_path=odds_raw_output,
        consensus_output_path=odds_consensus_output,
    )
    attributes_df = scrape_fighter_attributes(upcoming_df=upcoming_df, output_path=attributes_output)
    history_df = scrape_fighter_history(
        upcoming_df=upcoming_df,
        output_path=history_output,
        max_fights_per_fighter=max_fights_per_fighter,
    )
    if include_completed_results:
        try:
            results_df = scrape_completed_results(output_path=results_output, limit_events=limit_completed_events)
        except Exception as error:
            log(f"optional completed-results refresh failed: {type(error).__name__}: {error}")
            results_df = pd.DataFrame()
    else:
        results_df = pd.DataFrame()
    ready_df = build_live_prediction_input(
        upcoming_df=upcoming_df,
        odds_df=consensus_odds_df,
        attributes_df=attributes_df,
        history_df=history_df,
        output_path=ready_output,
    )
    shadow_df = pd.DataFrame()
    latest_recommendations_df = pd.DataFrame()
    if update_shadow_tracking:
        try:
            shadow_df, latest_recommendations_df = run_live_shadow_update(
                bankroll=bankroll,
                settle_tracking=settle_live_tracking,
                limit_completed_events=limit_completed_events,
            )
        except Exception as error:
            log(f"live shadow tracking update failed: {type(error).__name__}: {error}")

    print("Live merge coverage")
    print(f"  upcoming matchups: {len(upcoming_df)}")
    print(f"  consensus odds rows: {len(consensus_odds_df)}")
    print(f"  fighter attributes rows: {len(attributes_df)}")
    print(f"  fighter history rows: {len(history_df)}")
    print(f"  completed results rows: {len(results_df)}")
    if len(ready_df) > 0:
        print(f"  odds coverage: {((ready_df['odds_A'].notna() | ready_df['odds_B'].notna()).mean() * 100):.1f}%")
        print(f"  attribute coverage A_age: {(ready_df['A_age'].notna().mean() * 100):.1f}%")
        print(f"  history coverage A_days_since_last_fight: {(ready_df['A_days_since_last_fight'].notna().mean() * 100):.1f}%")
    if update_shadow_tracking:
        print(f"  shadow report rows: {len(shadow_df)}")
        print(f"  v2_core live recommendations: {len(latest_recommendations_df)}")

    return {
        "upcoming": upcoming_df,
        "odds_raw": raw_odds_df,
        "odds_consensus": consensus_odds_df,
        "attributes": attributes_df,
        "history": history_df,
        "results": results_df,
        "ready": ready_df,
        "shadow": shadow_df,
        "latest_recommendations": latest_recommendations_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh live UFC scrapes and build an enriched upcoming_fights.csv.")
    parser.add_argument("--upcoming-output", type=Path, default=DEFAULT_UPCOMING_OUTPUT, help="Upcoming fights CSV path.")
    parser.add_argument("--odds-raw-output", type=Path, default=DEFAULT_RAW_OUTPUT_PATH, help="Raw odds CSV path.")
    parser.add_argument(
        "--odds-consensus-output",
        type=Path,
        default=DEFAULT_CONSENSUS_OUTPUT_PATH,
        help="Consensus odds CSV path.",
    )
    parser.add_argument("--results-output", type=Path, default=DEFAULT_RESULTS_OUTPUT, help="Completed results CSV path.")
    parser.add_argument("--attributes-output", type=Path, default=DEFAULT_ATTRIBUTES_OUTPUT, help="Fighter attributes CSV path.")
    parser.add_argument("--history-output", type=Path, default=DEFAULT_HISTORY_OUTPUT, help="Fighter history CSV path.")
    parser.add_argument("--ready-output", type=Path, default=DEFAULT_READY_OUTPUT, help="Final upcoming fights CSV path.")
    parser.add_argument("--limit-completed-events", type=int, default=25, help="How many recent completed events to scrape.")
    parser.add_argument("--max-fights-per-fighter", type=int, default=12, help="How many recent fights to scrape per fighter.")
    parser.add_argument("--bankroll", type=float, default=10_000.0, help="Bankroll used for live V2_Core sizing columns.")
    parser.add_argument(
        "--include-completed-results",
        action="store_true",
        help="Also refresh the slower completed-results feed.",
    )
    parser.add_argument(
        "--skip-shadow-tracking",
        action="store_true",
        help="Skip updating the live shadow report and live tracking outputs.",
    )
    parser.add_argument(
        "--settle-live-tracking",
        action="store_true",
        help="After scraping completed results, settle matching live tracking rows and CLV fields.",
    )
    args = parser.parse_args()

    outputs = refresh_live_data(
        upcoming_output=args.upcoming_output,
        odds_raw_output=args.odds_raw_output,
        odds_consensus_output=args.odds_consensus_output,
        results_output=args.results_output,
        attributes_output=args.attributes_output,
        history_output=args.history_output,
        ready_output=args.ready_output,
        limit_completed_events=args.limit_completed_events,
        max_fights_per_fighter=args.max_fights_per_fighter,
        include_completed_results=args.include_completed_results,
        bankroll=args.bankroll,
        update_shadow_tracking=not args.skip_shadow_tracking,
        settle_live_tracking=args.settle_live_tracking,
    )
    print(f"Saved upcoming fights: {args.upcoming_output} ({len(outputs['upcoming'])} rows)")
    print(f"Saved raw odds: {args.odds_raw_output} ({len(outputs['odds_raw'])} rows)")
    print(f"Saved consensus odds: {args.odds_consensus_output} ({len(outputs['odds_consensus'])} rows)")
    print(f"Saved fighter attributes: {args.attributes_output} ({len(outputs['attributes'])} rows)")
    print(f"Saved fighter history: {args.history_output} ({len(outputs['history'])} rows)")
    print(f"Saved completed results: {args.results_output} ({len(outputs['results'])} rows)")
    print(f"Saved ready-to-score file: {args.ready_output} ({len(outputs['ready'])} rows)")


if __name__ == "__main__":
    main()
