from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scrapers.common import american_to_implied_probability, build_pair_key, normalize_fighter_name
from src.scrapers.fetch_completed_results import scrape_completed_results


DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "strategy"
LIVE_TRACKING_DIR = OUTPUTS_DIR / "live_tracking"

FUTURE_PREDICTIONS_PATH = DATA_DIR / "future_fight_predictions.csv"
SHADOW_REPORT_PATH = DATA_DIR / "future_card_shadow_report.csv"
UPCOMING_FIGHTS_PATH = DATA_DIR / "upcoming_fights.csv"
CURRENT_ODDS_PATH = DATA_DIR / "current_odds_scraped.csv"
COMPLETED_RESULTS_PATH = DATA_DIR / "completed_results_scraped.csv"
LIVE_TRACKING_PATH = LIVE_TRACKING_DIR / "live_bet_tracking.csv"
LATEST_RECOMMENDATIONS_PATH = LIVE_TRACKING_DIR / "latest_live_recommendations.csv"
FINAL_STATUS_PATH = PROJECT_ROOT / "docs" / "strategy" / "BETTING_STRATEGY_STATUS.md"

DEFAULT_BANKROLL = 10_000.0
V2_CORE_EDGE_THRESHOLD = 0.04
V2_CORE_CONFIDENCE_THRESHOLD = 0.65
FLAT_BENCHMARK_STAKE = 100.0
CONSERVATIVE_BANKROLL_FRACTION = 0.01
AGGRESSIVE_KELLY_CAP = 0.02


def american_profit(odds: float, stake: float) -> float:
    if pd.isna(odds) or odds == 0 or pd.isna(stake):
        return np.nan
    if odds > 0:
        return float(stake * (odds / 100.0))
    return float(stake * (100.0 / abs(odds)))


def payout_multiple(odds: float) -> float:
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds > 0:
        return float(odds / 100.0)
    return float(100.0 / abs(odds))


def expected_profit(probability: float, odds: float, stake: float) -> float:
    if pd.isna(probability) or pd.isna(odds) or pd.isna(stake) or stake <= 0:
        return 0.0
    profit_if_win = american_profit(odds, stake)
    return float((probability * profit_if_win) - ((1.0 - probability) * stake))


def kelly_fraction(probability: float, odds: float) -> float:
    b = payout_multiple(odds)
    if pd.isna(probability) or pd.isna(b) or b <= 0:
        return 0.0
    q = 1.0 - float(probability)
    fraction = ((b * float(probability)) - q) / b
    return max(float(fraction), 0.0)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _normalize_pair_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "fighter_A" in result.columns:
        result["fighter_A_normalized"] = result["fighter_A"].map(normalize_fighter_name)
    if "fighter_B" in result.columns:
        result["fighter_B_normalized"] = result["fighter_B"].map(normalize_fighter_name)
    if {"fighter_A", "fighter_B"}.issubset(result.columns):
        result["pair_key"] = result.apply(lambda row: build_pair_key(row["fighter_A"], row["fighter_B"]), axis=1)
    return result


def load_upcoming_context() -> pd.DataFrame:
    upcoming_df = _safe_read_csv(UPCOMING_FIGHTS_PATH)
    current_odds_df = _safe_read_csv(CURRENT_ODDS_PATH)

    upcoming_df = _normalize_pair_columns(upcoming_df)
    current_odds_df = _normalize_pair_columns(current_odds_df)

    if not upcoming_df.empty:
        keep_cols = [
            "pair_key",
            "event_name",
            "event_date",
            "event_location",
            "bout_url",
            "event_url",
            "odds_A",
            "odds_B",
            "fighter_A_implied_prob",
            "fighter_B_implied_prob",
            "fighter_A_normalized",
            "fighter_B_normalized",
        ]
        keep_cols = [column for column in keep_cols if column in upcoming_df.columns]
        upcoming_df = upcoming_df[keep_cols].drop_duplicates(subset=["pair_key"], keep="last")

    if not current_odds_df.empty:
        odds_keep_cols = [
            "pair_key",
            "event_name",
            "event_date",
            "odds_A",
            "odds_B",
            "fighter_A_implied_prob",
            "fighter_B_implied_prob",
        ]
        odds_keep_cols = [column for column in odds_keep_cols if column in current_odds_df.columns]
        current_odds_df = current_odds_df[odds_keep_cols].drop_duplicates(subset=["pair_key"], keep="last")

    if upcoming_df.empty:
        return current_odds_df
    if current_odds_df.empty:
        return upcoming_df

    merged = upcoming_df.merge(current_odds_df, on="pair_key", how="outer", suffixes=("", "_odds"))
    for column in ["event_name", "event_date", "odds_A", "odds_B", "fighter_A_implied_prob", "fighter_B_implied_prob"]:
        odds_column = f"{column}_odds"
        if odds_column in merged.columns:
            merged[column] = merged[column].combine_first(merged[odds_column])
    drop_cols = [column for column in merged.columns if column.endswith("_odds")]
    return merged.drop(columns=drop_cols)


def build_shadow_base() -> pd.DataFrame:
    shadow_df = _safe_read_csv(SHADOW_REPORT_PATH)
    predictions_df = _safe_read_csv(FUTURE_PREDICTIONS_PATH)
    context_df = load_upcoming_context()

    if not shadow_df.empty:
        base_df = shadow_df.copy()
    elif not predictions_df.empty:
        base_df = predictions_df.copy()
    else:
        raise FileNotFoundError(
            "No existing live shadow artifacts found. Expected either data/future_card_shadow_report.csv "
            "or data/future_fight_predictions.csv."
        )

    base_df = _normalize_pair_columns(base_df)
    if not context_df.empty and "pair_key" in base_df.columns:
        base_df = base_df.merge(context_df, on="pair_key", how="left", suffixes=("", "_context"))
        for column in ["event_name", "event_date", "odds_A", "odds_B", "fighter_A_implied_prob", "fighter_B_implied_prob"]:
            context_column = f"{column}_context"
            if context_column in base_df.columns:
                if column in base_df.columns:
                    base_df[column] = base_df[column].combine_first(base_df[context_column])
                else:
                    base_df[column] = base_df[context_column]
        drop_cols = [column for column in base_df.columns if column.endswith("_context")]
        if drop_cols:
            base_df = base_df.drop(columns=drop_cols)

    if "calibrated_probability" in base_df.columns and "p_model_A" not in base_df.columns:
        base_df["p_model_A"] = pd.to_numeric(base_df["calibrated_probability"], errors="coerce")
    elif "model_win_probability" in base_df.columns and "p_model_A" not in base_df.columns:
        base_df["p_model_A"] = pd.to_numeric(base_df["model_win_probability"], errors="coerce")
    else:
        base_df["p_model_A"] = pd.to_numeric(base_df.get("p_model_A"), errors="coerce")

    base_df["odds_A"] = pd.to_numeric(base_df.get("odds_A"), errors="coerce")
    base_df["odds_B"] = pd.to_numeric(base_df.get("odds_B"), errors="coerce")

    if "implied_probability" in base_df.columns and "implied_prob_A" not in base_df.columns:
        base_df["implied_prob_A"] = pd.to_numeric(base_df["implied_probability"], errors="coerce")
    elif "fighter_A_implied_prob" in base_df.columns and "implied_prob_A" not in base_df.columns:
        base_df["implied_prob_A"] = pd.to_numeric(base_df["fighter_A_implied_prob"], errors="coerce")
    else:
        base_df["implied_prob_A"] = pd.to_numeric(base_df.get("implied_prob_A"), errors="coerce")
    base_df["implied_prob_A"] = base_df["implied_prob_A"].combine_first(base_df["odds_A"].map(american_to_implied_probability))

    if "edge" in base_df.columns and "edge_A" not in base_df.columns:
        base_df["edge_A"] = pd.to_numeric(base_df["edge"], errors="coerce")
    else:
        base_df["edge_A"] = pd.to_numeric(base_df.get("edge_A"), errors="coerce")
    base_df["edge_A"] = base_df["edge_A"].combine_first(base_df["p_model_A"] - base_df["implied_prob_A"])

    if "calibrated_probability" not in base_df.columns:
        base_df["calibrated_probability"] = base_df["p_model_A"]
    if "implied_probability" not in base_df.columns:
        base_df["implied_probability"] = base_df["implied_prob_A"]
    if "edge" not in base_df.columns:
        base_df["edge"] = base_df["edge_A"]

    if "bout" not in base_df.columns:
        base_df["bout"] = base_df["fighter_A"].fillna("") + " vs. " + base_df["fighter_B"].fillna("")
        base_df["bout"] = base_df["bout"].str.strip()
    if "event_name" not in base_df.columns:
        base_df["event_name"] = ""
    if "event_date" not in base_df.columns:
        base_df["event_date"] = ""

    if "model_confidence_tier" not in base_df.columns:
        base_df["model_confidence_tier"] = np.select(
            [
                base_df["p_model_A"] >= 0.70,
                base_df["p_model_A"] >= 0.60,
            ],
            ["high", "medium"],
            default="low",
        )

    if "qualifies_under_strategy_D" not in base_df.columns:
        strategy_d = (base_df["edge_A"] > 0.0) & (base_df["odds_A"] <= 150.0)
        base_df["qualifies_under_strategy_D"] = np.where(strategy_d, "yes", "no")

    return base_df


def enrich_shadow_report(base_df: pd.DataFrame, bankroll: float) -> pd.DataFrame:
    result = base_df.copy()
    result["v2_core_bet_flag"] = (
        result["edge_A"].gt(V2_CORE_EDGE_THRESHOLD)
        & result["p_model_A"].ge(V2_CORE_CONFIDENCE_THRESHOLD)
    )

    result["stake_flat_100"] = np.where(result["v2_core_bet_flag"], FLAT_BENCHMARK_STAKE, 0.0)
    result["stake_bankroll_1pct"] = np.where(result["v2_core_bet_flag"], bankroll * CONSERVATIVE_BANKROLL_FRACTION, 0.0)

    raw_kelly = result.apply(
        lambda row: kelly_fraction(float(row["p_model_A"]), float(row["odds_A"]))
        if pd.notna(row["p_model_A"]) and pd.notna(row["odds_A"])
        else 0.0,
        axis=1,
    )
    result["kelly_fraction_raw"] = raw_kelly
    result["stake_kelly_capped_2pct"] = np.where(
        result["v2_core_bet_flag"],
        bankroll * np.minimum(raw_kelly, AGGRESSIVE_KELLY_CAP),
        0.0,
    )

    result["expected_profit_flat_100"] = result.apply(
        lambda row: expected_profit(float(row["p_model_A"]), float(row["odds_A"]), float(row["stake_flat_100"]))
        if row["v2_core_bet_flag"]
        else 0.0,
        axis=1,
    )
    result["expected_profit_bankroll_1pct"] = result.apply(
        lambda row: expected_profit(float(row["p_model_A"]), float(row["odds_A"]), float(row["stake_bankroll_1pct"]))
        if row["v2_core_bet_flag"]
        else 0.0,
        axis=1,
    )
    result["expected_profit_kelly_capped_2pct"] = result.apply(
        lambda row: expected_profit(float(row["p_model_A"]), float(row["odds_A"]), float(row["stake_kelly_capped_2pct"]))
        if row["v2_core_bet_flag"]
        else 0.0,
        axis=1,
    )
    result["sizing_profile"] = np.where(
        result["v2_core_bet_flag"],
        "benchmark_flat_100 | conservative_1pct_bankroll | aggressive_kelly_capped_2pct",
        "no_bet",
    )

    ordered_columns = [
        "event_name",
        "event_date",
        "bout",
        "fighter_A",
        "fighter_B",
        "p_model_A",
        "implied_prob_A",
        "edge_A",
        "odds_A",
        "v2_core_bet_flag",
        "stake_flat_100",
        "stake_bankroll_1pct",
        "stake_kelly_capped_2pct",
        "expected_profit_flat_100",
        "expected_profit_bankroll_1pct",
        "expected_profit_kelly_capped_2pct",
        "sizing_profile",
    ]
    for alias_column in ["calibrated_probability", "implied_probability", "edge", "qualifies_under_strategy_D", "model_confidence_tier"]:
        if alias_column in result.columns and alias_column not in ordered_columns:
            ordered_columns.append(alias_column)
    remaining_columns = [column for column in result.columns if column not in ordered_columns]
    return result[ordered_columns + remaining_columns]


def write_shadow_report(shadow_df: pd.DataFrame) -> None:
    SHADOW_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shadow_df.to_csv(SHADOW_REPORT_PATH, index=False)


def build_tracking_rows(shadow_df: pd.DataFrame) -> pd.DataFrame:
    recommendations_df = shadow_df[shadow_df["v2_core_bet_flag"]].copy()
    if recommendations_df.empty:
        return pd.DataFrame(
            columns=[
                "run_timestamp",
                "event",
                "fight",
                "fighter_A",
                "fighter_B",
                "selected_fighter",
                "odds_A",
                "odds_at_pick",
                "closing_odds",
                "line_movement",
                "beat_closing_line_flag",
                "p_model_A",
                "implied_prob_A",
                "edge_A",
                "stake_flat_100",
                "stake_bankroll_1pct",
                "stake_kelly_capped_2pct",
                "expected_profit_flat_100",
                "expected_profit_bankroll_1pct",
                "expected_profit_kelly_capped_2pct",
                "result",
                "profit_flat_100",
                "profit_bankroll_1pct",
                "profit_kelly_capped_2pct",
                "notes",
            ]
        )

    run_timestamp = datetime.now(timezone.utc).isoformat()
    tracking_df = pd.DataFrame(
        {
            "run_timestamp": run_timestamp,
            "event": recommendations_df["event_name"].fillna(""),
            "fight": recommendations_df["bout"].fillna(""),
            "fighter_A": recommendations_df["fighter_A"].fillna(""),
            "fighter_B": recommendations_df["fighter_B"].fillna(""),
            "selected_fighter": recommendations_df["fighter_A"].fillna(""),
            "odds_A": recommendations_df["odds_A"],
            "odds_at_pick": recommendations_df["odds_A"],
            "closing_odds": "",
            "line_movement": "",
            "beat_closing_line_flag": "",
            "p_model_A": recommendations_df["p_model_A"],
            "implied_prob_A": recommendations_df["implied_prob_A"],
            "edge_A": recommendations_df["edge_A"],
            "stake_flat_100": recommendations_df["stake_flat_100"],
            "stake_bankroll_1pct": recommendations_df["stake_bankroll_1pct"],
            "stake_kelly_capped_2pct": recommendations_df["stake_kelly_capped_2pct"],
            "expected_profit_flat_100": recommendations_df["expected_profit_flat_100"],
            "expected_profit_bankroll_1pct": recommendations_df["expected_profit_bankroll_1pct"],
            "expected_profit_kelly_capped_2pct": recommendations_df["expected_profit_kelly_capped_2pct"],
            "result": "",
            "profit_flat_100": "",
            "profit_bankroll_1pct": "",
            "profit_kelly_capped_2pct": "",
            "notes": "pending",
        }
    )
    return tracking_df


def append_live_tracking(tracking_rows_df: pd.DataFrame) -> pd.DataFrame:
    LIVE_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    existing_df = _safe_read_csv(LIVE_TRACKING_PATH)
    if existing_df.empty:
        final_df = tracking_rows_df.copy()
        final_df.to_csv(LIVE_TRACKING_PATH, index=False)
        return final_df

    existing_df["odds_A"] = pd.to_numeric(existing_df["odds_A"], errors="coerce")
    tracking_rows_df["odds_A"] = pd.to_numeric(tracking_rows_df["odds_A"], errors="coerce")

    existing_keys = {
        (
            str(row.event),
            str(row.fight),
            str(row.selected_fighter),
            round(float(row.odds_A), 6) if pd.notna(row.odds_A) else np.nan,
        )
        for row in existing_df.itertuples(index=False)
    }

    new_rows = []
    for row in tracking_rows_df.itertuples(index=False):
        key = (
            str(row.event),
            str(row.fight),
            str(row.selected_fighter),
            round(float(row.odds_A), 6) if pd.notna(row.odds_A) else np.nan,
        )
        if key not in existing_keys:
            new_rows.append(row._asdict())

    if new_rows:
        final_df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        final_df = existing_df

    final_df.to_csv(LIVE_TRACKING_PATH, index=False)
    return final_df


def _resolve_actual_outcome_for_tracking_row(row: pd.Series, completed_df: pd.DataFrame) -> tuple[str, float | None]:
    fighter_a_norm = normalize_fighter_name(row["fighter_A"])
    fighter_b_norm = normalize_fighter_name(row["fighter_B"])
    event_name = str(row["event"])

    direct = completed_df[
        (completed_df["event_name"] == event_name)
        & (completed_df["fighter_A_normalized"] == fighter_a_norm)
        & (completed_df["fighter_B_normalized"] == fighter_b_norm)
    ]
    if not direct.empty:
        won = int(pd.to_numeric(direct.iloc[0]["actual_outcome"], errors="coerce"))
        return ("win" if won == 1 else "loss"), float(won)

    reverse = completed_df[
        (completed_df["event_name"] == event_name)
        & (completed_df["fighter_A_normalized"] == fighter_b_norm)
        & (completed_df["fighter_B_normalized"] == fighter_a_norm)
    ]
    if not reverse.empty:
        opponent_won = int(pd.to_numeric(reverse.iloc[0]["actual_outcome"], errors="coerce"))
        won = 0 if opponent_won == 1 else 1
        return ("win" if won == 1 else "loss"), float(won)

    return "", None


def _compute_profit_from_result(result: str, odds: float, stake: float) -> float | str:
    if result == "":
        return ""
    if stake <= 0 or pd.isna(odds):
        return 0.0
    if result == "win":
        return american_profit(float(odds), float(stake))
    if result == "loss":
        return -float(stake)
    return ""


def update_clv_columns(tracking_df: pd.DataFrame, latest_shadow_df: pd.DataFrame) -> pd.DataFrame:
    result = tracking_df.copy()
    if result.empty:
        return result

    latest = latest_shadow_df.copy()
    if latest.empty:
        if "closing_odds" not in result.columns:
            result["closing_odds"] = ""
        if "line_movement" not in result.columns:
            result["line_movement"] = ""
        if "beat_closing_line_flag" not in result.columns:
            result["beat_closing_line_flag"] = ""
        return result

    latest["event_name"] = latest["event_name"].fillna("")
    latest["bout"] = latest["bout"].fillna("")
    latest_lookup = latest.set_index(["event_name", "bout"])

    closing_odds_values: list[object] = []
    line_movement_values: list[object] = []
    beat_closing_values: list[object] = []

    for row in result.itertuples(index=False):
        key = (str(row.event), str(row.fight))
        if key not in latest_lookup.index:
            closing_odds_values.append("")
            line_movement_values.append("")
            beat_closing_values.append("")
            continue

        latest_row = latest_lookup.loc[key]
        if isinstance(latest_row, pd.DataFrame):
            latest_row = latest_row.iloc[0]

        closing_odds = pd.to_numeric(latest_row["odds_A"], errors="coerce")
        odds_at_pick = pd.to_numeric(getattr(row, "odds_at_pick", row.odds_A), errors="coerce")
        if pd.isna(closing_odds) or pd.isna(odds_at_pick):
            closing_odds_values.append("")
            line_movement_values.append("")
            beat_closing_values.append("")
            continue

        implied_pick = american_to_implied_probability(odds_at_pick)
        implied_close = american_to_implied_probability(closing_odds)
        beat_closing = (
            bool(implied_pick < implied_close)
            if not pd.isna(implied_pick) and not pd.isna(implied_close)
            else ""
        )
        closing_odds_values.append(float(closing_odds))
        line_movement_values.append(float(closing_odds - odds_at_pick))
        beat_closing_values.append(beat_closing)

    result["closing_odds"] = closing_odds_values
    result["line_movement"] = line_movement_values
    result["beat_closing_line_flag"] = beat_closing_values
    return result


def settle_live_tracking(
    completed_results_path: Path = COMPLETED_RESULTS_PATH,
    latest_shadow_df: pd.DataFrame | None = None,
    limit_completed_events: int = 25,
) -> pd.DataFrame:
    tracking_df = _safe_read_csv(LIVE_TRACKING_PATH)
    if tracking_df.empty:
        return tracking_df

    completed_df = scrape_completed_results(output_path=completed_results_path, limit_events=limit_completed_events)
    if completed_df.empty:
        if latest_shadow_df is not None:
            tracking_df = update_clv_columns(tracking_df, latest_shadow_df)
            tracking_df.to_csv(LIVE_TRACKING_PATH, index=False)
        return tracking_df

    for column in ["odds_at_pick", "closing_odds", "line_movement", "beat_closing_line_flag"]:
        if column not in tracking_df.columns:
            tracking_df[column] = ""
    for column in ["result", "profit_flat_100", "profit_bankroll_1pct", "profit_kelly_capped_2pct", "notes"]:
        if column not in tracking_df.columns:
            tracking_df[column] = ""

    if latest_shadow_df is not None:
        tracking_df = update_clv_columns(tracking_df, latest_shadow_df)

    for idx, row in tracking_df.iterrows():
        resolved_result, actual_outcome = _resolve_actual_outcome_for_tracking_row(row, completed_df)
        if resolved_result == "":
            continue

        odds_at_pick = pd.to_numeric(row.get("odds_at_pick", row.get("odds_A")), errors="coerce")
        stake_flat = pd.to_numeric(row.get("stake_flat_100"), errors="coerce")
        stake_pct = pd.to_numeric(row.get("stake_bankroll_1pct"), errors="coerce")
        stake_kelly = pd.to_numeric(row.get("stake_kelly_capped_2pct"), errors="coerce")
        expected_flat = pd.to_numeric(row.get("expected_profit_flat_100"), errors="coerce")
        expected_pct = pd.to_numeric(row.get("expected_profit_bankroll_1pct"), errors="coerce")
        expected_kelly = pd.to_numeric(row.get("expected_profit_kelly_capped_2pct"), errors="coerce")

        profit_flat = _compute_profit_from_result(resolved_result, odds_at_pick, float(stake_flat) if pd.notna(stake_flat) else 0.0)
        profit_pct = _compute_profit_from_result(resolved_result, odds_at_pick, float(stake_pct) if pd.notna(stake_pct) else 0.0)
        profit_kelly = _compute_profit_from_result(resolved_result, odds_at_pick, float(stake_kelly) if pd.notna(stake_kelly) else 0.0)

        expected_vs_actual_parts: list[str] = []
        for label, expected_value, actual_value in [
            ("flat", expected_flat, profit_flat),
            ("1pct", expected_pct, profit_pct),
            ("kelly2", expected_kelly, profit_kelly),
        ]:
            if pd.notna(expected_value) and actual_value != "":
                delta = float(actual_value) - float(expected_value)
                expected_vs_actual_parts.append(f"{label}_delta={delta:.2f}")

        tracking_df.at[idx, "result"] = resolved_result
        tracking_df.at[idx, "profit_flat_100"] = profit_flat
        tracking_df.at[idx, "profit_bankroll_1pct"] = profit_pct
        tracking_df.at[idx, "profit_kelly_capped_2pct"] = profit_kelly
        tracking_df.at[idx, "notes"] = "settled" if not expected_vs_actual_parts else "settled; " + "; ".join(expected_vs_actual_parts)

    tracking_df.to_csv(LIVE_TRACKING_PATH, index=False)
    return tracking_df


def write_latest_recommendations(shadow_df: pd.DataFrame) -> pd.DataFrame:
    LIVE_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    latest_df = shadow_df[shadow_df["v2_core_bet_flag"]].copy()
    latest_df.to_csv(LATEST_RECOMMENDATIONS_PATH, index=False)
    return latest_df


def write_final_status() -> None:
    FINAL_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_STATUS_PATH.write_text(
        "\n".join(
            [
                "# Phase 3 Final Status",
                "",
                "## Locked Model",
                "- Logistic Regression + Platt calibration",
                "",
                "## Locked Strategy",
                "- V2_Core",
                "- edge_A > 0.04",
                "- p_model_A >= 0.65",
                "",
                "## Sizing Profiles",
                "- Flat $100 benchmark",
                "- 1% bankroll conservative",
                "- Kelly capped 2% aggressive",
                "",
                "## Preferred Sizing Decision Path",
                "- safer production choice: 1% bankroll",
                "- aggressive option: Kelly capped 2%",
                "- keep flat $100 as the benchmark reference line",
                "",
                "## Current Deployment Mode",
                "- live shadow tracking",
                "- no automatic real-money execution",
                "",
                "## Post-Card Workflow",
                "1. Refresh completed fight results",
                "2. Settle live_bet_tracking.csv with actual outcomes",
                "3. Update realized profit vs expected profit",
                "4. Record CLV fields: odds_at_pick, closing_odds, line_movement, beat_closing_line_flag",
                "",
                "## Remaining Final Steps",
                "1. Run live shadow mode for 10-20 events",
                "2. Record results after each event",
                "3. Compare live ROI vs backtest ROI",
                "4. Track CLV / closing-line proxy if available",
                "5. Decide final production sizing rule",
                "6. Freeze final strategy documentation",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_live_shadow_update(
    bankroll: float = DEFAULT_BANKROLL,
    settle_tracking: bool = False,
    limit_completed_events: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shadow_base_df = build_shadow_base()
    shadow_df = enrich_shadow_report(shadow_base_df, bankroll=bankroll)
    write_shadow_report(shadow_df)

    tracking_rows_df = build_tracking_rows(shadow_df)
    append_live_tracking(tracking_rows_df)
    latest_df = write_latest_recommendations(shadow_df)
    if settle_tracking:
        settle_live_tracking(latest_shadow_df=shadow_df, limit_completed_events=limit_completed_events)
    write_final_status()
    return shadow_df, latest_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Update the live shadow report with V2_Core stakes and tracking.")
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL, help="Bankroll used for live sizing columns.")
    parser.add_argument(
        "--settle-tracking",
        action="store_true",
        help="Also refresh completed results and settle any matching live tracking rows.",
    )
    parser.add_argument(
        "--limit-completed-events",
        type=int,
        default=25,
        help="How many recently completed events to scrape when settling tracking.",
    )
    args = parser.parse_args()

    shadow_df, latest_df = run_live_shadow_update(
        bankroll=float(args.bankroll),
        settle_tracking=bool(args.settle_tracking),
        limit_completed_events=int(args.limit_completed_events),
    )
    print(f"Updated shadow report: {SHADOW_REPORT_PATH} ({len(shadow_df)} rows)")
    print(f"Current V2_Core recommendations: {len(latest_df)}")
    print(f"Updated live tracking: {LIVE_TRACKING_PATH}")
    print(f"Updated final status: {FINAL_STATUS_PATH}")


if __name__ == "__main__":
    main()
