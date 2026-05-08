from __future__ import annotations

"""Merge raw live scrapes into an enriched pre-fight input table."""

from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    DATA_DIR,
    build_pair_key,
    collapse_whitespace,
    inches_to_cm,
    log,
    normalize_fighter_name,
    parse_clock_to_seconds,
    parse_event_date,
    write_csv,
)


DEFAULT_OUTPUT_PATH = DATA_DIR / "upcoming_fights.csv"


OBSERVED_AGGREGATE_COLUMNS = [
    "career_fights_observed",
    "career_wins_observed",
    "career_losses_observed",
    "career_draws_observed",
    "career_win_rate_observed",
    "career_sig_str_landed_per_round_observed",
    "career_sig_str_attempted_per_round_observed",
    "career_total_str_landed_per_round_observed",
    "career_total_str_attempted_per_round_observed",
    "career_td_landed_per_round_observed",
    "career_td_attempted_per_round_observed",
    "career_sub_att_per_round_observed",
    "career_ctrl_seconds_per_round_observed",
    "career_sig_str_absorbed_per_round_observed",
    "career_td_absorbed_per_round_observed",
    "career_sig_str_acc_observed",
    "career_td_acc_observed",
    "career_finish_rate_observed",
    "l3_fights_observed",
    "l3_win_rate_observed",
    "l3_sig_str_landed_per_round_observed",
    "l3_sig_str_attempted_per_round_observed",
    "l3_total_str_landed_per_round_observed",
    "l3_total_str_attempted_per_round_observed",
    "l3_td_landed_per_round_observed",
    "l3_td_attempted_per_round_observed",
    "l3_sub_att_per_round_observed",
    "l3_ctrl_seconds_per_round_observed",
    "l3_sig_str_absorbed_per_round_observed",
    "l3_sig_str_acc_observed",
    "l3_td_acc_observed",
    "days_since_last_fight",
]


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator is None or pd.isna(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _normalize_result(value: object) -> str:
    text = collapse_whitespace(value).lower()
    if text in {"win", "w"}:
        return "win"
    if text in {"loss", "l"}:
        return "loss"
    if text in {"draw", "d"}:
        return "draw"
    if "no contest" in text or text == "nc":
        return "no contest"
    return text


def _is_finish(method: object) -> bool:
    method_text = collapse_whitespace(method).lower()
    if not method_text:
        return False
    return "decision" not in method_text


def _total_fight_seconds(round_value: object, time_value: object) -> float:
    round_number = pd.to_numeric(pd.Series([round_value]), errors="coerce").iloc[0]
    time_seconds = parse_clock_to_seconds(time_value)
    if pd.isna(round_number) or pd.isna(time_seconds):
        return np.nan
    return float(max(round_number - 1, 0) * 300 + time_seconds)


def _aggregate_history_slice(history_df: pd.DataFrame) -> dict[str, float]:
    if history_df.empty:
        return {col: np.nan for col in OBSERVED_AGGREGATE_COLUMNS[:-1]}

    history = history_df.copy()
    history["result_norm"] = history["result"].map(_normalize_result)
    history["fight_seconds"] = history.apply(lambda row: _total_fight_seconds(row.get("round"), row.get("time")), axis=1)
    history["fight_round_equivalents"] = history["fight_seconds"] / 300.0
    history["finish_flag"] = history["method"].map(_is_finish).astype(float)

    rounds = history["fight_round_equivalents"].sum(min_count=1)
    wins = float((history["result_norm"] == "win").sum())
    losses = float((history["result_norm"] == "loss").sum())
    draws = float(history["result_norm"].isin(["draw", "no contest"]).sum())

    return {
        "career_fights_observed": float(len(history)),
        "career_wins_observed": wins,
        "career_losses_observed": losses,
        "career_draws_observed": draws,
        "career_win_rate_observed": _safe_rate(wins, len(history)),
        "career_sig_str_landed_per_round_observed": _safe_rate(history["sig_str_landed"].sum(min_count=1), rounds),
        "career_sig_str_attempted_per_round_observed": _safe_rate(history["sig_str_attempted"].sum(min_count=1), rounds),
        "career_total_str_landed_per_round_observed": _safe_rate(history["total_str_landed"].sum(min_count=1), rounds),
        "career_total_str_attempted_per_round_observed": _safe_rate(history["total_str_attempted"].sum(min_count=1), rounds),
        "career_td_landed_per_round_observed": _safe_rate(history["td_landed"].sum(min_count=1), rounds),
        "career_td_attempted_per_round_observed": _safe_rate(history["td_attempted"].sum(min_count=1), rounds),
        "career_sub_att_per_round_observed": _safe_rate(history["sub_att"].sum(min_count=1), rounds),
        "career_ctrl_seconds_per_round_observed": _safe_rate(history["ctrl_seconds"].sum(min_count=1), rounds),
        "career_sig_str_absorbed_per_round_observed": _safe_rate(history["opp_sig_str_landed"].sum(min_count=1), rounds),
        "career_td_absorbed_per_round_observed": _safe_rate(history["opp_td_landed"].sum(min_count=1), rounds),
        "career_sig_str_acc_observed": _safe_rate(
            history["sig_str_landed"].sum(min_count=1), history["sig_str_attempted"].sum(min_count=1)
        ),
        "career_td_acc_observed": _safe_rate(history["td_landed"].sum(min_count=1), history["td_attempted"].sum(min_count=1)),
        "career_finish_rate_observed": _safe_rate(
            history.loc[history["result_norm"] == "win", "finish_flag"].sum(min_count=1),
            (history["result_norm"] == "win").sum(),
        ),
        "l3_fights_observed": np.nan,
        "l3_win_rate_observed": np.nan,
        "l3_sig_str_landed_per_round_observed": np.nan,
        "l3_sig_str_attempted_per_round_observed": np.nan,
        "l3_total_str_landed_per_round_observed": np.nan,
        "l3_total_str_attempted_per_round_observed": np.nan,
        "l3_td_landed_per_round_observed": np.nan,
        "l3_td_attempted_per_round_observed": np.nan,
        "l3_sub_att_per_round_observed": np.nan,
        "l3_ctrl_seconds_per_round_observed": np.nan,
        "l3_sig_str_absorbed_per_round_observed": np.nan,
        "l3_sig_str_acc_observed": np.nan,
        "l3_td_acc_observed": np.nan,
    }


def build_history_aggregates(history_df: pd.DataFrame, upcoming_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate scraped fighter history into fighter-level pre-fight features."""
    if history_df.empty:
        return pd.DataFrame(columns=["fighter_name_normalized"] + OBSERVED_AGGREGATE_COLUMNS)

    history = history_df.copy()
    history["event_date_parsed"] = pd.to_datetime(history["event_date"].map(parse_event_date), errors="coerce")
    numeric_cols = [
        "sig_str_landed",
        "sig_str_attempted",
        "total_str_landed",
        "total_str_attempted",
        "td_landed",
        "td_attempted",
        "sub_att",
        "ctrl_seconds",
        "round",
    ]
    for col in numeric_cols:
        if col in history.columns:
            history[col] = pd.to_numeric(history[col], errors="coerce")
    for col in ["opp_sig_str_landed", "opp_td_landed"]:
        if col in history.columns:
            history[col] = pd.to_numeric(history[col], errors="coerce")
    history = history.sort_values(["fighter_name_normalized", "event_date_parsed"], ascending=[True, False]).reset_index(drop=True)

    if {"opp_sig_str_landed", "opp_td_landed"}.issubset(history.columns):
        paired = history.copy()
    else:
        paired = history.merge(
            history[
                [
                    "fighter_name_normalized",
                    "event_name",
                    "event_date",
                    "sig_str_landed",
                    "td_landed",
                ]
            ].rename(
                columns={
                    "fighter_name_normalized": "opponent_name_normalized",
                    "sig_str_landed": "opp_sig_str_landed",
                    "td_landed": "opp_td_landed",
                }
            ),
            on=["event_name", "event_date", "opponent_name_normalized"],
            how="left",
        )

    rows: list[dict[str, object]] = []
    upcoming_dates = {}
    if upcoming_df is not None and not upcoming_df.empty:
        for fighter_col in ("fighter_A", "fighter_B"):
            if fighter_col not in upcoming_df.columns:
                continue
            fighter_dates = upcoming_df[[fighter_col, "event_date"]].dropna(subset=[fighter_col]).copy()
            fighter_dates["fighter_normalized"] = fighter_dates[fighter_col].map(normalize_fighter_name)
            fighter_dates["event_date_parsed"] = pd.to_datetime(fighter_dates["event_date"].map(parse_event_date), errors="coerce")
            fighter_dates = fighter_dates.sort_values("event_date_parsed")
            for fighter_name, group in fighter_dates.groupby("fighter_normalized"):
                upcoming_dates[fighter_name] = group["event_date_parsed"].iloc[0]

    for fighter_name, group in paired.groupby("fighter_name_normalized", sort=True):
        career_metrics = _aggregate_history_slice(group)
        l3_metrics = _aggregate_history_slice(group.head(3))
        for key in list(l3_metrics.keys()):
            if key.startswith("career_"):
                l3_metrics[key.replace("career_", "l3_")] = l3_metrics.pop(key)

        last_fight_date = group["event_date_parsed"].dropna().iloc[0] if group["event_date_parsed"].notna().any() else pd.NaT
        next_fight_date = upcoming_dates.get(fighter_name, pd.NaT)
        days_since_last_fight = (
            float((next_fight_date - last_fight_date).days)
            if not pd.isna(next_fight_date) and not pd.isna(last_fight_date)
            else np.nan
        )

        row = {"fighter_name_normalized": fighter_name, **career_metrics, **l3_metrics, "days_since_last_fight": days_since_last_fight}
        row["last_fight_event_date"] = last_fight_date
        rows.append(row)

    aggregates_df = pd.DataFrame(rows)
    aggregates_df = aggregates_df.replace([np.inf, -np.inf], np.nan)
    return aggregates_df


def _rename_with_side(df: pd.DataFrame, key_col: str, side_prefix: str) -> pd.DataFrame:
    rename_map = {col: f"{side_prefix}_{col}" for col in df.columns if col != key_col}
    return df.rename(columns=rename_map)


def _build_stance_matchup(a_stance: object, b_stance: object) -> str:
    a_value = collapse_whitespace(a_stance).lower()
    b_value = collapse_whitespace(b_stance).lower()
    if not a_value or not b_value:
        return ""
    return f"{a_value}_vs_{b_value}"


def build_live_feature_table(
    upcoming_df: pd.DataFrame,
    odds_df: pd.DataFrame | None = None,
    attributes_df: pd.DataFrame | None = None,
    history_df: pd.DataFrame | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Build the enriched upcoming_fights.csv used by the live prediction workflow."""
    if upcoming_df.empty:
        empty_df = pd.DataFrame(columns=["fighter_A", "fighter_B", "odds_A", "odds_B", "date", "event_name"])
        write_csv(empty_df, output_path)
        return empty_df

    live_df = upcoming_df.copy()
    if "matchup_key" not in live_df.columns:
        live_df["matchup_key"] = live_df.apply(lambda row: build_pair_key(row["fighter_A"], row["fighter_B"]), axis=1)
    live_df["event_date_parsed"] = pd.to_datetime(live_df["event_date"].map(parse_event_date), errors="coerce")
    live_df["date"] = live_df["event_date"]

    if odds_df is not None and not odds_df.empty:
        odds_merge_cols = [
            "matchup_key",
            "fighter_A_moneyline",
            "fighter_B_moneyline",
            "fighter_A_implied_prob",
            "fighter_B_implied_prob",
            "sportsbook_count",
            "odds_timestamp",
        ]
        available_odds_cols = [col for col in odds_merge_cols if col in odds_df.columns]
        live_df = live_df.merge(odds_df[available_odds_cols].drop_duplicates(subset=["matchup_key"]), on="matchup_key", how="left")
        live_df["odds_A"] = live_df["fighter_A_moneyline"]
        live_df["odds_B"] = live_df["fighter_B_moneyline"]
    else:
        live_df["odds_A"] = np.nan
        live_df["odds_B"] = np.nan
        live_df["fighter_A_implied_prob"] = np.nan
        live_df["fighter_B_implied_prob"] = np.nan

    if attributes_df is not None and not attributes_df.empty:
        attrs = attributes_df.copy()
        attrs["fighter_name_normalized"] = attrs["fighter_name_normalized"].map(normalize_fighter_name)
        attr_cols = [
            "fighter_name_normalized",
            "fighter_profile_url",
            "height_inches",
            "reach_inches",
            "stance",
            "date_of_birth_raw",
            "age_years",
            "record_wins",
            "record_losses",
            "record_draws",
            "weight_class_history",
        ]
        attrs = attrs[[col for col in attr_cols if col in attrs.columns]].copy()
        live_df["fighter_A_normalized"] = live_df["fighter_A"].map(normalize_fighter_name)
        live_df["fighter_B_normalized"] = live_df["fighter_B"].map(normalize_fighter_name)
        live_df = live_df.merge(_rename_with_side(attrs, "fighter_name_normalized", "A"), left_on="fighter_A_normalized", right_on="fighter_name_normalized", how="left")
        live_df = live_df.drop(columns=["fighter_name_normalized"])
        live_df = live_df.merge(_rename_with_side(attrs, "fighter_name_normalized", "B"), left_on="fighter_B_normalized", right_on="fighter_name_normalized", how="left")
        live_df = live_df.drop(columns=["fighter_name_normalized"])
    else:
        for side in ("A", "B"):
            for col in ["height_inches", "reach_inches", "stance", "date_of_birth_raw", "age_years", "record_wins", "record_losses", "record_draws", "weight_class_history", "fighter_profile_url"]:
                live_df[f"{side}_{col}"] = np.nan if "stance" not in col and "weight_class_history" not in col and "url" not in col else ""

    for side in ("A", "B"):
        for col in ["height_inches", "reach_inches", "age_years", "record_wins", "record_losses", "record_draws"]:
            numeric_col = f"{side}_{col}"
            if numeric_col in live_df.columns:
                live_df[numeric_col] = pd.to_numeric(live_df[numeric_col], errors="coerce")

    live_df["A_height_cms"] = live_df["A_height_inches"].map(inches_to_cm)
    live_df["B_height_cms"] = live_df["B_height_inches"].map(inches_to_cm)
    live_df["A_reach_cms"] = live_df["A_reach_inches"].map(inches_to_cm)
    live_df["B_reach_cms"] = live_df["B_reach_inches"].map(inches_to_cm)
    live_df["A_age"] = live_df["A_age_years"]
    live_df["B_age"] = live_df["B_age_years"]
    live_df["age_diff"] = live_df["A_age"] - live_df["B_age"]
    live_df["height_diff"] = live_df["A_height_cms"] - live_df["B_height_cms"]
    live_df["reach_diff"] = live_df["A_reach_cms"] - live_df["B_reach_cms"]
    live_df["height_diff_inches"] = live_df["A_height_inches"] - live_df["B_height_inches"]
    live_df["reach_diff_inches"] = live_df["A_reach_inches"] - live_df["B_reach_inches"]
    live_df["stance_matchup"] = live_df.apply(lambda row: _build_stance_matchup(row.get("A_stance"), row.get("B_stance")), axis=1)

    if history_df is not None and not history_df.empty:
        aggregates = build_history_aggregates(history_df, upcoming_df=live_df)
        live_df = live_df.merge(_rename_with_side(aggregates, "fighter_name_normalized", "A"), left_on="fighter_A_normalized", right_on="fighter_name_normalized", how="left")
        live_df = live_df.drop(columns=["fighter_name_normalized"])
        live_df = live_df.merge(_rename_with_side(aggregates, "fighter_name_normalized", "B"), left_on="fighter_B_normalized", right_on="fighter_name_normalized", how="left")
        live_df = live_df.drop(columns=["fighter_name_normalized"])
    else:
        for side in ("A", "B"):
            for col in OBSERVED_AGGREGATE_COLUMNS:
                live_df[f"{side}_{col}"] = np.nan

    for side in ("A", "B"):
        live_df[f"{side}_career_fights"] = live_df[f"{side}_record_wins"] + live_df[f"{side}_record_losses"] + live_df[f"{side}_record_draws"]
        live_df[f"{side}_career_wins"] = live_df[f"{side}_record_wins"]
        live_df[f"{side}_career_losses"] = live_df[f"{side}_record_losses"]
        live_df[f"{side}_career_draws"] = live_df[f"{side}_record_draws"]
        live_df[f"{side}_career_win_rate"] = live_df[f"{side}_career_wins"] / live_df[f"{side}_career_fights"]

    live_df["career_fights_diff"] = live_df["A_career_fights"] - live_df["B_career_fights"]
    live_df["career_win_rate_diff_live"] = live_df["A_career_win_rate"] - live_df["B_career_win_rate"]
    live_df["days_since_last_fight_diff"] = live_df["A_days_since_last_fight"] - live_df["B_days_since_last_fight"]

    column_order = [
        "event_name",
        "event_date",
        "date",
        "event_location",
        "weight_class",
        "scheduled_rounds",
        "fighter_A",
        "fighter_B",
        "fighter_A_url",
        "fighter_B_url",
        "matchup_key",
        "odds_A",
        "odds_B",
        "fighter_A_implied_prob",
        "fighter_B_implied_prob",
        "sportsbook_count",
        "odds_timestamp",
        "A_age",
        "B_age",
        "age_diff",
        "A_height_inches",
        "B_height_inches",
        "A_height_cms",
        "B_height_cms",
        "height_diff_inches",
        "height_diff",
        "A_reach_inches",
        "B_reach_inches",
        "A_reach_cms",
        "B_reach_cms",
        "reach_diff_inches",
        "reach_diff",
        "A_stance",
        "B_stance",
        "stance_matchup",
        "A_career_fights",
        "B_career_fights",
        "A_career_wins",
        "B_career_wins",
        "A_career_losses",
        "B_career_losses",
        "A_career_win_rate",
        "B_career_win_rate",
        "career_fights_diff",
        "career_win_rate_diff_live",
        "A_days_since_last_fight",
        "B_days_since_last_fight",
        "days_since_last_fight_diff",
    ]
    ordered_existing = [col for col in column_order if col in live_df.columns]
    remaining_cols = [col for col in live_df.columns if col not in ordered_existing]
    live_df = live_df[ordered_existing + remaining_cols].copy()
    live_df = live_df.replace([np.inf, -np.inf], np.nan)
    write_csv(live_df, output_path)
    return live_df


def validate_live_feature_table(
    live_df: pd.DataFrame,
    key_fields: list[str] | None = None,
) -> None:
    """Print a smoke-test style validation summary for the final live input table."""
    key_fields = key_fields or [
        "fighter_A",
        "fighter_B",
        "odds_A",
        "A_age",
        "B_age",
        "age_diff",
        "A_career_fights",
        "B_career_fights",
        "A_l3_win_rate_observed",
        "B_l3_win_rate_observed",
    ]
    print("Final upcoming_fights.csv validation")
    print(f"  rows: {len(live_df)}")
    print(f"  columns: {', '.join(live_df.columns)}")
    print("  key null counts:")
    for col in key_fields:
        if col in live_df.columns:
            print(f"    {col}: {int(live_df[col].isna().sum())}")

    missing_attr_a = live_df.loc[live_df["A_age"].isna(), "fighter_A"].astype(str).tolist() if "A_age" in live_df.columns else []
    missing_attr_b = live_df.loc[live_df["B_age"].isna(), "fighter_B"].astype(str).tolist() if "B_age" in live_df.columns else []
    missing_odds = live_df.loc[live_df["odds_A"].isna() & live_df["odds_B"].isna(), ["fighter_A", "fighter_B"]]

    if missing_attr_a or missing_attr_b:
        unmatched = sorted(set(missing_attr_a + missing_attr_b))
        print(f"  unmatched fighter attributes: {', '.join(unmatched[:20])}" + (" ..." if len(unmatched) > 20 else ""))
    if not missing_odds.empty:
        matchup_strings = [f"{row.fighter_A} vs {row.fighter_B}" for row in missing_odds.itertuples()]
        print(f"  unmatched odds matchups: {', '.join(matchup_strings[:20])}" + (" ..." if len(matchup_strings) > 20 else ""))

    log("live input validation complete")
