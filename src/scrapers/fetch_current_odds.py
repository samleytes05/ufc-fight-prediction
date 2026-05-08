from __future__ import annotations

"""Scrape current UFC moneylines with raw-book and consensus outputs."""

import argparse
from pathlib import Path
import re

import pandas as pd

from .common import (
    DATA_DIR,
    american_to_implied_probability,
    as_absolute_url,
    build_pair_key,
    collapse_whitespace,
    fetch_html,
    log,
    looks_like_fighter_name,
    normalize_column_name,
    normalize_fighter_name,
    parse_american_odds,
    read_html_tables,
    utc_now_iso,
    write_csv,
)


BEST_FIGHT_ODDS_HOME = "https://www.bestfightodds.com/"
DEFAULT_RAW_OUTPUT_PATH = DATA_DIR / "current_odds_raw.csv"
DEFAULT_CONSENSUS_OUTPUT_PATH = DATA_DIR / "current_odds_scraped.csv"


def _extract_event_links(home_html: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"""href=["'](?P<href>/events/ufc[^"']+)["']""", re.IGNORECASE)
    events: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in pattern.finditer(home_html):
        href = as_absolute_url(BEST_FIGHT_ODDS_HOME, match.group("href"))
        slug = match.group("href").split("/events/")[-1]
        slug = re.sub(r"-\d+$", "", slug)
        label = collapse_whitespace(slug.replace("-", " ").title())
        if href not in seen:
            events.append((label, href))
            seen.add(href)
    return events


def _extract_event_date(page_html: str) -> str:
    match = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(st|nd|rd|th)?[,]?\s+\d{4}",
        page_html,
        flags=re.IGNORECASE,
    )
    return collapse_whitespace(match.group(0)) if match else ""


def _table_to_raw_rows(table: pd.DataFrame, event_name: str, event_date: str, source_url: str, odds_timestamp: str) -> list[dict[str, object]]:
    df = table.copy()
    df.columns = [normalize_column_name(col) for col in df.columns]
    if df.empty:
        return []

    text_cols = [col for col in df.columns if df[col].astype(str).map(looks_like_fighter_name).any()]
    if not text_cols:
        return []
    fighter_col = text_cols[0]
    book_cols = [col for col in df.columns if col != fighter_col]
    if not book_cols:
        return []

    fighter_rows: list[tuple[str, pd.Series]] = []
    for _, row in df.iterrows():
        fighter_name = re.sub(r"^\d+", "", collapse_whitespace(row[fighter_col])).replace("?", "").strip()
        if not looks_like_fighter_name(fighter_name):
            continue
        fighter_rows.append((fighter_name, row))

    raw_rows: list[dict[str, object]] = []
    for idx in range(0, len(fighter_rows) - 1, 2):
        fighter_a, row_a = fighter_rows[idx]
        fighter_b, row_b = fighter_rows[idx + 1]
        matchup_key = build_pair_key(fighter_a, fighter_b)
        for book_col in book_cols:
            odds_a = parse_american_odds(row_a.get(book_col))
            odds_b = parse_american_odds(row_b.get(book_col))
            if pd.isna(odds_a) and pd.isna(odds_b):
                continue
            raw_rows.append(
                {
                    "event_name": event_name,
                    "event_date": event_date,
                    "fighter_A": fighter_a,
                    "fighter_B": fighter_b,
                    "fighter_A_normalized": normalize_fighter_name(fighter_a),
                    "fighter_B_normalized": normalize_fighter_name(fighter_b),
                    "matchup_key": matchup_key,
                    "sportsbook": book_col,
                    "fighter_A_moneyline": odds_a,
                    "fighter_B_moneyline": odds_b,
                    "fighter_A_implied_prob": american_to_implied_probability(odds_a),
                    "fighter_B_implied_prob": american_to_implied_probability(odds_b),
                    "odds_timestamp": odds_timestamp,
                    "source_url": source_url,
                }
            )
    return raw_rows


def _build_consensus_odds(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(
            columns=[
                "event_name",
                "event_date",
                "fighter_A",
                "fighter_B",
                "fighter_A_normalized",
                "fighter_B_normalized",
                "matchup_key",
                "fighter_A_moneyline",
                "fighter_B_moneyline",
                "fighter_A_implied_prob",
                "fighter_B_implied_prob",
                "sportsbook_count",
                "odds_timestamp",
                "source_url",
                "odds_A",
                "odds_B",
            ]
        )

    grouped = (
        raw_df.groupby(
            ["event_name", "event_date", "fighter_A", "fighter_B", "fighter_A_normalized", "fighter_B_normalized", "matchup_key"],
            as_index=False,
        )
        .agg(
            fighter_A_moneyline=("fighter_A_moneyline", "median"),
            fighter_B_moneyline=("fighter_B_moneyline", "median"),
            sportsbook_count=("sportsbook", "nunique"),
            odds_timestamp=("odds_timestamp", "max"),
            source_url=("source_url", "first"),
        )
        .copy()
    )
    grouped["fighter_A_implied_prob"] = grouped["fighter_A_moneyline"].map(american_to_implied_probability)
    grouped["fighter_B_implied_prob"] = grouped["fighter_B_moneyline"].map(american_to_implied_probability)
    grouped["odds_A"] = grouped["fighter_A_moneyline"]
    grouped["odds_B"] = grouped["fighter_B_moneyline"]
    return grouped


def scrape_current_odds(
    raw_output_path: str | Path = DEFAULT_RAW_OUTPUT_PATH,
    consensus_output_path: str | Path = DEFAULT_CONSENSUS_OUTPUT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scrape current UFC odds into raw-book and consensus matchup files."""
    home_html = fetch_html(BEST_FIGHT_ODDS_HOME)
    event_links = _extract_event_links(home_html)
    odds_timestamp = utc_now_iso()

    raw_rows: list[dict[str, object]] = []
    for event_name, event_url in event_links:
        event_html = fetch_html(event_url)
        event_date = _extract_event_date(event_html)
        for table in read_html_tables(event_html):
            raw_rows.extend(_table_to_raw_rows(table, event_name, event_date, event_url, odds_timestamp))

    raw_df = pd.DataFrame(raw_rows)
    if not raw_df.empty:
        raw_df = raw_df.drop_duplicates(
            subset=["event_name", "matchup_key", "sportsbook", "fighter_A_moneyline", "fighter_B_moneyline"],
            keep="first",
        ).reset_index(drop=True)
    consensus_df = _build_consensus_odds(raw_df)
    if not consensus_df.empty:
        consensus_df = consensus_df.sort_values(["event_date", "event_name", "fighter_A", "fighter_B"]).reset_index(drop=True)

    write_csv(raw_df, raw_output_path)
    write_csv(consensus_df, consensus_output_path)
    log(f"odds rows scraped: raw={len(raw_df)} consensus={len(consensus_df)}")
    return raw_df, consensus_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape current UFC betting odds.")
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT_PATH, help="Where to save the raw odds CSV.")
    parser.add_argument(
        "--consensus-output",
        type=Path,
        default=DEFAULT_CONSENSUS_OUTPUT_PATH,
        help="Where to save the consensus odds CSV.",
    )
    args = parser.parse_args()
    raw_df, consensus_df = scrape_current_odds(raw_output_path=args.raw_output, consensus_output_path=args.consensus_output)
    print(f"Saved raw odds: {args.raw_output}")
    print(f"Saved consensus odds: {args.consensus_output}")
    print(f"Rows: raw={len(raw_df)} consensus={len(consensus_df)}")


if __name__ == "__main__":
    main()
