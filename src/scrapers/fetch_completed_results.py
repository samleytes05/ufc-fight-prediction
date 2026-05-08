from __future__ import annotations

"""Scrape recently completed UFC results for tracking updates."""

import argparse
from datetime import date
from pathlib import Path
import re

import pandas as pd

from .common import (
    DATA_DIR,
    extract_labeled_text,
    extract_ufcstats_event_links,
    fetch_html,
    normalize_column_name,
    normalize_fighter_name,
    parse_event_date,
    read_html_tables,
    write_csv,
)


UFCSTATS_COMPLETED_URL = "http://ufcstats.com/statistics/events/completed?page=all"
DEFAULT_OUTPUT_PATH = DATA_DIR / "completed_results_scraped.csv"


def _extract_event_page_metadata(event_html: str) -> dict[str, str]:
    title_match = re.search(
        r"""b-content__title-highlight">\s*(.*?)\s*</span>""",
        event_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return {
        "event_name": " ".join(title_match.group(1).split()) if title_match else "",
        "date": extract_labeled_text(event_html, "Date"),
        "location": extract_labeled_text(event_html, "Location"),
    }


def _find_result_table(event_html: str) -> pd.DataFrame:
    for table in read_html_tables(event_html):
        normalized_columns = [normalize_column_name(col) for col in table.columns]
        fighter_columns = [col for col in normalized_columns if "fighter" in col]
        if len(fighter_columns) >= 2 and any("w/l" in col or col == "w/l" for col in normalized_columns):
            result = table.copy()
            result.columns = normalized_columns
            return result
    return pd.DataFrame()


def _extract_completed_rows(
    result_table: pd.DataFrame,
    event_name: str,
    event_date: str,
    location: str,
    event_url: str,
) -> pd.DataFrame:
    if result_table.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "event_name",
                "location",
                "fighter_A",
                "fighter_B",
                "fighter_A_normalized",
                "fighter_B_normalized",
                "actual_outcome",
                "event_url",
            ]
        )

    fighter_columns = [col for col in result_table.columns if "fighter" in col]
    outcome_col = next(col for col in result_table.columns if "w/l" in col or col == "w/l")
    round_col = next((col for col in result_table.columns if col == "round"), None)
    time_col = next((col for col in result_table.columns if col == "time"), None)
    method_col = next((col for col in result_table.columns if "method" in col), None)

    result_df = result_table[[outcome_col, fighter_columns[0], fighter_columns[1]]].copy()
    result_df = result_df.rename(
        columns={outcome_col: "outcome_flag", fighter_columns[0]: "fighter_A", fighter_columns[1]: "fighter_B"}
    )
    result_df["fighter_A"] = result_df["fighter_A"].astype(str).str.strip()
    result_df["fighter_B"] = result_df["fighter_B"].astype(str).str.strip()
    result_df = result_df[(result_df["fighter_A"] != "") & (result_df["fighter_B"] != "")].copy()
    result_df["actual_outcome"] = (result_df["outcome_flag"].astype(str).str.upper() == "W").astype(int)
    result_df["date"] = event_date
    result_df["event_name"] = event_name
    result_df["location"] = location
    result_df["fighter_A_normalized"] = result_df["fighter_A"].map(normalize_fighter_name)
    result_df["fighter_B_normalized"] = result_df["fighter_B"].map(normalize_fighter_name)
    result_df["event_url"] = event_url
    if method_col is not None:
        result_df["method"] = result_table[method_col].astype(str)
    if round_col is not None:
        result_df["round"] = result_table[round_col]
    if time_col is not None:
        result_df["time"] = result_table[time_col]
    keep_cols = [
        "date",
        "event_name",
        "location",
        "fighter_A",
        "fighter_B",
        "fighter_A_normalized",
        "fighter_B_normalized",
        "actual_outcome",
        "method",
        "round",
        "time",
        "event_url",
    ]
    for col in keep_cols:
        if col not in result_df.columns:
            result_df[col] = ""
    return result_df[keep_cols].copy()


def scrape_completed_results(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    limit_events: int = 25,
) -> pd.DataFrame:
    """Scrape recently completed UFC results for live result updating."""
    page_html = fetch_html(UFCSTATS_COMPLETED_URL)
    event_links = extract_ufcstats_event_links(page_html)
    result_frames: list[pd.DataFrame] = []
    completed_events = 0
    today = pd.Timestamp(date.today())
    for event_url in event_links:
        if limit_events > 0 and completed_events >= limit_events:
            break
        event_html = fetch_html(event_url)
        metadata = _extract_event_page_metadata(event_html)
        event_date = pd.to_datetime(parse_event_date(metadata["date"]), errors="coerce")
        if not pd.isna(event_date) and event_date > today:
            continue
        result_table = _find_result_table(event_html)
        extracted = _extract_completed_rows(
            result_table=result_table,
            event_name=metadata["event_name"],
            event_date=metadata["date"],
            location=metadata["location"],
            event_url=event_url,
        )
        if extracted.empty:
            continue
        completed_events += 1
        result_frames.append(extracted)

    completed_df = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    completed_df = completed_df.drop_duplicates(subset=["event_name", "fighter_A_normalized", "fighter_B_normalized"])
    write_csv(completed_df, output_path)
    return completed_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape recently completed UFC results for live tracking.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Where to save the completed results CSV.")
    parser.add_argument("--limit-events", type=int, default=25, help="How many recent completed events to scrape.")
    args = parser.parse_args()
    df = scrape_completed_results(output_path=args.output, limit_events=args.limit_events)
    print(f"Saved completed results: {args.output}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
