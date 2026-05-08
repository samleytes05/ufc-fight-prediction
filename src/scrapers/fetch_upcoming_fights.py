from __future__ import annotations

"""Scrape upcoming UFC matchup and card metadata from UFCStats."""

import argparse
from pathlib import Path
import re

import pandas as pd

from .common import (
    DATA_DIR,
    build_pair_key,
    collapse_whitespace,
    extract_labeled_text,
    extract_ufcstats_event_links,
    fetch_html,
    log,
    lookup_fighter_directory_entries,
    normalize_column_name,
    normalize_fighter_name,
    parse_event_date,
    read_html_tables,
    safe_int,
    write_csv,
)


UFCSTATS_UPCOMING_URL = "http://ufcstats.com/statistics/events/upcoming"
DEFAULT_OUTPUT_PATH = DATA_DIR / "upcoming_fights_scraped.csv"


def _extract_event_page_metadata(event_html: str) -> dict[str, str]:
    title_match = re.search(
        r"""b-content__title-highlight">\s*(.*?)\s*</span>""",
        event_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    event_name = collapse_whitespace(title_match.group(1)) if title_match else ""
    event_date = extract_labeled_text(event_html, "Date")
    event_location = extract_labeled_text(event_html, "Location")
    return {
        "event_name": event_name,
        "event_date": event_date,
        "event_location": event_location,
    }


def _find_bout_table(event_html: str) -> pd.DataFrame:
    for table in read_html_tables(event_html):
        normalized_columns = [normalize_column_name(col) for col in table.columns]
        fighter_columns = [col for col in normalized_columns if "fighter" in col]
        if len(fighter_columns) >= 1:
            result = table.copy()
            result.columns = normalized_columns
            return result
    return pd.DataFrame()


def _extract_bout_rows_with_links(event_html: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    row_blocks = re.findall(
        r"<tr[^>]*class=\"[^\"]*b-fight-details__table-row[^\"]*\"[^>]*>(.*?)</tr>",
        event_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for row_html in row_blocks:
        if "fighter-details" not in row_html.lower():
            continue
        fight_link_match = re.search(
            r"""data-link=["'](http://ufcstats\.com/fight-details/[^"']+)["']""",
            row_html,
            flags=re.IGNORECASE,
        )
        fighter_link_matches = re.findall(
            r"""href=["'](http://ufcstats\.com/fighter-details/[^"']+)["'][^>]*>\s*([^<]+?)\s*</a>""",
            row_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if len(fighter_link_matches) < 2:
            continue
        weight_match = re.search(
            r"""b-fight-details__table-col[^>]*l-page_align_left[^>]*>\s*<p[^>]*>\s*([^<]+)""",
            row_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        round_matches = re.findall(
            r"""b-fight-details__table-col">\s*<p[^>]*>\s*(\d+)\s*</p>""",
            row_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        rows.append(
            {
                "bout_url": fight_link_match.group(1) if fight_link_match else "",
                "fighter_A_url": fighter_link_matches[0][0],
                "fighter_B_url": fighter_link_matches[1][0],
                "fighter_A_anchor": collapse_whitespace(fighter_link_matches[0][1]),
                "fighter_B_anchor": collapse_whitespace(fighter_link_matches[1][1]),
                "weight_class_html": collapse_whitespace(weight_match.group(1)) if weight_match else "",
                "scheduled_rounds_html": safe_int(round_matches[-1]) if round_matches else pd.NA,
            }
        )
    return rows


def _column_lookup(columns: list[str], keyword: str) -> str | None:
    for col in columns:
        if keyword in col:
            return col
    return None


def _extract_bouts(
    event_table: pd.DataFrame,
    bout_link_rows: list[dict[str, object]],
    event_name: str,
    event_date: str,
    event_location: str,
    event_url: str,
) -> pd.DataFrame:
    if event_table.empty:
        return pd.DataFrame(
            columns=[
                "event_name",
                "event_date",
                "event_location",
                "weight_class",
                "scheduled_rounds",
                "fighter_A",
                "fighter_B",
                "fighter_A_url",
                "fighter_B_url",
                "matchup_key",
                "event_url",
                "bout_url",
            ]
        )

    fighter_columns = [col for col in event_table.columns if "fighter" in col]
    fighter_a_col = fighter_columns[0]
    fighter_b_col = fighter_columns[1] if len(fighter_columns) >= 2 else fighter_columns[0]
    weight_col = _column_lookup(list(event_table.columns), "weight")
    rounds_col = next((col for col in event_table.columns if col in {"round", "rnd", "scheduled rounds"}), None)

    bouts = event_table.copy().reset_index(drop=True)
    aligned_count = min(len(bouts), len(bout_link_rows))
    bouts = bouts.iloc[:aligned_count].copy()
    link_df = pd.DataFrame(bout_link_rows[:aligned_count])
    bouts = pd.concat([bouts.reset_index(drop=True), link_df], axis=1)

    output = pd.DataFrame(
        {
            "event_name": event_name,
            "event_date": event_date,
            "event_location": event_location,
            "weight_class": (
                bouts[weight_col].map(collapse_whitespace)
                if weight_col
                else bouts.get("weight_class_html", pd.Series("", index=bouts.index))
            ),
            "scheduled_rounds": (
                bouts[rounds_col].map(safe_int)
                if rounds_col
                else pd.to_numeric(bouts.get("scheduled_rounds_html", pd.Series(pd.NA, index=bouts.index)), errors="coerce")
            ),
            "fighter_A": bouts["fighter_A_anchor"].where(bouts["fighter_A_anchor"] != "", bouts[fighter_a_col].map(collapse_whitespace)),
            "fighter_B": bouts["fighter_B_anchor"].where(bouts["fighter_B_anchor"] != "", bouts[fighter_b_col].map(collapse_whitespace)),
            "fighter_A_url": bouts["fighter_A_url"].fillna("").astype(str),
            "fighter_B_url": bouts["fighter_B_url"].fillna("").astype(str),
            "event_url": event_url,
            "bout_url": bouts["bout_url"].fillna("").astype(str),
        }
    )
    output["fighter_A_normalized"] = output["fighter_A"].map(normalize_fighter_name)
    output["fighter_B_normalized"] = output["fighter_B"].map(normalize_fighter_name)
    output["matchup_key"] = output.apply(lambda row: build_pair_key(row["fighter_A"], row["fighter_B"]), axis=1)
    output["date"] = output["event_date"]
    output["event_date_parsed"] = pd.to_datetime(output["event_date"].map(parse_event_date), errors="coerce")
    return output[
        [
            "event_name",
            "event_date",
            "date",
            "event_location",
            "weight_class",
            "scheduled_rounds",
            "fighter_A",
            "fighter_B",
            "fighter_A_normalized",
            "fighter_B_normalized",
            "fighter_A_url",
            "fighter_B_url",
            "matchup_key",
            "event_url",
            "bout_url",
        ]
    ].copy()


def scrape_upcoming_matchups(output_path: str | Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    """Scrape upcoming UFC matchups into a clean, matchup-level CSV."""
    page_html = fetch_html(UFCSTATS_UPCOMING_URL)
    event_links = extract_ufcstats_event_links(page_html)

    bout_frames: list[pd.DataFrame] = []
    for event_url in event_links:
        event_html = fetch_html(event_url)
        metadata = _extract_event_page_metadata(event_html)
        bout_table = _find_bout_table(event_html)
        bout_link_rows = _extract_bout_rows_with_links(event_html)
        bout_frames.append(
            _extract_bouts(
                event_table=bout_table,
                bout_link_rows=bout_link_rows,
                event_name=metadata["event_name"],
                event_date=metadata["event_date"],
                event_location=metadata["event_location"],
                event_url=event_url,
            )
        )

    upcoming_df = pd.concat(bout_frames, ignore_index=True) if bout_frames else pd.DataFrame()
    if not upcoming_df.empty:
        unresolved_mask = (upcoming_df["fighter_A_url"] == "") | (upcoming_df["fighter_B_url"] == "")
        if unresolved_mask.any():
            lookup_df = lookup_fighter_directory_entries(
                pd.concat([upcoming_df["fighter_A"], upcoming_df["fighter_B"]], ignore_index=True).tolist()
            )
            if not lookup_df.empty:
                lookup_map = lookup_df.set_index("fighter_name_normalized")["fighter_profile_url"].to_dict()
                upcoming_df.loc[upcoming_df["fighter_A_url"] == "", "fighter_A_url"] = upcoming_df.loc[
                    upcoming_df["fighter_A_url"] == "", "fighter_A_normalized"
                ].map(lookup_map).fillna("")
                upcoming_df.loc[upcoming_df["fighter_B_url"] == "", "fighter_B_url"] = upcoming_df.loc[
                    upcoming_df["fighter_B_url"] == "", "fighter_B_normalized"
                ].map(lookup_map).fillna("")
        upcoming_df = upcoming_df.drop_duplicates(subset=["event_name", "matchup_key"], keep="first")
        upcoming_df = upcoming_df.sort_values(["event_date", "event_name", "fighter_A", "fighter_B"]).reset_index(drop=True)
    write_csv(upcoming_df, output_path)
    log(f"upcoming matchups scraped: {len(upcoming_df)}")
    return upcoming_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape upcoming UFC fight matchups from UFCStats.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Where to save the scraped upcoming fights CSV.")
    args = parser.parse_args()
    df = scrape_upcoming_matchups(output_path=args.output)
    print(f"Saved upcoming fights: {args.output}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
