from __future__ import annotations

"""Scrape fighter-centric recent fight history for upcoming-card fighters."""

import argparse
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .common import (
    DATA_DIR,
    collapse_whitespace,
    extract_row_html_blocks,
    fetch_html,
    log,
    normalize_column_name,
    normalize_fighter_name,
    parse_clock_to_seconds,
    parse_event_date,
    read_html_tables,
    safe_float,
    safe_int,
    write_csv,
)


DEFAULT_OUTPUT_PATH = DATA_DIR / "fighter_recent_history_scraped.csv"


def _parse_space_pair(value: object) -> tuple[float, float]:
    text = collapse_whitespace(value)
    match = re.match(r"^(\d+)\s+(\d+)$", text)
    if not match:
        return np.nan, np.nan
    return float(match.group(1)), float(match.group(2))


def _find_profile_history_table(profile_html: str) -> pd.DataFrame:
    for table in read_html_tables(profile_html):
        df = table.copy()
        df.columns = [normalize_column_name(col) for col in df.columns]
        if ("w/l" in df.columns or "result" in df.columns) and "fighter" in df.columns and "event" in df.columns:
            return df
    return pd.DataFrame()


def _extract_profile_fight_links(profile_html: str) -> list[str]:
    row_blocks = re.findall(
        r"<tr[^>]*class=\"[^\"]*b-fight-details__table-row[^\"]*\"[^>]*>.*?</tr>",
        profile_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    links: list[str] = []
    for row_html in row_blocks:
        match = re.search(r"""href=["'](http://ufcstats\.com/fight-details/[^"']+)["']""", row_html, flags=re.IGNORECASE)
        if not match:
            match = re.search(r"""data-link=["'](http://ufcstats\.com/fight-details/[^"']+)["']""", row_html, flags=re.IGNORECASE)
        if match:
            links.append(match.group(1))
    return links


def _split_event_and_date(event_value: object, explicit_date_value: object) -> tuple[str, str]:
    explicit_date = collapse_whitespace(explicit_date_value)
    if explicit_date:
        return collapse_whitespace(event_value), explicit_date
    event_text = collapse_whitespace(event_value)
    date_match = re.search(
        r"(Jan\.|Feb\.|Mar\.|Apr\.|May|Jun\.|Jul\.|Aug\.|Sep\.|Oct\.|Nov\.|Dec\.)\s+\d{2},\s+\d{4}$",
        event_text,
        flags=re.IGNORECASE,
    )
    if not date_match:
        return event_text, ""
    event_date = collapse_whitespace(date_match.group(0))
    event_name = collapse_whitespace(event_text[: date_match.start()])
    return event_name, event_date


@lru_cache(maxsize=512)
def _fetch_fight_detail_html(fight_url: str) -> str:
    return fetch_html(fight_url)


def _extract_fight_detail_stats(fight_url: str, fighter_name_normalized: str) -> dict[str, object]:
    fight_html = _fetch_fight_detail_html(fight_url)
    totals_table = pd.DataFrame()
    for table in read_html_tables(fight_html):
        df = table.copy()
        df.columns = [normalize_column_name(col) for col in df.columns]
        if "fighter" in df.columns and "sig. str." in df.columns and "total str." in df.columns and "td" in df.columns:
            totals_table = df
            break

    default_stats = {
        "kd": np.nan,
        "sig_str_landed": np.nan,
        "sig_str_attempted": np.nan,
        "total_str_landed": np.nan,
        "total_str_attempted": np.nan,
        "td_landed": np.nan,
        "td_attempted": np.nan,
        "sub_att": np.nan,
        "rev": np.nan,
        "ctrl_seconds": np.nan,
    }
    if totals_table.empty:
        return default_stats

    totals_table["fighter_normalized"] = totals_table["fighter"].map(normalize_fighter_name)
    fighter_row = totals_table[totals_table["fighter_normalized"] == fighter_name_normalized]
    if fighter_row.empty:
        return default_stats
    fighter_row = fighter_row.iloc[0]

    sig_landed, sig_attempted = parse_of_stat(fighter_row.get("sig. str."))
    total_landed, total_attempted = parse_of_stat(fighter_row.get("total str."))
    td_landed, td_attempted = parse_of_stat(fighter_row.get("td"))
    return {
        "kd": safe_float(fighter_row.get("kd")),
        "sig_str_landed": sig_landed,
        "sig_str_attempted": sig_attempted,
        "total_str_landed": total_landed,
        "total_str_attempted": total_attempted,
        "td_landed": td_landed,
        "td_attempted": td_attempted,
        "sub_att": safe_float(fighter_row.get("sub. att")),
        "rev": safe_float(fighter_row.get("rev.")),
        "ctrl_seconds": parse_clock_to_seconds(fighter_row.get("ctrl")),
    }


def _prepare_profile_history_rows(
    fighter_name: str,
    fighter_url: str,
    max_fights_per_fighter: int,
    include_detail_stats: bool,
) -> list[dict[str, object]]:
    profile_html = fetch_html(fighter_url)
    history_table = _find_profile_history_table(profile_html)
    if history_table.empty:
        return []

    fight_links = _extract_profile_fight_links(profile_html)
    aligned_count = min(len(history_table), len(fight_links)) if fight_links else len(history_table)
    history_table = history_table.iloc[:aligned_count].copy().reset_index(drop=True)
    if fight_links:
        history_table["fight_url"] = fight_links[:aligned_count]
    else:
        history_table["fight_url"] = ""

    result_col = "w/l" if "w/l" in history_table.columns else "result"
    event_col = next((col for col in history_table.columns if col == "event"), None)
    date_col = next((col for col in history_table.columns if col == "date"), None)
    method_col = next((col for col in history_table.columns if "method" in col), None)
    round_col = next((col for col in history_table.columns if col in {"round", "rnd"}), None)
    time_col = next((col for col in history_table.columns if col == "time"), None)
    weight_col = next((col for col in history_table.columns if "weight" in col), None)
    fighter_col = "fighter"
    kd_col = next((col for col in history_table.columns if col == "kd"), None)
    str_col = next((col for col in history_table.columns if col == "str"), None)
    td_col = next((col for col in history_table.columns if col == "td"), None)
    sub_col = next((col for col in history_table.columns if col == "sub"), None)

    history_table = history_table[history_table[result_col].astype(str).str.lower().str.strip() != "next"].copy()
    history_table = history_table.head(max_fights_per_fighter).copy()
    if date_col:
        history_table["event_date_parsed"] = pd.to_datetime(history_table[date_col].map(parse_event_date), errors="coerce")
    else:
        history_table["event_date_parsed"] = history_table[event_col].map(lambda value: parse_event_date(_split_event_and_date(value, "")[1]))
    history_table = history_table.sort_values("event_date_parsed", ascending=False).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    fighter_name_normalized = normalize_fighter_name(fighter_name)
    for _, row in history_table.iterrows():
        fight_url = collapse_whitespace(row.get("fight_url", ""))
        stats = _extract_fight_detail_stats(fight_url, fighter_name_normalized) if include_detail_stats and fight_url else {
            "kd": np.nan,
            "sig_str_landed": np.nan,
            "sig_str_attempted": np.nan,
            "total_str_landed": np.nan,
            "total_str_attempted": np.nan,
            "td_landed": np.nan,
            "td_attempted": np.nan,
            "sub_att": np.nan,
            "rev": np.nan,
            "ctrl_seconds": np.nan,
        }
        fighter_cell = collapse_whitespace(row.get(fighter_col, ""))
        opponent_name = fighter_cell
        fighter_name_display = collapse_whitespace(fighter_name)
        if fighter_cell.lower().startswith(fighter_name_display.lower()):
            opponent_name = collapse_whitespace(fighter_cell[len(fighter_name_display) :])
        kd_for, _ = _parse_space_pair(row.get(kd_col, "")) if kd_col else (np.nan, np.nan)
        str_for, str_against = _parse_space_pair(row.get(str_col, "")) if str_col else (np.nan, np.nan)
        td_for, td_against = _parse_space_pair(row.get(td_col, "")) if td_col else (np.nan, np.nan)
        sub_for, _ = _parse_space_pair(row.get(sub_col, "")) if sub_col else (np.nan, np.nan)
        if pd.isna(kd_for) and kd_col:
            kd_for = safe_float(row.get(kd_col))
        if pd.isna(sub_for) and sub_col:
            sub_for = safe_float(row.get(sub_col))
        stats["kd"] = kd_for if pd.isna(stats["kd"]) else stats["kd"]
        stats["sig_str_landed"] = str_for if pd.isna(stats["sig_str_landed"]) else stats["sig_str_landed"]
        stats["opp_sig_str_landed"] = str_against
        stats["td_landed"] = td_for if pd.isna(stats["td_landed"]) else stats["td_landed"]
        stats["opp_td_landed"] = td_against
        stats["sub_att"] = sub_for if pd.isna(stats["sub_att"]) else stats["sub_att"]
        event_name, event_date = _split_event_and_date(row.get(event_col, ""), row.get(date_col, "") if date_col else "")
        rows.append(
            {
                "fighter_name": fighter_name,
                "fighter_name_normalized": fighter_name_normalized,
                "fighter_profile_url": fighter_url,
                "opponent_name": opponent_name,
                "opponent_name_normalized": normalize_fighter_name(opponent_name),
                "event_name": event_name,
                "event_date": event_date,
                "result": collapse_whitespace(row.get(result_col, "")).lower(),
                "weight_class": collapse_whitespace(row.get(weight_col, "")) if weight_col else "",
                "method": collapse_whitespace(row.get(method_col, "")) if method_col else "",
                "round": safe_int(row.get(round_col)) if round_col else np.nan,
                "time": collapse_whitespace(row.get(time_col, "")) if time_col else "",
                "fight_url": fight_url,
                **stats,
            }
        )
    return rows


def scrape_fighter_history(
    upcoming_df: pd.DataFrame,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    max_fights_per_fighter: int = 12,
    include_detail_stats: bool = False,
) -> pd.DataFrame:
    """Scrape fighter-centric recent fight history for all unique fighters on the upcoming card."""
    unique_targets: dict[str, str] = {}
    for fighter_col, url_col in (("fighter_A", "fighter_A_url"), ("fighter_B", "fighter_B_url")):
        if fighter_col not in upcoming_df.columns or url_col not in upcoming_df.columns:
            continue
        for _, row in upcoming_df[[fighter_col, url_col]].dropna(subset=[fighter_col, url_col]).iterrows():
            fighter_name = collapse_whitespace(row[fighter_col])
            fighter_url = collapse_whitespace(row[url_col])
            if fighter_name and fighter_url:
                unique_targets.setdefault(fighter_url, fighter_name)

    history_rows: list[dict[str, object]] = []
    for fighter_url, fighter_name in unique_targets.items():
        try:
            history_rows.extend(
                _prepare_profile_history_rows(
                    fighter_name=fighter_name,
                    fighter_url=fighter_url,
                    max_fights_per_fighter=max_fights_per_fighter,
                    include_detail_stats=include_detail_stats,
                )
            )
        except Exception as exc:  # pragma: no cover - network/source variability
            log(f"warning: fighter history scrape failed for {fighter_name} ({fighter_url}): {exc}")

    history_df = pd.DataFrame(history_rows)
    if not history_df.empty:
        history_df["event_date_parsed"] = pd.to_datetime(history_df["event_date"].map(parse_event_date), errors="coerce")
        history_df = history_df.sort_values(["fighter_name_normalized", "event_date_parsed"], ascending=[True, False]).reset_index(drop=True)
    write_csv(history_df, output_path)
    log(f"history rows scraped: {len(history_df)}")
    return history_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape recent fight history for upcoming-card fighters.")
    parser.add_argument(
        "--upcoming-input",
        type=Path,
        default=DATA_DIR / "upcoming_fights_scraped.csv",
        help="Upcoming fights scrape to source fighter names and profile links.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Where to save fighter history CSV.")
    parser.add_argument("--max-fights-per-fighter", type=int, default=12, help="How many recent fights to keep per fighter.")
    parser.add_argument(
        "--include-detail-stats",
        action="store_true",
        help="Also fetch per-fight detail pages for attempt/control fields. Slower but richer when available.",
    )
    args = parser.parse_args()

    upcoming_df = pd.read_csv(args.upcoming_input)
    df = scrape_fighter_history(
        upcoming_df=upcoming_df,
        output_path=args.output,
        max_fights_per_fighter=args.max_fights_per_fighter,
        include_detail_stats=args.include_detail_stats,
    )
    print(f"Saved fighter history: {args.output}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
