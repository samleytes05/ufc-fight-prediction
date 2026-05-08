from __future__ import annotations

"""Scrape fighter-level profile attributes for upcoming-card fighters."""

import argparse
from pathlib import Path

import pandas as pd

from .common import (
    DATA_DIR,
    calculate_age_years,
    collapse_whitespace,
    extract_labeled_text,
    fetch_html,
    log,
    lookup_fighter_directory_entries,
    normalize_fighter_name,
    parse_height_to_inches,
    parse_reach_to_inches,
    read_html_tables,
    write_csv,
)


DEFAULT_OUTPUT_PATH = DATA_DIR / "fighter_attributes_scraped.csv"


def _extract_record_from_history(fight_table: pd.DataFrame) -> tuple[float, float, float]:
    result_col = "result" if "result" in fight_table.columns else "w/l" if "w/l" in fight_table.columns else None
    if fight_table.empty or result_col is None:
        return float("nan"), float("nan"), float("nan")
    results = fight_table[result_col].astype(str).str.lower().str.strip()
    results = results.replace({"win": "win", "w": "win", "loss": "loss", "l": "loss", "draw": "draw", "d": "draw"})
    results = results[results != "next"]
    wins = float((results == "win").sum())
    losses = float((results == "loss").sum())
    draws = float(results.isin(["draw", "d", "nc", "no contest"]).sum())
    return wins, losses, draws


def _extract_fight_history_table(profile_html: str) -> pd.DataFrame:
    for table in read_html_tables(profile_html):
        df = table.copy()
        df.columns = [" ".join(str(col).strip().lower().split()) for col in df.columns]
        if ("result" in df.columns or "w/l" in df.columns) and "fighter" in df.columns:
            return df
    return pd.DataFrame()


def _scrape_single_fighter(fighter_name: str, fighter_url: str, default_event_date: pd.Timestamp | None) -> dict[str, object]:
    profile_html = fetch_html(fighter_url)
    history_table = _extract_fight_history_table(profile_html)

    height_raw = extract_labeled_text(profile_html, "Height")
    reach_raw = extract_labeled_text(profile_html, "Reach")
    stance = extract_labeled_text(profile_html, "STANCE") or extract_labeled_text(profile_html, "Stance")
    dob_raw = extract_labeled_text(profile_html, "DOB")

    record_raw = extract_labeled_text(profile_html, "Record")
    record_wins = record_losses = record_draws = float("nan")
    if record_raw:
        record_parts = [part for part in record_raw.replace(" ", "").split("-") if part != ""]
        if len(record_parts) >= 2:
            try:
                record_wins = float(record_parts[0])
                record_losses = float(record_parts[1])
                record_draws = float(record_parts[2]) if len(record_parts) >= 3 else 0.0
            except ValueError:
                record_wins = record_losses = record_draws = float("nan")
    if pd.isna(record_wins):
        record_wins, record_losses, record_draws = _extract_record_from_history(history_table)

    weight_class_history = ""
    if not history_table.empty:
        weight_cols = [col for col in history_table.columns if "weight" in col]
        if weight_cols:
            weights = sorted(
                {
                    collapse_whitespace(value)
                    for value in history_table[weight_cols[0]].dropna().astype(str)
                    if collapse_whitespace(value)
                }
            )
            weight_class_history = " | ".join(weights)

    age_years = calculate_age_years(dob_raw, default_event_date)
    return {
        "fighter_name": fighter_name,
        "fighter_name_normalized": normalize_fighter_name(fighter_name),
        "fighter_profile_url": fighter_url,
        "height_raw": height_raw,
        "height_inches": parse_height_to_inches(height_raw),
        "reach_raw": reach_raw,
        "reach_inches": parse_reach_to_inches(reach_raw),
        "stance": collapse_whitespace(stance) if stance else "",
        "date_of_birth_raw": dob_raw,
        "age_years": age_years,
        "record_wins": record_wins,
        "record_losses": record_losses,
        "record_draws": record_draws,
        "weight_class_history": weight_class_history,
    }


def scrape_fighter_attributes(
    upcoming_df: pd.DataFrame,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Scrape fighter profile attributes for all unique fighters on the upcoming card."""
    fighter_rows: list[dict[str, object]] = []
    candidate_cols = [
        ("fighter_A", "fighter_A_url", "event_date"),
        ("fighter_B", "fighter_B_url", "event_date"),
    ]
    unique_targets: dict[str, tuple[str, pd.Timestamp | None]] = {}
    for name_col, url_col, date_col in candidate_cols:
        if name_col not in upcoming_df.columns or url_col not in upcoming_df.columns:
            continue
        for _, row in upcoming_df[[name_col, url_col] + ([date_col] if date_col in upcoming_df.columns else [])].dropna(subset=[name_col, url_col]).iterrows():
            fighter_name = collapse_whitespace(row[name_col])
            fighter_url = collapse_whitespace(row[url_col])
            if not fighter_name or not fighter_url:
                continue
            default_event_date = pd.to_datetime(row.get(date_col), errors="coerce") if date_col in row else pd.NaT
            unique_targets.setdefault(fighter_url, (fighter_name, default_event_date))

    directory_lookup = lookup_fighter_directory_entries([fighter_name for fighter_name, _ in unique_targets.values()])
    directory_lookup = directory_lookup.set_index("fighter_name_normalized") if not directory_lookup.empty else pd.DataFrame()

    for fighter_url, (fighter_name, default_event_date) in unique_targets.items():
        try:
            fighter_rows.append(_scrape_single_fighter(fighter_name, fighter_url, default_event_date))
        except Exception as exc:  # pragma: no cover - network/source variability
            log(f"warning: fighter profile scrape failed for {fighter_name} ({fighter_url}): {exc}")

    attributes_df = pd.DataFrame(fighter_rows)
    if not attributes_df.empty:
        if not directory_lookup.empty:
            for field in ["height_raw", "reach_raw", "stance", "record_wins", "record_losses", "record_draws"]:
                directory_series = directory_lookup[field] if field in directory_lookup.columns else pd.Series(dtype=object)
                if field in {"height_raw", "reach_raw", "stance"}:
                    attributes_df[field] = attributes_df[field].replace("", pd.NA).combine_first(
                        attributes_df["fighter_name_normalized"].map(directory_series.to_dict())
                    )
                else:
                    attributes_df[field] = pd.to_numeric(attributes_df[field], errors="coerce").combine_first(
                        pd.to_numeric(attributes_df["fighter_name_normalized"].map(directory_series.to_dict()), errors="coerce")
                    )
            attributes_df["height_inches"] = attributes_df["height_inches"].combine_first(attributes_df["height_raw"].map(parse_height_to_inches))
            attributes_df["reach_inches"] = attributes_df["reach_inches"].combine_first(attributes_df["reach_raw"].map(parse_reach_to_inches))
        attributes_df = attributes_df.drop_duplicates(subset=["fighter_name_normalized"], keep="first").reset_index(drop=True)
    write_csv(attributes_df, output_path)
    log(f"fighter profiles found: {len(attributes_df)}")
    return attributes_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape fighter profile attributes for an upcoming card.")
    parser.add_argument(
        "--upcoming-input",
        type=Path,
        default=DATA_DIR / "upcoming_fights_scraped.csv",
        help="Upcoming fights scrape to source fighter names and profile links.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Where to save fighter attributes CSV.")
    args = parser.parse_args()

    upcoming_df = pd.read_csv(args.upcoming_input)
    df = scrape_fighter_attributes(upcoming_df=upcoming_df, output_path=args.output)
    print(f"Saved fighter attributes: {args.output}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
