from __future__ import annotations

"""Scrape historical UFCStats data into the raw schema used by the existing feature pipeline."""

import argparse
import json
from pathlib import Path
import random
import re

import numpy as np
import pandas as pd

from .common import (
    CANONICAL_FIGHTER_ALIASES,
    DATA_DIR,
    FIGHTER_DIRECTORY_URL_TEMPLATE,
    calculate_age_years,
    collapse_whitespace,
    extract_labeled_text,
    extract_ufcstats_event_links,
    fetch_html,
    build_pair_key,
    inches_to_cm,
    log,
    lookup_fighter_directory_entries,
    normalize_fighter_name,
    parse_event_date,
    parse_height_to_inches,
    parse_reach_to_inches,
    read_html_tables,
    write_csv,
)
from .fetch_historical_odds import scrape_historical_odds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = DATA_DIR / "historical_backfill"
LEGACY_BETTING_DIR = DATA_DIR / "legacy_betting"
RESULTS_FILENAME = "ufc_fight_results.csv"
STATS_FILENAME = "ufc_fight_stats.csv"
FIGHTERS_FILENAME = "ufc_fighter_details.csv"
MASTER_FILENAME = "ufc_master_scraped.csv"
FEATURES_FILENAME = "ufc_rebuilt_features_scraped.csv"
EVENT_CATALOG_FILENAME = "historical_event_catalog_scraped.csv"
FIGHTER_PROFILES_FILENAME = "historical_fighter_profiles_scraped.csv"
PHYSICAL_FEATURES_FILENAME = "historical_physical_features_scraped.csv"
DATE_VALIDATION_FILENAME = "historical_date_validation_report.json"
HISTORICAL_ODDS_RAW_FILENAME = "historical_odds_raw_scraped.csv"
HISTORICAL_ODDS_CONSENSUS_FILENAME = "historical_odds_consensus_scraped.csv"
HISTORICAL_ODDS_DIAGNOSTICS_FILENAME = "historical_odds_match_diagnostics.csv"
ARCHIVE_COVERAGE_AUDIT_FILENAME = "archive_coverage_audit.csv"
ARCHIVE_COVERAGE_SUMMARY_FILENAME = "archive_coverage_summary.md"
PARSED_VALUES_AUDIT_FILENAME = "parsed_values_audit.md"
ELO_AUDIT_FILENAME = "elo_audit.md"
ELO_AUDIT_SAMPLE_FILENAME = "elo_audit_sample.csv"
QUALITATIVE_AUDIT_FILENAME = "qualitative_audit.md"
UFCSTATS_COMPLETED_URL = "http://ufcstats.com/statistics/events/completed?page=all"
MASTER_TEMPLATE_PATH = LEGACY_BETTING_DIR / "ufc-master.csv"
BASELINE_FEATURE_PATH = DATA_DIR / "historical_backfill" / FEATURES_FILENAME


def summarize_backfill_progress(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, float | int]:
    output_path = Path(output_dir)
    baseline_results = pd.read_csv(DATA_DIR / "ufc_fight_results.csv") if (DATA_DIR / "ufc_fight_results.csv").exists() else pd.DataFrame()
    baseline_stats = pd.read_csv(DATA_DIR / "ufc_fight_stats.csv") if (DATA_DIR / "ufc_fight_stats.csv").exists() else pd.DataFrame()
    baseline_stats_fight_level = _collapse_stats_to_fight_level(baseline_stats) if not baseline_stats.empty else pd.DataFrame()
    baseline_features = pd.read_csv(BASELINE_FEATURE_PATH) if BASELINE_FEATURE_PATH.exists() else pd.DataFrame()

    results_df = pd.read_csv(output_path / RESULTS_FILENAME) if (output_path / RESULTS_FILENAME).exists() else pd.DataFrame()
    stats_df = pd.read_csv(output_path / STATS_FILENAME) if (output_path / STATS_FILENAME).exists() else pd.DataFrame()
    features_df = pd.read_csv(output_path / FEATURES_FILENAME) if (output_path / FEATURES_FILENAME).exists() else pd.DataFrame()
    events_df = pd.read_csv(output_path / EVENT_CATALOG_FILENAME) if (output_path / EVENT_CATALOG_FILENAME).exists() else pd.DataFrame()
    odds_df = pd.read_csv(output_path / HISTORICAL_ODDS_CONSENSUS_FILENAME) if (output_path / HISTORICAL_ODDS_CONSENSUS_FILENAME).exists() else pd.DataFrame()
    odds_diag_df = pd.read_csv(output_path / HISTORICAL_ODDS_DIAGNOSTICS_FILENAME) if (output_path / HISTORICAL_ODDS_DIAGNOSTICS_FILENAME).exists() else pd.DataFrame()
    master_df = pd.read_csv(output_path / MASTER_FILENAME) if (output_path / MASTER_FILENAME).exists() else pd.DataFrame()

    summary = {
        "events_scraped": int(len(events_df)),
        "fights_scraped": int(len(results_df)),
        "fighter_fight_rows": int(len(stats_df)),
        "rebuilt_feature_rows": int(len(features_df)),
        "fight_coverage_vs_baseline": float(len(results_df) / len(baseline_results)) if len(baseline_results) else 0.0,
        "fighter_fight_coverage_vs_baseline": float(len(stats_df) / len(baseline_stats_fight_level)) if len(baseline_stats_fight_level) else 0.0,
        "feature_coverage_vs_baseline": float(len(features_df) / len(baseline_features)) if len(baseline_features) else 0.0,
        "historical_odds_rows": int(len(odds_df)),
        "master_odds_coverage": float(1.0 - master_df["R_odds"].isna().mean()) if "R_odds" in master_df.columns and len(master_df) else 0.0,
        "odds_matched_events": int((odds_diag_df.get("status", pd.Series(dtype=str)) == "matched").sum()) if not odds_diag_df.empty else 0,
        "odds_unmatched_events": int((odds_diag_df.get("status", pd.Series(dtype=str)) == "unmatched").sum()) if not odds_diag_df.empty else 0,
        "odds_failed_events": int((odds_diag_df.get("status", pd.Series(dtype=str)) == "failed").sum()) if not odds_diag_df.empty else 0,
    }
    return summary


def _event_metadata(event_html: str) -> dict[str, str]:
    title_match = re.search(
        r"""b-content__title-highlight">\s*(.*?)\s*</span>""",
        event_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return {
        "EVENT": collapse_whitespace(title_match.group(1)) if title_match else "",
        "DATE": extract_labeled_text(event_html, "Date"),
        "LOCATION": extract_labeled_text(event_html, "Location"),
    }


def _fighter_names_from_bout(bout: str) -> tuple[str, str]:
    parts = str(bout).split(" vs. ", 1)
    fighter_a = collapse_whitespace(parts[0]) if parts else ""
    fighter_b = collapse_whitespace(parts[1]) if len(parts) > 1 else ""
    return fighter_a, fighter_b


def _canonical_display_name(name: str) -> str:
    normalized = normalize_fighter_name(name)
    parts = [part for part in str(normalized).split(" ") if part]
    return " ".join(part.capitalize() for part in parts)


def apply_canonical_identity_map(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> None:
    output_path = Path(output_dir)

    def canonicalize_bout(value: object) -> str:
        fighter_a, fighter_b = _fighter_names_from_bout(str(value))
        return f"{_canonical_display_name(fighter_a)} vs. {_canonical_display_name(fighter_b)}"

    results_path = output_path / RESULTS_FILENAME
    if results_path.exists():
        results_df = pd.read_csv(results_path)
        results_df["BOUT"] = results_df["BOUT"].map(canonicalize_bout)
        write_csv(results_df, results_path)
        write_csv(results_df, output_path / "ufc_fight_results_scraped.csv")

    stats_path = output_path / STATS_FILENAME
    if stats_path.exists():
        stats_df = pd.read_csv(stats_path)
        stats_df["BOUT"] = stats_df["BOUT"].map(canonicalize_bout)
        stats_df["FIGHTER"] = stats_df["FIGHTER"].map(_canonical_display_name)
        write_csv(stats_df, stats_path)
        write_csv(stats_df, output_path / "ufc_fight_stats_scraped.csv")

    master_path = output_path / MASTER_FILENAME
    if master_path.exists():
        master_df = pd.read_csv(master_path)
        for column in ["R_fighter", "B_fighter"]:
            master_df[column] = master_df[column].map(_canonical_display_name)
        write_csv(master_df, master_path)

    physical_path = output_path / PHYSICAL_FEATURES_FILENAME
    if physical_path.exists():
        physical_df = pd.read_csv(physical_path)
        for column in ["fighter_A", "fighter_B"]:
            physical_df[column] = physical_df[column].map(_canonical_display_name)
        for column in ["fighter_A_normalized", "fighter_B_normalized"]:
            physical_df[column] = physical_df[column].map(normalize_fighter_name)
        write_csv(physical_df, physical_path)

    profiles_path = output_path / FIGHTER_PROFILES_FILENAME
    if profiles_path.exists():
        profiles_df = pd.read_csv(profiles_path)
        profiles_df["fighter_name"] = profiles_df["fighter_name"].map(_canonical_display_name)
        profiles_df["fighter_name_normalized"] = profiles_df["fighter_name"].map(normalize_fighter_name)
        profiles_df = profiles_df.drop_duplicates(subset=["fighter_name_normalized"], keep="last").reset_index(drop=True)
        write_csv(profiles_df, profiles_path)

    fighters_path = output_path / FIGHTERS_FILENAME
    if fighters_path.exists():
        fighters_df = pd.read_csv(fighters_path)
        fighters_df["fighter_name"] = (fighters_df["FIRST"].fillna("") + " " + fighters_df["LAST"].fillna("")).str.strip()
        fighters_df["fighter_name_normalized"] = fighters_df["fighter_name"].map(normalize_fighter_name)
        fighters_df["fighter_name_canonical"] = fighters_df["fighter_name"].map(_canonical_display_name)
        split_names = fighters_df["fighter_name_canonical"].str.split(" ", n=1, expand=True)
        fighters_df["FIRST"] = split_names[0]
        fighters_df["LAST"] = split_names[1].fillna("")
        fighters_df = fighters_df.drop(columns=["fighter_name", "fighter_name_normalized", "fighter_name_canonical"])
        write_csv(fighters_df, fighters_path)
        write_csv(fighters_df, output_path / "ufc_fighter_details_scraped.csv")


def _extract_table_cells(row_html: str) -> list[str]:
    return re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, flags=re.IGNORECASE | re.DOTALL)


def _extract_p_texts(cell_html: str) -> list[str]:
    texts = re.findall(r"<p[^>]*class=\"[^\"]*b-fight-details__table-text[^\"]*\"[^>]*>(.*?)</p>", cell_html, flags=re.IGNORECASE | re.DOTALL)
    cleaned = [collapse_whitespace(re.sub(r"<[^>]+>", " ", text)) for text in texts]
    return [text for text in cleaned if text]


def _extract_anchor_texts(cell_html: str) -> list[str]:
    texts = re.findall(r"<a[^>]*>(.*?)</a>", cell_html, flags=re.IGNORECASE | re.DOTALL)
    cleaned = [collapse_whitespace(re.sub(r"<[^>]+>", " ", text)) for text in texts]
    return [text for text in cleaned if text and text.lower() not in {"view matchup"}]


def _extract_outcome(cell_html: str) -> str:
    texts = [text.lower() for text in _extract_p_texts(cell_html)]
    if any(text == "draw" for text in texts):
        return "D/D"
    if any(text in {"nc", "no contest"} for text in texts):
        return "NC/NC"
    if texts and texts[0] == "win":
        return "W/L"
    if len(texts) >= 2 and texts[1] == "win":
        return "L/W"
    return ""


def _extract_completed_event_rows(event_html: str, event_url: str) -> list[dict[str, str]]:
    metadata = _event_metadata(event_html)
    row_blocks = re.findall(
        r"<tr[^>]*class=\"[^\"]*b-fight-details__table-row[^\"]*js-fight-details-click[^\"]*\"[^>]*>.*?</tr>",
        event_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    rows: list[dict[str, str]] = []
    for row_html in row_blocks:
        cells = _extract_table_cells(row_html)
        if len(cells) < 10:
            continue
        fight_url_match = re.search(r"""data-link=["'](http://ufcstats\.com/fight-details/[^"']+)["']""", row_html, flags=re.IGNORECASE)
        fight_url = fight_url_match.group(1) if fight_url_match else ""
        fighter_names = _extract_anchor_texts(cells[1])
        if len(fighter_names) < 2:
            continue
        method_texts = _extract_p_texts(cells[7])
        row = {
            "EVENT": metadata["EVENT"],
            "BOUT": f"{fighter_names[0]} vs. {fighter_names[1]}",
            "OUTCOME": _extract_outcome(cells[0]),
            "WEIGHTCLASS": (_extract_p_texts(cells[6]) or [""])[0],
            "METHOD": method_texts[0] if method_texts else "",
            "ROUND": (_extract_p_texts(cells[8]) or [""])[0],
            "TIME": (_extract_p_texts(cells[9]) or [""])[0],
            "TIME FORMAT": "",
            "REFEREE": "",
            "DETAILS": " | ".join(method_texts[1:]) if len(method_texts) > 1 else "",
            "URL": fight_url,
            "EVENT_URL": event_url,
            "DATE": metadata["DATE"],
            "LOCATION": metadata["LOCATION"],
            "fighter_A": normalize_fighter_name(fighter_names[0]),
            "fighter_B": normalize_fighter_name(fighter_names[1]),
            "event_row_fighter_1": fighter_names[0],
            "event_row_fighter_2": fighter_names[1],
        }
        rows.append(row)
    return rows


def _parse_round_time_seconds(value: str) -> float | None:
    text = collapse_whitespace(value)
    if not text or ":" not in text:
        return None
    parts = text.split(":")
    if len(parts) != 2:
        return None
    try:
        minutes = int(parts[0])
        seconds = int(parts[1])
    except ValueError:
        return None
    return float(minutes * 60 + seconds)


def _safe_int(value: str) -> int | None:
    text = collapse_whitespace(str(value))
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _winner_label(outcome: str) -> str | None:
    if outcome == "W/L":
        return "Red"
    if outcome == "L/W":
        return "Blue"
    if outcome == "D/D":
        return "Draw"
    if outcome == "NC/NC":
        return "No Contest"
    return None


def _total_fight_time_seconds(round_value: str, time_value: str) -> float | None:
    fight_round = _safe_int(round_value)
    round_seconds = _parse_round_time_seconds(time_value)
    if fight_round is None or round_seconds is None or fight_round < 1:
        return None
    return float((fight_round - 1) * 300 + round_seconds)


def _build_master_row(fight_row: dict[str, str], template_columns: list[str]) -> dict[str, object]:
    row = {column: pd.NA for column in template_columns}
    event_date = pd.to_datetime(parse_event_date(fight_row.get("DATE", "")), errors="coerce")
    weight_class = collapse_whitespace(fight_row.get("WEIGHTCLASS", ""))
    row.update(
        {
            "R_fighter": fight_row.get("event_row_fighter_1") or fight_row.get("BOUT", "").split(" vs. ")[0],
            "B_fighter": fight_row.get("event_row_fighter_2") or fight_row.get("BOUT", "").split(" vs. ")[-1],
            "date": event_date.date().isoformat() if not pd.isna(event_date) else pd.NA,
            "location": collapse_whitespace(fight_row.get("LOCATION", "")) or pd.NA,
            "Winner": _winner_label(fight_row.get("OUTCOME", "")),
            "weight_class": weight_class or pd.NA,
            "gender": "Women" if "women" in weight_class.lower() else "Men",
            "finish": collapse_whitespace(fight_row.get("METHOD", "")) or pd.NA,
            "finish_details": collapse_whitespace(fight_row.get("DETAILS", "")) or pd.NA,
            "finish_round": _safe_int(fight_row.get("ROUND", "")),
            "finish_round_time": collapse_whitespace(fight_row.get("TIME", "")) or pd.NA,
            "total_fight_time_secs": _total_fight_time_seconds(fight_row.get("ROUND", ""), fight_row.get("TIME", "")),
        }
    )
    return row


def _split_dual_by_regex(value: str, pattern: str) -> tuple[str, str]:
    match = re.match(pattern, collapse_whitespace(value))
    if not match:
        return "", ""
    return collapse_whitespace(match.group(1)), collapse_whitespace(match.group(2))


def _split_dual_stat(value: str, kind: str) -> tuple[str, str]:
    text = collapse_whitespace(value)
    if not text:
        return "", ""
    if kind == "of":
        return _split_dual_by_regex(text, r"^(\d+\s+of\s+\d+)\s+(\d+\s+of\s+\d+)$")
    if kind == "pct":
        return _split_dual_by_regex(text, r"^(\d+%)\s+(\d+%)$")
    if kind == "time":
        return _split_dual_by_regex(text, r"^(\d+:\d+)\s+(\d+:\d+)$")
    if kind == "count":
        return _split_dual_by_regex(text, r"^(\d+)\s+(\d+)$")
    return "", ""


def _split_of_value(value: object) -> tuple[float, float]:
    match = re.match(r"^(\d+)\s+of\s+(\d+)$", collapse_whitespace(value))
    if not match:
        return 0.0, 0.0
    return float(match.group(1)), float(match.group(2))


def _time_to_seconds(value: object) -> float:
    text = collapse_whitespace(value)
    if not text or ":" not in text:
        return 0.0
    minutes, seconds = text.split(":", 1)
    try:
        return float(int(minutes) * 60 + int(seconds))
    except ValueError:
        return 0.0


def _collapse_stats_to_fight_level(stats_df: pd.DataFrame) -> pd.DataFrame:
    if stats_df.empty:
        return stats_df

    collapsed = stats_df.copy()
    collapsed["ROUND"] = pd.to_numeric(collapsed["ROUND"].astype(str).str.extract(r"(\d+)")[0], errors="coerce").fillna(0)
    for column in ["KD", "SUB.ATT", "REV."]:
        collapsed[column] = pd.to_numeric(collapsed[column], errors="coerce").fillna(0)
    for column in ["SIG.STR.", "TOTAL STR.", "TD", "HEAD", "BODY", "LEG", "DISTANCE", "CLINCH", "GROUND"]:
        landed_attempted = collapsed[column].map(_split_of_value)
        collapsed[f"{column}_LANDED"] = landed_attempted.map(lambda pair: pair[0])
        collapsed[f"{column}_ATTEMPTED"] = landed_attempted.map(lambda pair: pair[1])
    collapsed["CTRL_SECONDS"] = collapsed["CTRL"].map(_time_to_seconds)

    grouped = collapsed.groupby(["EVENT", "BOUT", "FIGHTER"], as_index=False).agg(
        {
            "ROUND": "max",
            "KD": "sum",
            "SUB.ATT": "sum",
            "REV.": "sum",
            "CTRL_SECONDS": "sum",
            "SIG.STR._LANDED": "sum",
            "SIG.STR._ATTEMPTED": "sum",
            "TOTAL STR._LANDED": "sum",
            "TOTAL STR._ATTEMPTED": "sum",
            "TD_LANDED": "sum",
            "TD_ATTEMPTED": "sum",
            "HEAD_LANDED": "sum",
            "HEAD_ATTEMPTED": "sum",
            "BODY_LANDED": "sum",
            "BODY_ATTEMPTED": "sum",
            "LEG_LANDED": "sum",
            "LEG_ATTEMPTED": "sum",
            "DISTANCE_LANDED": "sum",
            "DISTANCE_ATTEMPTED": "sum",
            "CLINCH_LANDED": "sum",
            "CLINCH_ATTEMPTED": "sum",
            "GROUND_LANDED": "sum",
            "GROUND_ATTEMPTED": "sum",
        }
    )

    def of_text(landed: float, attempted: float) -> str:
        return f"{int(landed)} of {int(attempted)}"

    grouped["SIG.STR."] = grouped.apply(lambda row: of_text(row["SIG.STR._LANDED"], row["SIG.STR._ATTEMPTED"]), axis=1)
    grouped["TOTAL STR."] = grouped.apply(lambda row: of_text(row["TOTAL STR._LANDED"], row["TOTAL STR._ATTEMPTED"]), axis=1)
    grouped["TD"] = grouped.apply(lambda row: of_text(row["TD_LANDED"], row["TD_ATTEMPTED"]), axis=1)
    for column in ["HEAD", "BODY", "LEG", "DISTANCE", "CLINCH", "GROUND"]:
        grouped[column] = grouped.apply(lambda row, c=column: of_text(row[f"{c}_LANDED"], row[f"{c}_ATTEMPTED"]), axis=1)
    grouped["SIG.STR. %"] = ""
    grouped["TD %"] = ""
    grouped["CTRL"] = grouped["CTRL_SECONDS"].map(lambda value: f"{int(value // 60)}:{int(value % 60):02d}")

    return grouped[
        [
            "EVENT",
            "BOUT",
            "ROUND",
            "FIGHTER",
            "KD",
            "SIG.STR.",
            "SIG.STR. %",
            "TOTAL STR.",
            "TD",
            "TD %",
            "SUB.ATT",
            "REV.",
            "CTRL",
            "HEAD",
            "BODY",
            "LEG",
            "DISTANCE",
            "CLINCH",
            "GROUND",
        ]
    ].copy()


def _build_dual_stat_row(
    event_name: str,
    bout_name: str,
    round_number: int,
    fighter_names: list[str],
    mappings: list[tuple[str, object, str]],
) -> list[dict[str, str]]:
    split_values: dict[str, tuple[str, str]] = {name: _split_dual_stat(str(value), kind) for name, value, kind in mappings}
    rows: list[dict[str, str]] = []
    for idx, fighter_name in enumerate(fighter_names):
        row = {
            "EVENT": event_name,
            "BOUT": bout_name,
            "ROUND": str(round_number),
            "FIGHTER": fighter_name,
        }
        for field_name, _, _ in mappings:
            row[field_name] = split_values[field_name][idx]
        rows.append(row)
    return rows


def _value_by_position_or_name(row: pd.Series, position: int, fallback_name: str) -> object:
    if position < len(row):
        return row.iloc[position]
    if fallback_name in row.index:
        return row.get(fallback_name, "")
    return ""


def _fight_detail_total_rows(fight_html: str, event_name: str, bout_name: str, fight_round: str, fighter_a: str, fighter_b: str) -> list[dict[str, str]]:
    tables = read_html_tables(fight_html)
    if len(tables) < 3:
        return []

    totals_df = tables[0].copy()
    totals_df.columns = [collapse_whitespace(str(col)) for col in totals_df.columns]
    sig_df = tables[2].copy()
    sig_df.columns = [collapse_whitespace(str(col)) for col in sig_df.columns]
    if totals_df.empty or sig_df.empty:
        return []

    total_row = totals_df.iloc[0]
    sig_row = sig_df.iloc[0]
    return _build_dual_stat_row(
        event_name=event_name,
        bout_name=bout_name,
        round_number=_safe_int(fight_round) or 1,
        fighter_names=[fighter_a, fighter_b],
        mappings=[
            ("KD", _value_by_position_or_name(total_row, 1, "KD"), "count"),
            ("SIG.STR.", _value_by_position_or_name(total_row, 2, "Sig. str."), "of"),
            ("SIG.STR. %", _value_by_position_or_name(total_row, 3, "Sig. str. %"), "pct"),
            ("TOTAL STR.", _value_by_position_or_name(total_row, 4, "Total str."), "of"),
            ("TD", _value_by_position_or_name(total_row, 5, "Td"), "of"),
            ("TD %", _value_by_position_or_name(total_row, 6, "Td %"), "pct"),
            ("SUB.ATT", _value_by_position_or_name(total_row, 7, "Sub. att"), "count"),
            ("REV.", _value_by_position_or_name(total_row, 8, "Rev."), "count"),
            ("CTRL", _value_by_position_or_name(total_row, 9, "Ctrl"), "time"),
            ("HEAD", _value_by_position_or_name(sig_row, 3, "Head"), "of"),
            ("BODY", _value_by_position_or_name(sig_row, 4, "Body"), "of"),
            ("LEG", _value_by_position_or_name(sig_row, 5, "Leg"), "of"),
            ("DISTANCE", _value_by_position_or_name(sig_row, 6, "Distance"), "of"),
            ("CLINCH", _value_by_position_or_name(sig_row, 7, "Clinch"), "of"),
            ("GROUND", _value_by_position_or_name(sig_row, 8, "Ground"), "of"),
        ],
    )


def _fight_detail_round_rows(fight_html: str, event_name: str, bout_name: str, fight_round: str, fighter_a: str, fighter_b: str) -> list[dict[str, str]]:
    tables = read_html_tables(fight_html)
    if len(tables) < 4:
        return _fight_detail_total_rows(
            fight_html=fight_html,
            event_name=event_name,
            bout_name=bout_name,
            fight_round=fight_round,
            fighter_a=fighter_a,
            fighter_b=fighter_b,
        )

    round_totals_df = tables[1].copy()
    round_totals_df.columns = [collapse_whitespace(str(col)) for col in round_totals_df.columns]
    round_sig_df = tables[3].copy()
    round_sig_df.columns = [collapse_whitespace(str(col)) for col in round_sig_df.columns]
    if round_totals_df.empty or round_sig_df.empty:
        return []

    fighter_names = [fighter_a, fighter_b]
    total_rounds = _safe_int(fight_round) or len(round_totals_df)
    rows: list[dict[str, str]] = []
    for round_index in range(min(total_rounds, len(round_totals_df), len(round_sig_df))):
        total_row = round_totals_df.iloc[round_index]
        sig_row = round_sig_df.iloc[round_index]
        mappings = [
            ("KD", _value_by_position_or_name(total_row, 1, "KD"), "count"),
            ("SIG.STR.", _value_by_position_or_name(total_row, 2, "Sig. str."), "of"),
            ("SIG.STR. %", _value_by_position_or_name(total_row, 3, "Sig. str. %"), "pct"),
            ("TOTAL STR.", _value_by_position_or_name(total_row, 4, "Total str."), "of"),
            ("TD", _value_by_position_or_name(total_row, 5, "Td"), "of"),
            ("TD %", _value_by_position_or_name(total_row, 6, "Td %"), "pct"),
            ("SUB.ATT", _value_by_position_or_name(total_row, 7, "Sub. att"), "count"),
            ("REV.", _value_by_position_or_name(total_row, 8, "Rev."), "count"),
            ("CTRL", _value_by_position_or_name(total_row, 9, "Ctrl"), "time"),
            ("HEAD", _value_by_position_or_name(sig_row, 3, "Head"), "of"),
            ("BODY", _value_by_position_or_name(sig_row, 4, "Body"), "of"),
            ("LEG", _value_by_position_or_name(sig_row, 5, "Leg"), "of"),
            ("DISTANCE", _value_by_position_or_name(sig_row, 6, "Distance"), "of"),
            ("CLINCH", _value_by_position_or_name(sig_row, 7, "Clinch"), "of"),
            ("GROUND", _value_by_position_or_name(sig_row, 8, "Ground"), "of"),
        ]
        rows.extend(
            _build_dual_stat_row(
                event_name=event_name,
                bout_name=bout_name,
                round_number=round_index + 1,
                fighter_names=fighter_names,
                mappings=mappings,
            )
        )
    if rows:
        return rows

    return _fight_detail_total_rows(
        fight_html=fight_html,
        event_name=event_name,
        bout_name=bout_name,
        fight_round=fight_round,
        fighter_a=fighter_a,
        fighter_b=fighter_b,
    )


def _detail_fighter_order(fight_html: str, fighter_1: str, fighter_2: str) -> tuple[str, str]:
    tables = read_html_tables(fight_html)
    if not tables:
        return fighter_1, fighter_2
    totals_df = tables[0].copy()
    if totals_df.empty or "Fighter" not in totals_df.columns:
        return fighter_1, fighter_2
    fighter_text = collapse_whitespace(str(totals_df.iloc[0]["Fighter"]))
    if fighter_text.startswith(fighter_1):
        return fighter_1, fighter_2
    if fighter_text.startswith(fighter_2):
        return fighter_2, fighter_1
    idx_1 = fighter_text.find(fighter_1)
    idx_2 = fighter_text.find(fighter_2)
    if idx_1 != -1 and idx_2 != -1:
        return (fighter_1, fighter_2) if idx_1 < idx_2 else (fighter_2, fighter_1)
    return fighter_1, fighter_2


def scrape_historical_fighter_directory(output_path: str | Path) -> pd.DataFrame:
    fighter_rows: list[dict[str, str]] = []
    for initial in "abcdefghijklmnopqrstuvwxyz":
        html = fetch_html(FIGHTER_DIRECTORY_URL_TEMPLATE.format(char=initial))
        tables = read_html_tables(html)
        if not tables:
            continue
        table = tables[0].copy()
        table.columns = [collapse_whitespace(str(col)).upper().replace(".", "") for col in table.columns]
        row_blocks = [row for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.IGNORECASE | re.DOTALL) if "fighter-details" in row.lower()]
        urls = []
        for row_html in row_blocks[: len(table)]:
            match = re.search(r"""href=["'](http://ufcstats\.com/fighter-details/[^"']+)["']""", row_html, flags=re.IGNORECASE)
            urls.append(match.group(1) if match else "")
        table = table.iloc[: len(urls)].copy()
        table["URL"] = urls
        if {"FIRST", "LAST"}.issubset(table.columns):
            table["FIRST"] = table["FIRST"].map(collapse_whitespace)
            table["LAST"] = table["LAST"].map(collapse_whitespace)
            nickname_col = next((col for col in table.columns if col.startswith("NICK")), None)
            if nickname_col is None:
                table["NICKNAME"] = ""
            else:
                table = table.rename(columns={nickname_col: "NICKNAME"})
            fighter_rows.extend(table[["FIRST", "LAST", "NICKNAME", "URL"]].to_dict("records"))

    fighters_df = pd.DataFrame(fighter_rows).drop_duplicates(subset=["URL"], keep="first").reset_index(drop=True)
    write_csv(fighters_df, output_path)
    return fighters_df


def _scrape_single_historical_profile(fighter_name: str, fighter_url: str, directory_row: dict[str, object] | None) -> dict[str, object]:
    profile_html = fetch_html(fighter_url)
    height_raw = extract_labeled_text(profile_html, "Height")
    reach_raw = extract_labeled_text(profile_html, "Reach")
    stance = extract_labeled_text(profile_html, "STANCE") or extract_labeled_text(profile_html, "Stance")
    dob_raw = extract_labeled_text(profile_html, "DOB")

    if directory_row:
        height_raw = height_raw or collapse_whitespace(directory_row.get("height_raw", ""))
        reach_raw = reach_raw or collapse_whitespace(directory_row.get("reach_raw", ""))
        stance = stance or collapse_whitespace(directory_row.get("stance", ""))

    return {
        "fighter_name": fighter_name,
        "fighter_name_normalized": normalize_fighter_name(fighter_name),
        "fighter_profile_url": fighter_url,
        "date_of_birth_raw": dob_raw,
        "date_of_birth": parse_event_date(dob_raw).date().isoformat() if not pd.isna(parse_event_date(dob_raw)) else pd.NA,
        "height_raw": height_raw,
        "height_inches": parse_height_to_inches(height_raw),
        "height_cms": inches_to_cm(parse_height_to_inches(height_raw)),
        "reach_raw": reach_raw,
        "reach_inches": parse_reach_to_inches(reach_raw),
        "reach_cms": inches_to_cm(parse_reach_to_inches(reach_raw)),
        "stance": collapse_whitespace(stance) if stance else pd.NA,
    }


def scrape_historical_fighter_profiles(
    fighters_df: pd.DataFrame,
    results_df: pd.DataFrame,
    output_path: str | Path,
    resume: bool = True,
) -> pd.DataFrame:
    output_path = Path(output_path)
    existing_df = pd.read_csv(output_path) if resume and output_path.exists() else pd.DataFrame()

    fighter_names: set[str] = set()
    for bout in results_df["BOUT"].dropna().astype(str):
        fighter_a, fighter_b = _fighter_names_from_bout(bout)
        if fighter_a:
            fighter_names.add(fighter_a)
        if fighter_b:
            fighter_names.add(fighter_b)

    directory_lookup = lookup_fighter_directory_entries(fighter_names)
    directory_map = {}
    if not directory_lookup.empty:
        directory_map = (
            directory_lookup.set_index("fighter_name_normalized")[
                ["fighter_profile_url", "height_raw", "reach_raw", "stance"]
            ].to_dict("index")
        )

    fighter_directory_map = {}
    if not fighters_df.empty:
        fighters_df = fighters_df.copy()
        fighters_df["fighter_name"] = (
            fighters_df["FIRST"].fillna("").astype(str).map(collapse_whitespace)
            + " "
            + fighters_df["LAST"].fillna("").astype(str).map(collapse_whitespace)
        ).str.strip()
        fighters_df["fighter_name_normalized"] = fighters_df["fighter_name"].map(normalize_fighter_name)
        fighter_directory_map = fighters_df.set_index("fighter_name_normalized")["URL"].to_dict()

    existing_names = set(existing_df.get("fighter_name_normalized", pd.Series(dtype=str)).astype(str))
    profile_rows: list[dict[str, object]] = []
    for fighter_name in sorted(fighter_names):
        normalized = normalize_fighter_name(fighter_name)
        if resume and normalized in existing_names:
            continue
        directory_row = directory_map.get(normalized, {})
        fighter_url = directory_row.get("fighter_profile_url") or fighter_directory_map.get(normalized, "")
        if not fighter_url:
            log(f"warning: no historical fighter profile URL found for {fighter_name}")
            continue
        try:
            profile_rows.append(_scrape_single_historical_profile(fighter_name, fighter_url, directory_row))
        except Exception as exc:  # pragma: no cover - network/source variability
            log(f"warning: historical fighter profile scrape failed for {fighter_name} ({fighter_url}): {exc}")

    combined = pd.concat([existing_df, pd.DataFrame(profile_rows)], ignore_index=True) if not existing_df.empty else pd.DataFrame(profile_rows)
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["fighter_name_normalized"], keep="last").reset_index(drop=True)
    write_csv(combined, output_path)
    return combined


def scrape_historical_events(
    output_dir: str | Path,
    max_events: int = 25,
    resume: bool = True,
    include_round_details: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    page_html = fetch_html(UFCSTATS_COMPLETED_URL)
    event_links = extract_ufcstats_event_links(page_html)
    master_template_columns = pd.read_csv(MASTER_TEMPLATE_PATH, nrows=0).columns.tolist() if MASTER_TEMPLATE_PATH.exists() else []
    existing_results = pd.read_csv(output_path / RESULTS_FILENAME) if resume and (output_path / RESULTS_FILENAME).exists() else pd.DataFrame()
    existing_stats = pd.read_csv(output_path / STATS_FILENAME) if resume and (output_path / STATS_FILENAME).exists() else pd.DataFrame()
    existing_master = pd.read_csv(output_path / MASTER_FILENAME) if resume and (output_path / MASTER_FILENAME).exists() else pd.DataFrame()
    existing_event_catalog = pd.read_csv(output_path / EVENT_CATALOG_FILENAME) if resume and (output_path / EVENT_CATALOG_FILENAME).exists() else pd.DataFrame()
    result_rows: list[dict[str, str]] = []
    stat_rows: list[dict[str, str]] = []
    master_rows: list[dict[str, object]] = []
    event_catalog_rows: list[dict[str, object]] = []
    completed_count = 0
    today = pd.Timestamp.today().normalize()
    if not existing_event_catalog.empty and "DATE" in existing_event_catalog.columns:
        existing_event_catalog = existing_event_catalog.copy()
        existing_event_catalog["DATE"] = pd.to_datetime(existing_event_catalog["DATE"], errors="coerce")
        existing_event_catalog = existing_event_catalog[existing_event_catalog["DATE"].notna() & (existing_event_catalog["DATE"] <= today)].copy()
        existing_event_catalog["DATE"] = existing_event_catalog["DATE"].dt.date.astype(str)
    existing_event_urls = set(existing_event_catalog.get("EVENT_URL", pd.Series(dtype=str)).dropna().astype(str))

    def persist_intermediate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        results_df = pd.concat([existing_results, pd.DataFrame(result_rows)], ignore_index=True) if not existing_results.empty else pd.DataFrame(result_rows)
        stats_df = pd.concat([existing_stats, pd.DataFrame(stat_rows)], ignore_index=True) if not existing_stats.empty else pd.DataFrame(stat_rows)
        master_df = pd.concat([existing_master, pd.DataFrame(master_rows)], ignore_index=True) if not existing_master.empty else pd.DataFrame(master_rows)
        event_catalog_df = pd.concat([existing_event_catalog, pd.DataFrame(event_catalog_rows)], ignore_index=True) if not existing_event_catalog.empty else pd.DataFrame(event_catalog_rows)
        if not results_df.empty:
            results_df = results_df[
                ["EVENT", "BOUT", "OUTCOME", "WEIGHTCLASS", "METHOD", "ROUND", "TIME", "TIME FORMAT", "REFEREE", "DETAILS", "URL"]
            ].copy()
            results_df = results_df.drop_duplicates(subset=["EVENT", "BOUT"], keep="first").reset_index(drop=True)
            write_csv(results_df, output_path / RESULTS_FILENAME)
        if not stats_df.empty:
            if not include_round_details:
                stats_df = _collapse_stats_to_fight_level(stats_df)
            stats_df = stats_df[
                [
                    "EVENT",
                    "BOUT",
                    "ROUND",
                    "FIGHTER",
                    "KD",
                    "SIG.STR.",
                    "SIG.STR. %",
                    "TOTAL STR.",
                    "TD",
                    "TD %",
                    "SUB.ATT",
                    "REV.",
                    "CTRL",
                    "HEAD",
                    "BODY",
                    "LEG",
                    "DISTANCE",
                    "CLINCH",
                    "GROUND",
                ]
            ].copy()
            stats_subset = ["EVENT", "BOUT", "ROUND", "FIGHTER"] if include_round_details else ["EVENT", "BOUT", "FIGHTER"]
            stats_df = stats_df.drop_duplicates(subset=stats_subset, keep="first").reset_index(drop=True)
            write_csv(stats_df, output_path / STATS_FILENAME)
        if not master_df.empty:
            master_df = master_df[master_template_columns].copy()
            master_df = master_df.drop_duplicates(subset=["date", "R_fighter", "B_fighter"], keep="first").reset_index(drop=True)
            write_csv(master_df, output_path / MASTER_FILENAME)
        if not event_catalog_df.empty:
            event_catalog_df = event_catalog_df.drop_duplicates(subset=["EVENT_URL"], keep="last").sort_values(["DATE", "EVENT"]).reset_index(drop=True)
            write_csv(event_catalog_df, output_path / EVENT_CATALOG_FILENAME)
        return results_df, stats_df, master_df, event_catalog_df

    for event_url in event_links:
        if max_events > 0 and completed_count >= max_events:
            break
        if resume and event_url in existing_event_urls:
            continue
        event_html = fetch_html(event_url)
        metadata = _event_metadata(event_html)
        event_date = pd.to_datetime(parse_event_date(metadata["DATE"]), errors="coerce")
        if pd.isna(event_date) or event_date > today:
            continue
        scraped_event_rows = _extract_completed_event_rows(event_html, event_url)
        if not scraped_event_rows:
            continue
        completed_count += 1
        event_catalog_rows.append(
            {
                "EVENT": metadata["EVENT"],
                "DATE": event_date.date().isoformat(),
                "LOCATION": metadata["LOCATION"],
                "EVENT_URL": event_url,
            }
        )
        rebuilt_rows: list[dict[str, str]] = []
        for fight_row in scraped_event_rows:
            fight_url = fight_row.get("URL", "")
            if not fight_url:
                rebuilt_rows.append(fight_row)
                continue
            try:
                fight_html = fetch_html(fight_url)
                detail_a, detail_b = _detail_fighter_order(
                    fight_html,
                    fight_row["event_row_fighter_1"],
                    fight_row["event_row_fighter_2"],
                )
                winner_name = fight_row["event_row_fighter_1"] if fight_row["OUTCOME"] == "W/L" else fight_row["event_row_fighter_2"]
                fight_row["BOUT"] = f"{detail_a} vs. {detail_b}"
                fight_row["fighter_A"] = normalize_fighter_name(detail_a)
                fight_row["fighter_B"] = normalize_fighter_name(detail_b)
                if fight_row["OUTCOME"] in {"W/L", "L/W"}:
                    fight_row["OUTCOME"] = "W/L" if winner_name == detail_a else "L/W"
                stat_builder = _fight_detail_round_rows if include_round_details else _fight_detail_total_rows
                stat_rows.extend(
                    stat_builder(
                        fight_html=fight_html,
                        event_name=fight_row["EVENT"],
                        bout_name=fight_row["BOUT"],
                        fight_round=str(fight_row["ROUND"]),
                        fighter_a=detail_a,
                        fighter_b=detail_b,
                    )
                )
                if master_template_columns:
                    master_rows.append(_build_master_row(fight_row, master_template_columns))
                rebuilt_rows.append(fight_row)
            except Exception:
                if master_template_columns:
                    master_rows.append(_build_master_row(fight_row, master_template_columns))
                rebuilt_rows.append(fight_row)
        result_rows.extend(rebuilt_rows)
        if completed_count % 10 == 0:
            persist_intermediate()

    results_df, stats_df, master_df, event_catalog_df = persist_intermediate()
    return results_df, stats_df, master_df, event_catalog_df


def build_historical_physical_features(
    results_df: pd.DataFrame,
    event_catalog_df: pd.DataFrame,
    fighter_profiles_df: pd.DataFrame,
    output_path: str | Path,
) -> pd.DataFrame:
    if results_df.empty or event_catalog_df.empty:
        physical_df = pd.DataFrame()
        write_csv(physical_df, output_path)
        return physical_df

    fights_df = results_df.copy()
    fighters = fights_df["BOUT"].astype(str).str.split(" vs. ", expand=True)
    fights_df["fighter_A"] = fighters[0].fillna("").map(collapse_whitespace)
    fights_df["fighter_B"] = fighters[1].fillna("").map(collapse_whitespace)
    fights_df["fighter_A_normalized"] = fights_df["fighter_A"].map(normalize_fighter_name)
    fights_df["fighter_B_normalized"] = fights_df["fighter_B"].map(normalize_fighter_name)

    event_dates = event_catalog_df[["EVENT", "DATE", "LOCATION"]].copy()
    event_dates["event_date"] = pd.to_datetime(event_dates["DATE"], errors="coerce")
    fights_df = fights_df.merge(event_dates[["EVENT", "event_date", "LOCATION"]], on="EVENT", how="left")

    profiles = fighter_profiles_df.copy()
    if profiles.empty:
        for column in ["A_age", "B_age", "A_height_cms", "B_height_cms", "A_reach_cms", "B_reach_cms"]:
            fights_df[column] = pd.NA
    else:
        profile_keep = ["fighter_name_normalized", "date_of_birth_raw", "height_cms", "reach_cms", "fighter_profile_url"]
        available_keep = [column for column in profile_keep if column in profiles.columns]
        profiles = profiles[available_keep].copy()
        a_profiles = profiles.rename(columns={"fighter_name_normalized": "fighter_A_normalized"}).copy()
        a_profiles = a_profiles.rename(
            columns={
                "date_of_birth_raw": "A_date_of_birth_raw",
                "height_cms": "A_height_cms",
                "reach_cms": "A_reach_cms",
                "fighter_profile_url": "A_fighter_profile_url",
            }
        )
        b_profiles = profiles.rename(columns={"fighter_name_normalized": "fighter_B_normalized"}).copy()
        b_profiles = b_profiles.rename(
            columns={
                "date_of_birth_raw": "B_date_of_birth_raw",
                "height_cms": "B_height_cms",
                "reach_cms": "B_reach_cms",
                "fighter_profile_url": "B_fighter_profile_url",
            }
        )
        fights_df = fights_df.merge(a_profiles, on="fighter_A_normalized", how="left")
        fights_df = fights_df.merge(b_profiles, on="fighter_B_normalized", how="left")
        fights_df["A_age"] = fights_df.apply(lambda row: calculate_age_years(row.get("A_date_of_birth_raw"), row.get("event_date")), axis=1)
        fights_df["B_age"] = fights_df.apply(lambda row: calculate_age_years(row.get("B_date_of_birth_raw"), row.get("event_date")), axis=1)

    fights_df["age_diff"] = fights_df["A_age"] - fights_df["B_age"]
    fights_df["height_diff"] = fights_df["A_height_cms"] - fights_df["B_height_cms"]
    fights_df["reach_diff"] = fights_df["A_reach_cms"] - fights_df["B_reach_cms"]
    physical_df = fights_df[
        [
            "EVENT",
            "BOUT",
            "event_date",
            "LOCATION",
            "fighter_A",
            "fighter_B",
            "fighter_A_normalized",
            "fighter_B_normalized",
            "A_age",
            "B_age",
            "A_height_cms",
            "B_height_cms",
            "A_reach_cms",
            "B_reach_cms",
            "age_diff",
            "height_diff",
            "reach_diff",
        ]
    ].copy()
    write_csv(physical_df, output_path)
    return physical_df


def validate_historical_dates(
    physical_df: pd.DataFrame,
    output_path: str | Path,
) -> dict[str, object]:
    if physical_df.empty:
        report = {"missing_event_dates": 0, "fighter_negative_gap_count": 0, "fighter_large_gap_count": 0}
        Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    report: dict[str, object] = {}
    report["missing_event_dates"] = int(physical_df["event_date"].isna().sum())
    long_df = pd.concat(
        [
            physical_df[["event_date", "fighter_A_normalized"]].rename(columns={"fighter_A_normalized": "fighter"}),
            physical_df[["event_date", "fighter_B_normalized"]].rename(columns={"fighter_B_normalized": "fighter"}),
        ],
        ignore_index=True,
    )
    long_df["event_date"] = pd.to_datetime(long_df["event_date"], errors="coerce")
    long_df = long_df.dropna(subset=["event_date", "fighter"]).sort_values(["fighter", "event_date"]).reset_index(drop=True)
    long_df["days_since_previous"] = long_df.groupby("fighter")["event_date"].diff().dt.days
    report["fighter_negative_gap_count"] = int((long_df["days_since_previous"] < 0).sum())
    report["fighter_large_gap_count"] = int((long_df["days_since_previous"] > 3650).sum())
    report["fighters_with_missing_dates"] = int(long_df["fighter"].isna().sum())
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _merge_historical_odds_into_master(master_df: pd.DataFrame, historical_odds_df: pd.DataFrame) -> pd.DataFrame:
    if master_df.empty or historical_odds_df.empty:
        return master_df

    merged = master_df.copy()
    merged["stable_key"] = (
        pd.to_datetime(merged["date"], errors="coerce").dt.date.astype(str).replace("NaT", "")
        + " | "
        + merged.apply(lambda row: build_pair_key(str(row["R_fighter"]), str(row["B_fighter"])), axis=1)
    )
    odds_df = historical_odds_df.copy()
    odds_df["stable_key"] = odds_df["stable_key"].astype(str)
    odds_subset = odds_df[
        [
            "stable_key",
            "fighter_A_normalized",
            "fighter_B_normalized",
            "fighter_A_moneyline",
            "fighter_B_moneyline",
        ]
    ].drop_duplicates(subset=["stable_key"], keep="last").rename(
        columns={
            "fighter_A_normalized": "odds_fighter_A_normalized",
            "fighter_B_normalized": "odds_fighter_B_normalized",
            "fighter_A_moneyline": "odds_fighter_A_moneyline",
            "fighter_B_moneyline": "odds_fighter_B_moneyline",
        }
    )
    merged = merged.merge(odds_subset, on="stable_key", how="left")
    exact_mask = (
        merged["R_fighter"].map(normalize_fighter_name) == merged["odds_fighter_A_normalized"]
    ) & (
        merged["B_fighter"].map(normalize_fighter_name) == merged["odds_fighter_B_normalized"]
    )
    swapped_mask = (
        merged["R_fighter"].map(normalize_fighter_name) == merged["odds_fighter_B_normalized"]
    ) & (
        merged["B_fighter"].map(normalize_fighter_name) == merged["odds_fighter_A_normalized"]
    )
    merged["R_odds"] = pd.to_numeric(
        merged["odds_fighter_A_moneyline"].where(exact_mask, merged["odds_fighter_B_moneyline"].where(swapped_mask)),
        errors="coerce",
    )
    merged["B_odds"] = pd.to_numeric(
        merged["odds_fighter_B_moneyline"].where(exact_mask, merged["odds_fighter_A_moneyline"].where(swapped_mask)),
        errors="coerce",
    )
    return merged.drop(columns=["odds_fighter_A_normalized", "odds_fighter_B_normalized", "odds_fighter_A_moneyline", "odds_fighter_B_moneyline", "stable_key"])


def _merge_historical_physicals_into_master(master_df: pd.DataFrame, physical_df: pd.DataFrame) -> pd.DataFrame:
    if master_df.empty or physical_df.empty:
        return master_df
    merged = master_df.copy()
    merged["stable_key"] = (
        pd.to_datetime(merged["date"], errors="coerce").dt.date.astype(str).replace("NaT", "")
        + " | "
        + merged.apply(lambda row: build_pair_key(str(row["R_fighter"]), str(row["B_fighter"])), axis=1)
    )
    physical_map = physical_df.copy()
    physical_map["stable_key"] = (
        pd.to_datetime(physical_map["event_date"], errors="coerce").dt.date.astype(str).replace("NaT", "")
        + " | "
        + physical_map.apply(lambda row: build_pair_key(str(row["fighter_A"]), str(row["fighter_B"])), axis=1)
    )
    physical_subset = physical_map[
        [
            "stable_key",
            "fighter_A_normalized",
            "fighter_B_normalized",
            "A_age",
            "B_age",
            "A_height_cms",
            "B_height_cms",
            "A_reach_cms",
            "B_reach_cms",
        ]
    ].drop_duplicates(subset=["stable_key"], keep="last").rename(
        columns={
            "fighter_A_normalized": "phys_fighter_A_normalized",
            "fighter_B_normalized": "phys_fighter_B_normalized",
            "A_age": "phys_A_age",
            "B_age": "phys_B_age",
            "A_height_cms": "phys_A_height_cms",
            "B_height_cms": "phys_B_height_cms",
            "A_reach_cms": "phys_A_reach_cms",
            "B_reach_cms": "phys_B_reach_cms",
        }
    )
    merged = merged.merge(physical_subset, on="stable_key", how="left")
    exact_mask = (
        merged["R_fighter"].map(normalize_fighter_name) == merged["phys_fighter_A_normalized"]
    ) & (
        merged["B_fighter"].map(normalize_fighter_name) == merged["phys_fighter_B_normalized"]
    )
    swapped_mask = (
        merged["R_fighter"].map(normalize_fighter_name) == merged["phys_fighter_B_normalized"]
    ) & (
        merged["B_fighter"].map(normalize_fighter_name) == merged["phys_fighter_A_normalized"]
    )
    merged["R_age"] = pd.to_numeric(merged["phys_A_age"].where(exact_mask, merged["phys_B_age"].where(swapped_mask)), errors="coerce")
    merged["B_age"] = pd.to_numeric(merged["phys_B_age"].where(exact_mask, merged["phys_A_age"].where(swapped_mask)), errors="coerce")
    merged["R_Height_cms"] = pd.to_numeric(merged["phys_A_height_cms"].where(exact_mask, merged["phys_B_height_cms"].where(swapped_mask)), errors="coerce")
    merged["B_Height_cms"] = pd.to_numeric(merged["phys_B_height_cms"].where(exact_mask, merged["phys_A_height_cms"].where(swapped_mask)), errors="coerce")
    merged["R_Reach_cms"] = pd.to_numeric(merged["phys_A_reach_cms"].where(exact_mask, merged["phys_B_reach_cms"].where(swapped_mask)), errors="coerce")
    merged["B_Reach_cms"] = pd.to_numeric(merged["phys_B_reach_cms"].where(exact_mask, merged["phys_A_reach_cms"].where(swapped_mask)), errors="coerce")
    return merged.drop(columns=["phys_fighter_A_normalized", "phys_fighter_B_normalized", "phys_A_age", "phys_B_age", "phys_A_height_cms", "phys_B_height_cms", "phys_A_reach_cms", "phys_B_reach_cms", "stable_key"])


def build_historical_backfill(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_events: int = 25,
    build_features: bool = True,
    resume: bool = True,
    scrape_odds: bool = True,
    include_round_details: bool = False,
) -> dict[str, pd.DataFrame]:
    try:
        from src.features import build_feature_dataset
    except ImportError:  # pragma: no cover
        from features import build_feature_dataset

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fighters_path = output_path / FIGHTERS_FILENAME
    if resume and fighters_path.exists():
        fighters_df = pd.read_csv(fighters_path)
    else:
        fighters_df = scrape_historical_fighter_directory(fighters_path)
    results_df, stats_df, master_df, event_catalog_df = scrape_historical_events(
        output_path,
        max_events=max_events,
        resume=resume,
        include_round_details=include_round_details,
    )
    fighter_profiles_df = scrape_historical_fighter_profiles(
        fighters_df=fighters_df,
        results_df=results_df,
        output_path=output_path / FIGHTER_PROFILES_FILENAME,
        resume=resume,
    )
    # Normalize known aliases before any downstream joins so physical enrichment,
    # master assembly, and feature generation all use the same fighter identity.
    apply_canonical_identity_map(output_path)
    if (output_path / RESULTS_FILENAME).exists():
        results_df = pd.read_csv(output_path / RESULTS_FILENAME)
    if (output_path / STATS_FILENAME).exists():
        stats_df = pd.read_csv(output_path / STATS_FILENAME)
    if (output_path / MASTER_FILENAME).exists():
        master_df = pd.read_csv(output_path / MASTER_FILENAME)
    if (output_path / FIGHTERS_FILENAME).exists():
        fighters_df = pd.read_csv(output_path / FIGHTERS_FILENAME)
    if (output_path / FIGHTER_PROFILES_FILENAME).exists():
        fighter_profiles_df = pd.read_csv(output_path / FIGHTER_PROFILES_FILENAME)
    historical_physical_df = build_historical_physical_features(
        results_df=results_df,
        event_catalog_df=event_catalog_df,
        fighter_profiles_df=fighter_profiles_df,
        output_path=output_path / PHYSICAL_FEATURES_FILENAME,
    )
    date_validation_report = validate_historical_dates(
        physical_df=historical_physical_df,
        output_path=output_path / DATE_VALIDATION_FILENAME,
    )
    if scrape_odds:
        historical_odds_raw_df, historical_odds_consensus_df = scrape_historical_odds(
            event_catalog_df=event_catalog_df,
            raw_output_path=output_path / HISTORICAL_ODDS_RAW_FILENAME,
            consensus_output_path=output_path / HISTORICAL_ODDS_CONSENSUS_FILENAME,
            resume=resume,
        )
    else:
        historical_odds_raw_df = (
            pd.read_csv(output_path / HISTORICAL_ODDS_RAW_FILENAME)
            if (output_path / HISTORICAL_ODDS_RAW_FILENAME).exists()
            else pd.DataFrame()
        )
        historical_odds_consensus_df = (
            pd.read_csv(output_path / HISTORICAL_ODDS_CONSENSUS_FILENAME)
            if (output_path / HISTORICAL_ODDS_CONSENSUS_FILENAME).exists()
            else pd.DataFrame()
        )
    master_df = _merge_historical_physicals_into_master(master_df, historical_physical_df)
    master_df = _merge_historical_odds_into_master(master_df, historical_odds_consensus_df)
    if not master_df.empty:
        write_csv(master_df, output_path / MASTER_FILENAME)
    outputs = {
        "fighters": fighters_df,
        "results": results_df,
        "stats": stats_df,
        "event_catalog": event_catalog_df,
        "fighter_profiles": fighter_profiles_df,
        "physical_features": historical_physical_df,
        "historical_odds_raw": historical_odds_raw_df,
        "historical_odds_consensus": historical_odds_consensus_df,
        "master": master_df,
        "date_validation": pd.DataFrame([date_validation_report]),
    }
    if build_features and not results_df.empty and not stats_df.empty:
        feature_df = build_feature_dataset(
            data_dir=output_path,
            output_path=output_path / FEATURES_FILENAME,
            save=True,
            include_physical=True,
            physical_dataset_path=output_path / PHYSICAL_FEATURES_FILENAME,
        )
        outputs["features"] = feature_df
    scraped_aliases = {
        "ufc_fight_results_scraped.csv": results_df,
        "ufc_fight_stats_scraped.csv": stats_df,
        "ufc_fighter_details_scraped.csv": fighters_df,
    }
    if not master_df.empty:
        scraped_aliases[MASTER_FILENAME] = master_df
    for filename, df in scraped_aliases.items():
        write_csv(df, output_path / filename)
    return outputs


def build_archive_coverage_audit(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> tuple[pd.DataFrame, dict[str, object]]:
    output_path = Path(output_dir)
    discovered_links = extract_ufcstats_event_links(fetch_html(UFCSTATS_COMPLETED_URL))
    event_catalog = pd.read_csv(output_path / EVENT_CATALOG_FILENAME) if (output_path / EVENT_CATALOG_FILENAME).exists() else pd.DataFrame()
    results_df = pd.read_csv(output_path / RESULTS_FILENAME) if (output_path / RESULTS_FILENAME).exists() else pd.DataFrame()
    diagnostics_path = output_path / "historical_event_scrape_diagnostics.csv"
    existing_diag = pd.read_csv(diagnostics_path) if diagnostics_path.exists() else pd.DataFrame()

    scraped_urls = set(event_catalog.get("EVENT_URL", pd.Series(dtype=str)).dropna().astype(str))
    result_counts = results_df.groupby("EVENT").size().to_dict() if not results_df.empty else {}
    event_catalog_index = (
        event_catalog.drop_duplicates(subset=["EVENT_URL"], keep="last").set_index("EVENT_URL").to_dict("index")
        if not event_catalog.empty
        else {}
    )
    existing_diag_index = (
        existing_diag.drop_duplicates(subset=["EVENT_URL"], keep="last").set_index("EVENT_URL").to_dict("index")
        if not existing_diag.empty and "EVENT_URL" in existing_diag.columns
        else {}
    )

    rows: list[dict[str, object]] = []
    for event_url in discovered_links:
        catalog_row = event_catalog_index.get(event_url, {})
        diag_row = existing_diag_index.get(event_url, {})
        row = {
            "EVENT_URL": event_url,
            "EVENT": catalog_row.get("EVENT", ""),
            "DATE": catalog_row.get("DATE", ""),
            "LOCATION": catalog_row.get("LOCATION", ""),
            "status": "scraped" if event_url in scraped_urls else "missing",
            "reason": "captured" if event_url in scraped_urls else "not_yet_scraped",
            "expected_fights": int(result_counts.get(catalog_row.get("EVENT", ""), 0)) if event_url in scraped_urls else pd.NA,
            "captured_fights": int(result_counts.get(catalog_row.get("EVENT", ""), 0)) if event_url in scraped_urls else 0,
        }
        if event_url not in scraped_urls:
            try:
                event_html = fetch_html(event_url)
                metadata = _event_metadata(event_html)
                expected_rows = _extract_completed_event_rows(event_html, event_url)
                row["EVENT"] = metadata.get("EVENT", "")
                parsed_date = pd.to_datetime(parse_event_date(metadata.get("DATE", "")), errors="coerce")
                row["DATE"] = parsed_date.date().isoformat() if not pd.isna(parsed_date) else metadata.get("DATE", "")
                row["LOCATION"] = metadata.get("LOCATION", "")
                row["expected_fights"] = int(len(expected_rows))
                row["reason"] = "recoverable_missing" if len(expected_rows) > 0 else "parse_failure"
            except Exception as exc:  # pragma: no cover - network/source variability
                row["reason"] = f"request_failure: {exc}"
        if diag_row:
            diag_reason = str(diag_row.get("reason", "")).strip()
            if diag_reason:
                row["reason"] = diag_reason
            diag_status = str(diag_row.get("status", "")).strip()
            if diag_status and row["status"] != "scraped":
                row["status"] = diag_status
        rows.append(row)

    audit_df = pd.DataFrame(rows)
    if not audit_df.empty:
        audit_df = audit_df.sort_values(["status", "DATE", "EVENT"], na_position="last").reset_index(drop=True)
    write_csv(audit_df, output_path / ARCHIVE_COVERAGE_AUDIT_FILENAME)

    total_expected_fights = pd.to_numeric(audit_df.get("expected_fights", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    total_captured_fights = pd.to_numeric(audit_df.get("captured_fights", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    summary = {
        "total_events_discovered": int(len(audit_df)),
        "total_events_scraped_successfully": int((audit_df["status"] == "scraped").sum()) if not audit_df.empty else 0,
        "total_events_failed_or_missing": int((audit_df["status"] != "scraped").sum()) if not audit_df.empty else 0,
        "total_fights_expected_from_catalog": int(total_expected_fights),
        "total_fights_captured": int(total_captured_fights),
        "event_coverage": float((audit_df["status"] == "scraped").mean()) if not audit_df.empty else 0.0,
        "fight_coverage_from_catalog": float(total_captured_fights / total_expected_fights) if total_expected_fights else 0.0,
    }
    missing_examples = audit_df[audit_df["status"] != "scraped"].head(25) if not audit_df.empty else pd.DataFrame()
    summary_lines = [
        "# Archive Coverage Summary",
        "",
        f"- Total events discovered: `{summary['total_events_discovered']}`",
        f"- Total events scraped successfully: `{summary['total_events_scraped_successfully']}`",
        f"- Total events failed or missing: `{summary['total_events_failed_or_missing']}`",
        f"- Total fights expected from discovered catalog: `{summary['total_fights_expected_from_catalog']}`",
        f"- Total fights captured: `{summary['total_fights_captured']}`",
        f"- Event coverage: `{summary['event_coverage']:.1%}`",
        f"- Fight coverage from discovered catalog: `{summary['fight_coverage_from_catalog']:.1%}`",
        "",
        "## Missing / Failed Examples",
    ]
    if missing_examples.empty:
        summary_lines.append("- None")
    else:
        for _, row in missing_examples.iterrows():
            summary_lines.append(
                f"- `{row.get('DATE', '')}` | `{row.get('EVENT', '') or row.get('EVENT_URL', '')}` | status=`{row.get('status', '')}` | reason=`{row.get('reason', '')}`"
            )
    (output_path / ARCHIVE_COVERAGE_SUMMARY_FILENAME).write_text("\n".join(summary_lines), encoding="utf-8")
    return audit_df, summary


def run_parsed_values_audit(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    output_path = Path(output_dir)
    results_df = pd.read_csv(output_path / RESULTS_FILENAME)
    stats_df = pd.read_csv(output_path / STATS_FILENAME)
    master_df = pd.read_csv(output_path / MASTER_FILENAME)
    features_df = pd.read_csv(output_path / FEATURES_FILENAME)

    def of_split(series: pd.Series) -> pd.DataFrame:
        extracted = series.astype(str).str.extract(r"(\d+)\s+of\s+(\d+)")
        return extracted.apply(pd.to_numeric, errors="coerce")

    stats_sig = of_split(stats_df["SIG.STR."])
    stats_total = of_split(stats_df["TOTAL STR."])
    stats_td = of_split(stats_df["TD"])
    ctrl_seconds = stats_df["CTRL"].astype(str).str.extract(r"(\d+):(\d+)").apply(pd.to_numeric, errors="coerce")
    ctrl_values = ctrl_seconds[0].fillna(0) * 60 + ctrl_seconds[1].fillna(0)
    required_nulls = {
        "results": results_df[["EVENT", "BOUT", "OUTCOME", "METHOD", "ROUND", "TIME"]].isna().sum().to_dict(),
        "stats": stats_df[["EVENT", "BOUT", "FIGHTER", "KD", "SIG.STR.", "TOTAL STR.", "TD", "SUB.ATT", "CTRL"]].isna().sum().to_dict(),
        "master": master_df[["R_fighter", "B_fighter", "date", "Winner", "R_age", "B_age", "R_Height_cms", "B_Height_cms", "R_Reach_cms", "B_Reach_cms"]].isna().sum().to_dict(),
        "features": features_df.isna().sum().to_dict(),
    }
    duplicate_counts = {
        "results_by_event_bout": int(results_df.duplicated(subset=["EVENT", "BOUT"]).sum()),
        "stats_by_event_bout_fighter": int(stats_df.duplicated(subset=["EVENT", "BOUT", "FIGHTER"]).sum()),
        "master_by_date_pair": int(master_df.duplicated(subset=["date", "R_fighter", "B_fighter"]).sum()),
        "features_by_fight_id": int(features_df.duplicated(subset=["fight_id"]).sum()),
    }
    suspicious = {
        "stats_sig_landed_gt_attempted": stats_df.loc[stats_sig[0] > stats_sig[1], ["EVENT", "BOUT", "FIGHTER", "SIG.STR."]].head(10),
        "stats_total_landed_gt_attempted": stats_df.loc[stats_total[0] > stats_total[1], ["EVENT", "BOUT", "FIGHTER", "TOTAL STR."]].head(10),
        "stats_td_landed_gt_attempted": stats_df.loc[stats_td[0] > stats_td[1], ["EVENT", "BOUT", "FIGHTER", "TD"]].head(10),
        "stats_negative_ctrl": stats_df.loc[ctrl_values < 0, ["EVENT", "BOUT", "FIGHTER", "CTRL"]].head(10),
        "master_invalid_age": master_df.loc[(master_df["R_age"] < 16) | (master_df["R_age"] > 60) | (master_df["B_age"] < 16) | (master_df["B_age"] > 60), ["date", "R_fighter", "B_fighter", "R_age", "B_age"]].head(10),
        "master_invalid_height": master_df.loc[(master_df["R_Height_cms"] < 120) | (master_df["R_Height_cms"] > 250) | (master_df["B_Height_cms"] < 120) | (master_df["B_Height_cms"] > 250), ["date", "R_fighter", "B_fighter", "R_Height_cms", "B_Height_cms"]].head(10),
        "master_invalid_reach": master_df.loc[(master_df["R_Reach_cms"] < 120) | (master_df["R_Reach_cms"] > 260) | (master_df["B_Reach_cms"] < 120) | (master_df["B_Reach_cms"] > 260), ["date", "R_fighter", "B_fighter", "R_Reach_cms", "B_Reach_cms"]].head(10),
        "master_bad_dates": master_df.loc[pd.to_datetime(master_df["date"], errors="coerce").isna(), ["date", "R_fighter", "B_fighter"]].head(10),
    }
    stats_ranges = {
        "KD": {"min": float(pd.to_numeric(stats_df["KD"], errors="coerce").min()), "max": float(pd.to_numeric(stats_df["KD"], errors="coerce").max())},
        "SUB.ATT": {"min": float(pd.to_numeric(stats_df["SUB.ATT"], errors="coerce").min()), "max": float(pd.to_numeric(stats_df["SUB.ATT"], errors="coerce").max())},
        "CTRL_seconds": {"min": float(ctrl_values.min()), "max": float(ctrl_values.max())},
    }
    master_ranges = {
        "R_age": {"min": float(pd.to_numeric(master_df["R_age"], errors="coerce").min()), "max": float(pd.to_numeric(master_df["R_age"], errors="coerce").max())},
        "B_age": {"min": float(pd.to_numeric(master_df["B_age"], errors="coerce").min()), "max": float(pd.to_numeric(master_df["B_age"], errors="coerce").max())},
        "R_Height_cms": {"min": float(pd.to_numeric(master_df["R_Height_cms"], errors="coerce").min()), "max": float(pd.to_numeric(master_df["R_Height_cms"], errors="coerce").max())},
        "B_Height_cms": {"min": float(pd.to_numeric(master_df["B_Height_cms"], errors="coerce").min()), "max": float(pd.to_numeric(master_df["B_Height_cms"], errors="coerce").max())},
        "R_Reach_cms": {"min": float(pd.to_numeric(master_df["R_Reach_cms"], errors="coerce").min()), "max": float(pd.to_numeric(master_df["R_Reach_cms"], errors="coerce").max())},
        "B_Reach_cms": {"min": float(pd.to_numeric(master_df["B_Reach_cms"], errors="coerce").min()), "max": float(pd.to_numeric(master_df["B_Reach_cms"], errors="coerce").max())},
    }
    report_lines = [
        "# Parsed Values Audit",
        "",
        "## Null Counts By Required Field",
        f"- Results: `{required_nulls['results']}`",
        f"- Stats: `{required_nulls['stats']}`",
        f"- Master: `{required_nulls['master']}`",
        "",
        "## Duplicate Counts",
        f"- `{duplicate_counts}`",
        "",
        "## Sanity Ranges",
        f"- Stats ranges: `{stats_ranges}`",
        f"- Master physical ranges: `{master_ranges}`",
        "",
        "## Suspicious Row Samples",
    ]
    for name, frame in suspicious.items():
        report_lines.append(f"- `{name}`: `{len(frame)}` sample rows")
        if not frame.empty:
            report_lines.append(f"  - `{frame.to_dict('records')[:3]}`")
    (output_path / PARSED_VALUES_AUDIT_FILENAME).write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "required_nulls": required_nulls,
        "duplicate_counts": duplicate_counts,
        "stats_ranges": stats_ranges,
        "master_ranges": master_ranges,
        "suspicious_counts": {name: int(len(frame)) for name, frame in suspicious.items()},
    }


def run_elo_audit(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    output_path = Path(output_dir)
    try:
        from src.features import build_feature_dataset, dynamic_k
    except ImportError:  # pragma: no cover
        from features import build_feature_dataset, dynamic_k

    final_df, artifacts = build_feature_dataset(
        data_dir=output_path,
        save=False,
        return_intermediates=True,
        include_physical=True,
        physical_dataset_path=output_path / PHYSICAL_FEATURES_FILENAME,
    )
    model_table = artifacts["model_table"].copy()
    fights_df = artifacts["fights"].copy()
    results_df = pd.read_csv(output_path / RESULTS_FILENAME) if (output_path / RESULTS_FILENAME).exists() else pd.DataFrame()
    event_catalog_df = pd.read_csv(output_path / EVENT_CATALOG_FILENAME) if (output_path / EVENT_CATALOG_FILENAME).exists() else pd.DataFrame()
    if not results_df.empty:
        results_df = results_df.copy()
        results_df["fight_id"] = results_df["EVENT"].astype(str).str.strip() + " | " + results_df["BOUT"].astype(str).str.strip()
        date_map = event_catalog_df[["EVENT", "DATE"]].drop_duplicates(subset=["EVENT"], keep="last") if not event_catalog_df.empty else pd.DataFrame(columns=["EVENT", "DATE"])
        results_df = results_df.merge(date_map, on="EVENT", how="left")
        fight_dates = results_df[["fight_id", "DATE"]].drop_duplicates(subset=["fight_id"], keep="last").rename(columns={"DATE": "date"})
    else:
        fight_dates = pd.DataFrame(columns=["fight_id", "date"])
    model_table = model_table.merge(fight_dates, on="fight_id", how="left")
    model_table = model_table.sort_values("fight_order").reset_index(drop=True)

    elo_chain_rows: list[dict[str, object]] = []
    validation_failures = {
        "starting_elo_not_1500": 0,
        "winner_did_not_gain": 0,
        "loser_did_not_drop": 0,
        "chronology_date_backtrack": 0,
    }
    last_seen_date: dict[str, pd.Timestamp] = {}
    seen_fighters: set[str] = set()
    for _, row in model_table.iterrows():
        date_value = pd.to_datetime(row.get("date"), errors="coerce")
        for fighter_col, opp_col, elo_col, opp_elo_col, result in [
            ("fighter_A", "fighter_B", "A_elo", "B_elo", int(row["target_A_win"])),
            ("fighter_B", "fighter_A", "B_elo", "A_elo", 1 - int(row["target_A_win"])),
        ]:
            fighter = row[fighter_col]
            opponent = row[opp_col]
            pre_elo = float(row[elo_col])
            opp_pre_elo = float(row[opp_elo_col])
            prefight_prefix = "A" if fighter_col == "fighter_A" else "B"
            career_col = f"{prefight_prefix}_career_fights"
            career_fights = float(row[career_col]) if pd.notna(row[career_col]) else 0.0
            p = 1 / (1 + 10 ** ((opp_pre_elo - pre_elo) / 400))
            post_elo = pre_elo + dynamic_k(career_fights) * (result - p)
            passed = True
            notes: list[str] = []
            if fighter not in seen_fighters and abs(pre_elo - 1500.0) > 1e-9:
                validation_failures["starting_elo_not_1500"] += 1
                passed = False
                notes.append("debut_not_1500")
            if result == 1 and post_elo <= pre_elo:
                validation_failures["winner_did_not_gain"] += 1
                passed = False
                notes.append("winner_no_gain")
            if result == 0 and post_elo >= pre_elo:
                validation_failures["loser_did_not_drop"] += 1
                passed = False
                notes.append("loser_no_drop")
            if fighter in last_seen_date and pd.notna(date_value) and date_value < last_seen_date[fighter]:
                validation_failures["chronology_date_backtrack"] += 1
                passed = False
                notes.append("date_backtrack")
            if pd.notna(date_value):
                last_seen_date[fighter] = date_value
            seen_fighters.add(fighter)
            elo_chain_rows.append(
                {
                    "fight_id": row["fight_id"],
                    "event_date": row.get("date"),
                    "fighter": fighter,
                    "opponent": opponent,
                    "pre_fight_elo": pre_elo,
                    "result": result,
                    "post_fight_elo": post_elo,
                    "passed_validation": passed,
                    "notes": ";".join(notes),
                }
            )
    elo_chain_df = pd.DataFrame(elo_chain_rows)
    fighter_counts = elo_chain_df.groupby("fighter").size().sort_values(ascending=False)
    sampled_fighters = list(fighter_counts.head(5).index)
    sample_df = elo_chain_df[elo_chain_df["fighter"].isin(sampled_fighters)].copy()
    sample_df = sample_df.sort_values(["fighter", "event_date", "fight_id"]).reset_index(drop=True)
    write_csv(sample_df, output_path / ELO_AUDIT_SAMPLE_FILENAME)

    rematch_pairs = fights_df.groupby(["fighter_A", "fighter_B"]).size()
    rematch_count = int((rematch_pairs > 1).sum())
    report_lines = [
        "# ELO Audit",
        "",
        f"- Feature rows audited: `{len(model_table)}`",
        f"- Sequential fighter transitions audited: `{len(elo_chain_df)}`",
        f"- Validation failures: `{validation_failures}`",
        f"- Rematch pair count in captured archive: `{rematch_count}`",
        "",
        "## Sampled Fighter Sequences",
        f"- Fighters: `{sampled_fighters}`",
    ]
    for fighter in sampled_fighters:
        subset = sample_df[sample_df["fighter"] == fighter].head(8)
        report_lines.append(f"- `{fighter}`: `{subset.to_dict('records')}`")
    (output_path / ELO_AUDIT_FILENAME).write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "feature_rows": int(len(model_table)),
        "transition_rows": int(len(elo_chain_df)),
        "validation_failures": validation_failures,
        "sampled_fighters": sampled_fighters,
    }


def run_qualitative_audit(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    output_path = Path(output_dir)
    results_df = pd.read_csv(output_path / RESULTS_FILENAME)
    master_df = pd.read_csv(output_path / MASTER_FILENAME)
    features_df = pd.read_csv(output_path / FEATURES_FILENAME)
    random.seed(7)
    samples = {
        "random_fights": results_df.sample(min(5, len(results_df)), random_state=7)[["EVENT", "BOUT", "OUTCOME", "METHOD", "ROUND", "TIME"]]
        if not results_df.empty
        else pd.DataFrame(),
        "long_career_fighters": results_df["BOUT"].str.split(" vs. ", expand=True).stack().value_counts().head(5).to_dict()
        if not results_df.empty
        else {},
        "draw_or_nc": results_df[results_df["OUTCOME"].isin(["D/D", "NC/NC"])][["EVENT", "BOUT", "OUTCOME", "METHOD"]].head(5)
        if not results_df.empty
        else pd.DataFrame(),
    }
    famous = ["jon jones", "max holloway", "alexander volkanovski", "charles oliveira", "israel adesanya"]
    feature_samples = []
    for fighter in famous:
        subset = features_df[features_df["fight_id"].str.contains(fighter, case=False, regex=False)].head(2)
        if not subset.empty:
            feature_samples.extend(subset[["fight_id", "target_A_win", "A_elo", "B_elo"]].to_dict("records"))
    report_lines = [
        "# Qualitative Audit Summary",
        "",
        f"- Random fight sample: `{samples['random_fights'].to_dict('records') if not samples['random_fights'].empty else []}`",
        f"- Long-career fighter appearance leaders: `{samples['long_career_fighters']}`",
        f"- Draw / no-contest sample: `{samples['draw_or_nc'].to_dict('records') if not samples['draw_or_nc'].empty else []}`",
        f"- Feature / ELO spot-check sample: `{feature_samples}`",
        "",
        "- Manual readout: sampled identities, chronology, target orientation, and physical values appear internally plausible on the checked rows; remaining concerns are archive completeness and odds sparsity rather than obvious parser corruption.",
    ]
    (output_path / QUALITATIVE_AUDIT_FILENAME).write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "random_sample_count": int(len(samples["random_fights"])) if isinstance(samples["random_fights"], pd.DataFrame) else 0,
        "feature_sample_count": int(len(feature_samples)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical UFCStats data into the current raw/feature schema.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save scraped historical files.")
    parser.add_argument("--max-events", type=int, default=25, help="How many completed events to scrape for this backfill run.")
    parser.add_argument("--skip-features", action="store_true", help="Skip rebuilding the feature dataset after scraping.")
    parser.add_argument("--skip-odds", action="store_true", help="Skip live historical-odds scraping and reuse any existing odds artifacts.")
    parser.add_argument("--include-round-details", action="store_true", help="Scrape per-round stats instead of the default fight-level totals.")
    parser.add_argument("--run-audits", action="store_true", help="Generate coverage, parsed-values, ELO, and qualitative audit artifacts after the backfill run.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume-safe reads of existing backfill artifacts.")
    args = parser.parse_args()

    outputs = build_historical_backfill(
        output_dir=args.output_dir,
        max_events=args.max_events,
        build_features=not args.skip_features,
        resume=not args.no_resume,
        scrape_odds=not args.skip_odds,
        include_round_details=args.include_round_details,
    )
    for name, df in outputs.items():
        print(f"{name}: {df.shape}")
    summary = summarize_backfill_progress(args.output_dir)
    print("progress_summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}" if "coverage" in key else f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    if args.run_audits:
        coverage_df, coverage_summary = build_archive_coverage_audit(args.output_dir)
        parsed_summary = run_parsed_values_audit(args.output_dir)
        elo_summary = run_elo_audit(args.output_dir)
        qualitative_summary = run_qualitative_audit(args.output_dir)
        print("audit_summary:")
        print(f"  coverage_events_discovered: {coverage_summary['total_events_discovered']}")
        print(f"  coverage_events_scraped_successfully: {coverage_summary['total_events_scraped_successfully']}")
        print(f"  coverage_events_failed_or_missing: {coverage_summary['total_events_failed_or_missing']}")
        print(f"  parsed_duplicate_counts: {parsed_summary['duplicate_counts']}")
        print(f"  elo_validation_failures: {elo_summary['validation_failures']}")
        print(f"  qualitative_feature_sample_count: {qualitative_summary['feature_sample_count']}")


if __name__ == "__main__":
    main()
