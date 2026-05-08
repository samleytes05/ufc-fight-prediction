from __future__ import annotations

"""Scrape historical UFC odds from Best Fight Odds archive/search pages."""

import argparse
from pathlib import Path
import re
from urllib.parse import quote_plus

import pandas as pd

from .common import (
    DATA_DIR,
    as_absolute_url,
    collapse_whitespace,
    extract_row_html_blocks,
    fetch_html,
    log,
    parse_event_date,
    read_html_tables,
    write_csv,
)
from .fetch_current_odds import BEST_FIGHT_ODDS_HOME, _build_consensus_odds, _table_to_raw_rows


DEFAULT_RAW_OUTPUT_PATH = DATA_DIR / "historical_backfill" / "historical_odds_raw_scraped.csv"
DEFAULT_CONSENSUS_OUTPUT_PATH = DATA_DIR / "historical_backfill" / "historical_odds_consensus_scraped.csv"
DEFAULT_DIAGNOSTICS_PATH = DATA_DIR / "historical_backfill" / "historical_odds_match_diagnostics.csv"


def _candidate_queries(event_name: str) -> list[str]:
    event_name = collapse_whitespace(event_name)
    if not event_name:
        return []
    queries = [event_name]
    if ":" in event_name:
        head = collapse_whitespace(event_name.split(":", 1)[0])
        if head and head not in queries:
            queries.append(head)
    numbered_match = re.search(r"\bufc\s+\d+\b", event_name, flags=re.IGNORECASE)
    if numbered_match:
        numbered = collapse_whitespace(numbered_match.group(0))
        if numbered not in queries:
            queries.append(numbered)
    if "vs." in event_name.lower():
        matchup = collapse_whitespace(event_name.split(":", 1)[-1])
        if matchup and matchup not in queries:
            queries.append(matchup)
    return queries


def _search_event_candidates(event_query: str) -> list[dict[str, object]]:
    query = quote_plus(event_query)
    search_url = f"{BEST_FIGHT_ODDS_HOME}search?query={query}"
    html = fetch_html(search_url)
    candidates: list[dict[str, object]] = []

    row_blocks = extract_row_html_blocks(html)
    for row_html in row_blocks:
        if "/events/" not in row_html:
            continue
        link_match = re.search(r'href="(?P<href>/events/[^"]+)">(?P<label>[^<]+)<', row_html, flags=re.IGNORECASE)
        date_match = re.search(
            r'([A-Z][a-z]{2,8}\s+\d{1,2}(?:st|nd|rd|th)?\s+\d{4})',
            row_html,
            flags=re.IGNORECASE,
        )
        if not link_match:
            continue
        href = link_match.group("href")
        label = collapse_whitespace(link_match.group("label"))
        if "ufc" not in label.lower() and "ufc" not in href.lower():
            continue
        candidates.append(
            {
                "event_label": label,
                "event_url": as_absolute_url(BEST_FIGHT_ODDS_HOME, href),
                "event_date": collapse_whitespace(date_match.group(1)) if date_match else "",
                "search_url": search_url,
                "search_query": event_query,
            }
        )
    deduped = pd.DataFrame(candidates).drop_duplicates(subset=["event_url"]).to_dict("records") if candidates else []
    return deduped


def _score_candidate(event_name: str, event_date: object, candidate: dict[str, object]) -> float:
    name_tokens = {token for token in re.findall(r"[a-z0-9]+", event_name.lower()) if token not in {"ufc", "fight", "night", "vs"}}
    label_tokens = {token for token in re.findall(r"[a-z0-9]+", str(candidate["event_label"]).lower()) if token not in {"ufc", "fight", "night", "vs"}}
    token_score = len(name_tokens & label_tokens)
    target_date = pd.to_datetime(event_date, errors="coerce")
    candidate_date = pd.to_datetime(parse_event_date(candidate.get("event_date", "")), errors="coerce")
    date_score = 0.0
    if not pd.isna(target_date) and not pd.isna(candidate_date):
        day_gap = abs((target_date - candidate_date).days)
        date_score = max(0.0, 10.0 - min(day_gap, 10.0))
    return token_score + date_score


def _match_bestfightodds_event(event_name: str, event_date: object) -> dict[str, object] | None:
    candidates: list[dict[str, object]] = []
    for query in _candidate_queries(event_name):
        candidates.extend(_search_event_candidates(query))
    if not candidates:
        return None
    scored = sorted(candidates, key=lambda candidate: _score_candidate(event_name, event_date, candidate), reverse=True)
    best = scored[0]
    if _score_candidate(event_name, event_date, best) <= 0:
        return None
    return best


def scrape_historical_odds(
    event_catalog_df: pd.DataFrame,
    raw_output_path: str | Path = DEFAULT_RAW_OUTPUT_PATH,
    consensus_output_path: str | Path = DEFAULT_CONSENSUS_OUTPUT_PATH,
    diagnostics_output_path: str | Path = DEFAULT_DIAGNOSTICS_PATH,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_output_path = Path(raw_output_path)
    consensus_output_path = Path(consensus_output_path)
    diagnostics_output_path = Path(diagnostics_output_path)
    existing_raw = pd.read_csv(raw_output_path) if resume and raw_output_path.exists() else pd.DataFrame()
    existing_events = set(existing_raw.get("event_name", pd.Series(dtype=str)).astype(str) + " | " + existing_raw.get("event_date", pd.Series(dtype=str)).astype(str))
    existing_diagnostics = pd.read_csv(diagnostics_output_path) if resume and diagnostics_output_path.exists() else pd.DataFrame()
    if not existing_diagnostics.empty:
        existing_diagnostics = existing_diagnostics.drop_duplicates(subset=["event_name", "event_date"], keep="last")
        processed_events = set(existing_diagnostics["event_name"].astype(str) + " | " + existing_diagnostics["event_date"].astype(str))
    else:
        processed_events = set()

    raw_rows = existing_raw.to_dict("records") if not existing_raw.empty else []
    diagnostics_rows = existing_diagnostics.to_dict("records") if not existing_diagnostics.empty else []
    for _, event_row in event_catalog_df.sort_values("DATE").iterrows():
        event_name = collapse_whitespace(event_row.get("EVENT", ""))
        event_date = collapse_whitespace(event_row.get("DATE", ""))
        event_key = f"{event_name} | {event_date}"
        if resume and event_key in existing_events and event_key in processed_events:
            continue
        diagnostic = {
            "event_name": event_name,
            "event_date": event_date,
            "status": "",
            "reason": "",
            "search_query": "",
            "matched_event_label": "",
            "matched_event_url": "",
            "rows_scraped": 0,
        }
        try:
            match = _match_bestfightodds_event(event_name, event_date)
        except Exception as exc:  # pragma: no cover - network/source variability
            log(f"warning: historical odds event search failed for {event_name} ({event_date}): {exc}")
            diagnostic["status"] = "failed"
            diagnostic["reason"] = f"search_error: {exc}"
            diagnostics_rows.append(diagnostic)
            continue
        if not match:
            log(f"warning: no BestFightOdds event match found for {event_name} ({event_date})")
            diagnostic["status"] = "unmatched"
            diagnostic["reason"] = "no_page_found"
            diagnostics_rows.append(diagnostic)
            continue
        diagnostic["search_query"] = str(match.get("search_query", ""))
        diagnostic["matched_event_label"] = str(match.get("event_label", ""))
        diagnostic["matched_event_url"] = str(match.get("event_url", ""))
        try:
            event_html = fetch_html(match["event_url"])
            event_raw_rows: list[dict[str, object]] = []
            for table in read_html_tables(event_html):
                event_raw_rows.extend(_table_to_raw_rows(table, event_name, event_date, str(match["event_url"]), event_date))
            raw_rows.extend(event_raw_rows)
            diagnostic["status"] = "matched" if event_raw_rows else "matched_no_rows"
            diagnostic["reason"] = "ok" if event_raw_rows else "parsing_failure"
            diagnostic["rows_scraped"] = len(event_raw_rows)
        except Exception as exc:  # pragma: no cover - source variability
            log(f"warning: historical odds scrape failed for {event_name} ({event_date}): {exc}")
            diagnostic["status"] = "failed"
            diagnostic["reason"] = f"scrape_error: {exc}"
        diagnostics_rows.append(diagnostic)

    raw_df = pd.DataFrame(raw_rows)
    if not raw_df.empty:
        raw_df = raw_df.drop_duplicates(
            subset=["event_name", "event_date", "matchup_key", "sportsbook", "fighter_A_moneyline", "fighter_B_moneyline"],
            keep="last",
        ).reset_index(drop=True)
    consensus_df = _build_consensus_odds(raw_df)
    if not consensus_df.empty:
        consensus_df["stable_key"] = (
            pd.to_datetime(consensus_df["event_date"], errors="coerce").dt.date.astype(str).replace("NaT", "")
            + " | "
            + consensus_df["matchup_key"].astype(str)
        )
    diagnostics_df = pd.DataFrame(diagnostics_rows)
    if not diagnostics_df.empty:
        diagnostics_df = diagnostics_df.drop_duplicates(subset=["event_name", "event_date"], keep="last").reset_index(drop=True)
    write_csv(raw_df, raw_output_path)
    write_csv(consensus_df, consensus_output_path)
    write_csv(diagnostics_df, diagnostics_output_path)
    if not diagnostics_df.empty:
        matched = int((diagnostics_df["status"] == "matched").sum())
        unmatched = int((diagnostics_df["status"] == "unmatched").sum())
        failed = int((diagnostics_df["status"] == "failed").sum())
        matched_no_rows = int((diagnostics_df["status"] == "matched_no_rows").sum())
        log(
            "historical odds diagnostics: "
            f"matched={matched} unmatched={unmatched} failed={failed} matched_no_rows={matched_no_rows}"
        )
    log(f"historical odds rows scraped: raw={len(raw_df)} consensus={len(consensus_df)}")
    return raw_df, consensus_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape historical UFC odds from Best Fight Odds.")
    parser.add_argument(
        "--event-catalog",
        type=Path,
        default=DATA_DIR / "historical_backfill" / "historical_event_catalog_scraped.csv",
        help="Historical event catalog to match against Best Fight Odds archive/search results.",
    )
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT_PATH)
    parser.add_argument("--consensus-output", type=Path, default=DEFAULT_CONSENSUS_OUTPUT_PATH)
    args = parser.parse_args()
    event_catalog_df = pd.read_csv(args.event_catalog)
    raw_df, consensus_df = scrape_historical_odds(
        event_catalog_df=event_catalog_df,
        raw_output_path=args.raw_output,
        consensus_output_path=args.consensus_output,
    )
    print(f"Saved raw historical odds: {args.raw_output} ({len(raw_df)} rows)")
    print(f"Saved consensus historical odds: {args.consensus_output} ({len(consensus_df)} rows)")


if __name__ == "__main__":
    main()
