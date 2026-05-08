from __future__ import annotations

"""Shared helpers for live UFC scraping and live-input derivation."""

from datetime import date, datetime, timezone
from functools import lru_cache
from html.parser import HTMLParser
from html import unescape
from http.client import IncompleteRead
from io import StringIO
from pathlib import Path
import time
import re
from statistics import median
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

try:
    from requests.exceptions import ChunkedEncodingError as RequestsChunkedEncodingError
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from requests.exceptions import Timeout as RequestsTimeout
except ImportError:  # pragma: no cover
    RequestsChunkedEncodingError = ()
    RequestsConnectionError = ()
    RequestsTimeout = ()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_TIMEOUT = 30
DEFAULT_FETCH_RETRIES = 3
DEFAULT_FETCH_BACKOFF_SECONDS = 1.0
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
FIGHTER_DIRECTORY_URL_TEMPLATE = "http://ufcstats.com/statistics/fighters?char={char}&page=all"

CANONICAL_FIGHTER_ALIASES = {
    "bibulatov magomed": "magomed bibulatov",
    "patricio freire": "patricio pitbull",
    "kai kamaka": "kai kamaka iii",
    "rafael cerquiera": "rafael cerqueira",
}

FORBIDDEN_MARKET_TERMS = {
    "over",
    "under",
    "decision",
    "round",
    "fight",
    "draw",
    "result",
    "tko",
    "ko",
    "submission",
    "inside",
    "distance",
    "starts",
    "won't",
    "ends",
    "either",
    "other",
    "significant",
    "strikes",
    "takedowns",
    "scorecards",
    "handicap",
    "points",
    "awarded",
    "fotn",
    "potn",
}


def log(message: str) -> None:
    """Print a scraper-friendly log line."""
    print(f"[live-data] {message}")


def _is_retryable_fetch_exception(error: Exception) -> bool:
    """Return whether a fetch exception is likely transient and safe to retry."""
    if isinstance(error, IncompleteRead):
        return True
    if RequestsChunkedEncodingError and isinstance(error, RequestsChunkedEncodingError):
        return True
    if RequestsConnectionError and isinstance(error, RequestsConnectionError):
        return True
    if RequestsTimeout and isinstance(error, RequestsTimeout):
        return True
    if isinstance(error, TimeoutError):
        return True
    if isinstance(error, URLError):
        return True
    if isinstance(error, HTTPError):
        return 500 <= int(error.code) < 600
    return False


def fetch_html(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_FETCH_RETRIES,
    backoff_seconds: float = DEFAULT_FETCH_BACKOFF_SECONDS,
) -> str:
    """Fetch HTML with a browser-style user agent and transient-network retries."""
    request = Request(url, headers={"User-Agent": USER_AGENT})
    attempts = max(1, int(retries))
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as error:
            last_error = error
            if not _is_retryable_fetch_exception(error) or attempt == attempts:
                raise
            sleep_seconds = backoff_seconds * (2 ** (attempt - 1))
            log(
                f"fetch attempt {attempt}/{attempts} failed for {url} with "
                f"{type(error).__name__}: {error}. Retrying in {sleep_seconds:.1f}s."
            )
            time.sleep(sleep_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to fetch HTML for {url}")


def read_html_tables(html: str) -> list[pd.DataFrame]:
    """Parse HTML tables into pandas DataFrames."""
    try:
        return pd.read_html(StringIO(html))
    except (ImportError, ValueError):
        return _read_html_tables_fallback(html)


class _TableParser(HTMLParser):
    """Minimal HTML table parser for scraper fallback use."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._current_table: list[list[str]] | None = None
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            self._current_table = []
        elif tag == "tr" and self._current_table is not None:
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            self._current_cell = []
            self._in_cell = True
        elif tag == "br" and self._in_cell and self._current_cell is not None:
            self._current_cell.append(" ")

    def handle_data(self, data: str) -> None:
        if self._in_cell and self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._current_row is not None and self._current_cell is not None:
            self._current_row.append(collapse_whitespace(unescape("".join(self._current_cell))))
            self._current_cell = None
            self._in_cell = False
        elif tag == "tr" and self._current_table is not None and self._current_row is not None:
            if any(cell != "" for cell in self._current_row):
                self._current_table.append(self._current_row)
            self._current_row = None
        elif tag == "table" and self._current_table is not None:
            if self._current_table:
                self.tables.append(self._current_table)
            self._current_table = None


def _read_html_tables_fallback(html: str) -> list[pd.DataFrame]:
    """Fallback parser when pandas cannot use lxml/html5lib."""
    parser = _TableParser()
    parser.feed(html)
    frames: list[pd.DataFrame] = []
    for raw_table in parser.tables:
        if not raw_table:
            continue
        header = raw_table[0]
        body = raw_table[1:] if len(raw_table) > 1 else []
        max_len = max(len(row) for row in raw_table)
        header = header + [f"unnamed_{idx}" for idx in range(len(header), max_len)]
        normalized_body = [row + [""] * (max_len - len(row)) for row in body]
        if normalized_body:
            frames.append(pd.DataFrame(normalized_body, columns=header[:max_len]))
        else:
            frames.append(pd.DataFrame(columns=header[:max_len]))
    return frames


def collapse_whitespace(value: object) -> str:
    """Collapse repeated whitespace into single spaces."""
    return " ".join(str(value).split())


def strip_html_tags(html: str) -> str:
    """Remove HTML tags and unescape entities."""
    return collapse_whitespace(unescape(re.sub(r"<[^>]+>", " ", html)))


def normalize_fighter_name(name: str) -> str:
    """Normalize fighter names to match the modeling pipeline convention."""
    normalized = collapse_whitespace(name).strip().lower()
    normalized = re.sub(r"\b(jr\.?|sr\.?|ii|iv)\b", "", normalized).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return CANONICAL_FIGHTER_ALIASES.get(normalized, normalized)


def build_pair_key(fighter_a: str, fighter_b: str) -> str:
    """Build an unordered normalized fighter-pair key."""
    normalized = sorted([normalize_fighter_name(fighter_a), normalize_fighter_name(fighter_b)])
    return " || ".join(normalized)


def write_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist a dataframe to disk."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    return destination


def normalize_column_name(col: object) -> str:
    """Flatten table column labels into lowercase strings."""
    if isinstance(col, tuple):
        parts = [str(part).strip() for part in col if str(part).strip() and not str(part).startswith("Unnamed")]
        return " ".join(parts).strip().lower()
    return str(col).strip().lower()


def safe_float(value: object) -> float:
    """Convert values to float, returning NaN on failure."""
    if pd.isna(value):
        return np.nan
    text = collapse_whitespace(value).replace(",", "")
    if text == "":
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def safe_int(value: object) -> float:
    """Convert values to integer-like float, returning NaN on failure."""
    value_float = safe_float(value)
    return float(int(value_float)) if not pd.isna(value_float) else np.nan


def parse_of_stat(value: object) -> tuple[float, float]:
    """Parse stats stored as 'landed of attempted'."""
    if pd.isna(value):
        return np.nan, np.nan
    match = re.search(r"(\d+)\s+of\s+(\d+)", str(value))
    if not match:
        return np.nan, np.nan
    return float(match.group(1)), float(match.group(2))


def parse_clock_to_seconds(value: object) -> float:
    """Parse MM:SS or H:MM:SS clock strings into seconds."""
    if pd.isna(value):
        return np.nan
    text = collapse_whitespace(value)
    if not text:
        return np.nan
    parts = text.split(":")
    try:
        if len(parts) == 2:
            minutes, seconds = parts
            return float(int(minutes) * 60 + int(seconds))
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return float(int(hours) * 3600 + int(minutes) * 60 + int(seconds))
    except ValueError:
        return np.nan
    return np.nan


def parse_event_date(value: object) -> pd.Timestamp:
    """Parse event dates from common UFC source formats."""
    if pd.isna(value):
        return pd.NaT
    text = collapse_whitespace(value)
    if not text:
        return pd.NaT
    cleaned = re.sub(r"(\d{1,2})(st|nd|rd|th)", r"\1", text, flags=re.IGNORECASE)
    return pd.to_datetime(cleaned, errors="coerce")


def parse_american_odds(value: object) -> float:
    """Parse American odds from text values."""
    if pd.isna(value):
        return np.nan
    text = collapse_whitespace(value).replace("âˆ’", "-")
    if not text:
        return np.nan
    if text.upper() == "EVEN":
        return 100.0
    match = re.search(r"([+-]?\d{2,4})", text)
    if not match:
        return np.nan
    try:
        return float(match.group(1))
    except ValueError:
        return np.nan


def american_to_implied_probability(odds: object) -> float:
    """Convert American odds into implied probability."""
    odds_value = parse_american_odds(odds)
    if pd.isna(odds_value) or odds_value == 0:
        return np.nan
    if odds_value > 0:
        return float(100.0 / (odds_value + 100.0))
    return float(abs(odds_value) / (abs(odds_value) + 100.0))


def consensus_american_odds(values: Iterable[object]) -> float:
    """Compute a simple consensus line from multiple sportsbook columns."""
    odds = [parse_american_odds(value) for value in values]
    odds = [odd for odd in odds if not pd.isna(odd)]
    if not odds:
        return np.nan
    return float(median(odds))


def inches_to_cm(value: object) -> float:
    """Convert inches to centimeters."""
    inches = safe_float(value)
    if pd.isna(inches):
        return np.nan
    return float(inches * 2.54)


def parse_height_to_inches(value: object) -> float:
    """Parse fighter height strings into inches."""
    if pd.isna(value):
        return np.nan
    text = collapse_whitespace(value).lower()
    if not text:
        return np.nan
    feet_match = re.search(r"(\d+)\s*'\s*(\d+)", text)
    if feet_match:
        return float(int(feet_match.group(1)) * 12 + int(feet_match.group(2)))
    inch_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:in|inches|\")", text)
    if inch_match:
        return float(inch_match.group(1))
    cm_match = re.search(r"(\d+(?:\.\d+)?)\s*cm", text)
    if cm_match:
        return float(cm_match.group(1)) / 2.54
    return safe_float(text)


def parse_reach_to_inches(value: object) -> float:
    """Parse fighter reach strings into inches."""
    return parse_height_to_inches(value)


def calculate_age_years(dob: object, as_of: pd.Timestamp | None = None) -> float:
    """Calculate age in years from a date of birth."""
    birth_date = parse_event_date(dob)
    if pd.isna(birth_date):
        return np.nan
    anchor = as_of if as_of is not None and not pd.isna(as_of) else pd.Timestamp(date.today())
    if pd.isna(anchor):
        return np.nan
    return float((anchor - birth_date).days / 365.25)


def as_absolute_url(base_url: str, href: str) -> str:
    """Resolve a relative URL against a base URL."""
    return urljoin(base_url, href)


def extract_ufcstats_event_links(html: str) -> list[str]:
    """Extract UFCStats event detail links from an events page."""
    links = re.findall(r"""href=["'](https?://ufcstats\.com/event-details/[^"']+)["']""", html, flags=re.IGNORECASE)
    deduped: list[str] = []
    seen: set[str] = set()
    for link in links:
        if link not in seen:
            deduped.append(link)
            seen.add(link)
    return deduped


def extract_row_html_blocks(html: str) -> list[str]:
    """Extract HTML table row blocks in document order."""
    return re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.IGNORECASE | re.DOTALL)


def looks_like_fighter_name(text: object) -> bool:
    """Heuristic filter to keep real fighter names and drop prop-market labels."""
    value = collapse_whitespace(text)
    if not value:
        return False
    lower_value = value.lower()
    if any(term in lower_value for term in FORBIDDEN_MARKET_TERMS):
        return False
    tokens = re.findall(r"[A-Za-z][A-Za-z'.-]*", value)
    return len(tokens) >= 2


def extract_labeled_text(html: str, label: str) -> str:
    """Extract simple 'Label: Value' fields from HTML."""
    escaped_label = re.escape(label)
    patterns = [
        rf"{escaped_label}\s*:?\s*</[^>]+>\s*([^<]+)",
        rf"{escaped_label}\s*:?\s*([^<\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.IGNORECASE)
        if match:
            return collapse_whitespace(unescape(match.group(1)))
    text = strip_html_tags(html)
    match = re.search(rf"{escaped_label}\s*:?\s*([A-Za-z0-9,'\".+/\- ]+)", text, flags=re.IGNORECASE)
    return collapse_whitespace(match.group(1)) if match else ""


def utc_now_iso() -> str:
    """Return a UTC timestamp string for odds snapshots."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@lru_cache(maxsize=64)
def _fetch_fighter_directory_page(initial_char: str) -> str:
    return fetch_html(FIGHTER_DIRECTORY_URL_TEMPLATE.format(char=initial_char.lower()))


def lookup_fighter_directory_entries(fighter_names: Iterable[str]) -> pd.DataFrame:
    """Resolve UFCStats fighter profile URLs and directory metadata by fighter name."""
    normalized_targets = sorted({normalize_fighter_name(name) for name in fighter_names if normalize_fighter_name(name)})
    if not normalized_targets:
        return pd.DataFrame(
            columns=[
                "fighter_name",
                "fighter_name_normalized",
                "fighter_profile_url",
                "height_raw",
                "reach_raw",
                "stance",
                "record_wins",
                "record_losses",
                "record_draws",
            ]
        )

    initials = sorted({name[0] for name in normalized_targets if name and name[0].isalpha()})
    rows: list[dict[str, object]] = []
    for initial in initials:
        html = _fetch_fighter_directory_page(initial)
        tables = read_html_tables(html)
        if not tables:
            continue
        table = tables[0].copy()
        raw_rows = [row for row in extract_row_html_blocks(html) if "fighter-details" in row.lower()]
        aligned_count = min(len(table), len(raw_rows))
        table = table.iloc[:aligned_count].copy().reset_index(drop=True)
        table.columns = [collapse_whitespace(str(col)) for col in table.columns]
        url_values: list[str] = []
        for row_html in raw_rows[:aligned_count]:
            match = re.search(r"""href=["'](http://ufcstats\.com/fighter-details/[^"']+)["']""", row_html, flags=re.IGNORECASE)
            url_values.append(match.group(1) if match else "")
        table["fighter_profile_url"] = url_values
        first_col = next((col for col in table.columns if col.lower() == "first"), None)
        last_col = next((col for col in table.columns if col.lower() == "last"), None)
        if first_col is None or last_col is None:
            continue
        table["fighter_name"] = (
            table[first_col].map(collapse_whitespace) + " " + table[last_col].map(collapse_whitespace)
        ).str.strip()
        table["fighter_name_normalized"] = table["fighter_name"].map(normalize_fighter_name)
        rename_map = {
            next((col for col in table.columns if col.lower() == "ht."), ""): "height_raw",
            next((col for col in table.columns if col.lower() == "reach"), ""): "reach_raw",
            next((col for col in table.columns if col.lower() == "stance"), ""): "stance",
            next((col for col in table.columns if col.lower() == "w"), ""): "record_wins",
            next((col for col in table.columns if col.lower() == "l"), ""): "record_losses",
            next((col for col in table.columns if col.lower() == "d"), ""): "record_draws",
        }
        rename_map = {key: value for key, value in rename_map.items() if key}
        table = table.rename(columns=rename_map)
        keep_cols = [
            "fighter_name",
            "fighter_name_normalized",
            "fighter_profile_url",
            "height_raw",
            "reach_raw",
            "stance",
            "record_wins",
            "record_losses",
            "record_draws",
        ]
        for col in keep_cols:
            if col not in table.columns:
                table[col] = np.nan if col.startswith("record_") else ""
        rows.extend(table[keep_cols].to_dict("records"))

    lookup_df = pd.DataFrame(rows)
    if lookup_df.empty:
        return lookup_df
    lookup_df = lookup_df.drop_duplicates(subset=["fighter_name_normalized"], keep="first").reset_index(drop=True)
    lookup_df = lookup_df[lookup_df["fighter_name_normalized"].isin(normalized_targets)].copy()
    return lookup_df
