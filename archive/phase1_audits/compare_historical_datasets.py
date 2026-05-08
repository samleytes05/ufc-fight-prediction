from __future__ import annotations

"""Compare imported historical baselines against scraped historical backfill outputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.features import build_feature_dataset
except ImportError:  # pragma: no cover
    from features import build_feature_dataset

try:
    from src.scrapers.backfill_historical_dataset import _collapse_stats_to_fight_level
except ImportError:  # pragma: no cover
    from backfill_historical_dataset import _collapse_stats_to_fight_level


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "data"
DEFAULT_BACKFILL_DIR = PROJECT_ROOT / "data" / "historical_backfill"
DEFAULT_REPORT_PATH = DEFAULT_BACKFILL_DIR / "historical_comparison_report.md"
DEFAULT_JSON_PATH = DEFAULT_BACKFILL_DIR / "historical_comparison_report.json"

COMPLETION_THRESHOLDS = {
    "training_data_independence": {
        "results_overlap_min": 0.95,
        "stats_overlap_min": 0.95,
        "feature_overlap_min": 0.95,
        "feature_target_consistency_min": 0.999,
        "physical_coverage_min": 0.95,
        "missing_event_dates_max": 0,
        "negative_gap_count_max": 0,
    },
    "betting_data_independence": {
        "training_requirements": "Must already satisfy training-data independence thresholds.",
        "historical_odds_coverage_min": 0.95,
        "master_overlap_min": 0.95,
        "winner_consistency_min": 0.999,
    },
}


LEGACY_DATA_AUDIT = {
    "ufc_fight_results.csv": {
        "unique_essential": ["EVENT", "BOUT", "OUTCOME", "METHOD", "ROUND", "TIME"],
        "duplicated_but_useful": ["WEIGHTCLASS", "URL"],
        "duplicated_and_unnecessary": ["REFEREE", "DETAILS", "TIME FORMAT"],
        "unused_by_current_feature_pipeline": ["REFEREE", "DETAILS", "TIME FORMAT", "WEIGHTCLASS", "URL"],
        "contribution": "Defines fight identity, winner/target orientation, finish method, and fight-end timing used by the feature pipeline.",
    },
    "ufc_fight_stats.csv": {
        "unique_essential": ["EVENT", "BOUT", "ROUND", "FIGHTER", "KD", "SIG.STR.", "TOTAL STR.", "TD", "SUB.ATT", "CTRL"],
        "duplicated_but_useful": ["SIG.STR. %", "TD %"],
        "duplicated_and_unnecessary": ["REV.", "HEAD", "BODY", "LEG", "DISTANCE", "CLINCH", "GROUND"],
        "unused_by_current_feature_pipeline": ["REV.", "HEAD", "BODY", "LEG", "DISTANCE", "CLINCH", "GROUND", "SIG.STR. %", "TD %"],
        "contribution": "Provides the only raw striking, takedown, submission, knockdown, and control counts needed to build fighter histories and all rolling/career aggregates.",
    },
    "ufc_fighter_details.csv": {
        "unique_essential": [],
        "duplicated_but_useful": ["FIRST", "LAST", "URL"],
        "duplicated_and_unnecessary": ["NICKNAME"],
        "unused_by_current_feature_pipeline": ["FIRST", "LAST", "NICKNAME", "URL"],
        "contribution": "Does not feed the current training feature pipeline directly. Its main value is as a fighter-directory lookup source for scraper matching.",
    },
    "ufc-master.csv": {
        "unique_essential": ["R_odds", "B_odds", "R_age", "B_age", "R_Height_cms", "B_Height_cms", "R_Reach_cms", "B_Reach_cms", "date", "R_fighter", "B_fighter"],
        "duplicated_but_useful": ["Winner", "weight_class"],
        "duplicated_and_unnecessary": ["R_ev", "B_ev", "all engineered streak/rank/aggregate columns"],
        "unused_by_current_feature_pipeline": ["all precomputed streak, rank, and aggregate fighter columns", "R_ev", "B_ev", "finish", "finish_details", "finish_round", "finish_round_time", "total_fight_time_secs"],
        "contribution": "Only unique legacy source for historical odds and per-fight physical matchup inputs in the current baseline.",
    },
}


MINIMUM_SELF_OWNED_CONTRACT = {
    "identifiers": ["event_name", "event_date", "bout/fighter pairing", "fighter names plus stable fighter URLs when possible"],
    "results": ["winner/outcome orientation", "method", "fight-ending round", "fight-ending time"],
    "fighter_bio": ["fighter identity mapping", "date_of_birth or age history for age-at-fight", "height", "reach"],
    "stats": ["fighter-level fight stats at a grain sufficient to aggregate career and recent form", "round count or fight duration", "KD", "significant strikes landed/attempted", "total strikes landed/attempted", "takedowns landed/attempted", "submission attempts", "control time"],
    "time_features": ["event_date for strict chronological ordering", "event_date per fighter to compute days since last fight when needed"],
    "market_data": ["fighter-side odds only if EV evaluation or strategy simulation is required"],
    "not_required_due_to_redundancy": ["precomputed streak/rank/aggregate columns from ufc-master.csv", "SIG.STR.% and TD% when landed/attempted are present", "zone split columns (HEAD/BODY/LEG/DISTANCE/CLINCH/GROUND) for the current feature set", "fighter nickname fields", "referee/details/time-format fields for the current model dataset"],
}


RAW_FILE_CONFIG = {
    "ufc_fight_results": {
        "filename": "ufc_fight_results.csv",
        "key_columns": ["EVENT", "BOUT"],
        "discrepancy_columns": ["OUTCOME", "WEIGHTCLASS", "METHOD", "ROUND", "TIME"],
    },
    "ufc_fight_stats": {
        "filename": "ufc_fight_stats.csv",
        "key_columns": ["EVENT", "BOUT", "ROUND", "FIGHTER"],
        "discrepancy_columns": ["KD", "SIG.STR.", "TOTAL STR.", "TD", "SUB.ATT", "CTRL"],
    },
    "ufc_fighter_details": {
        "filename": "ufc_fighter_details.csv",
        "key_columns": ["URL"],
        "fallback_key_columns": ["FIRST", "LAST"],
        "discrepancy_columns": ["FIRST", "LAST", "NICKNAME"],
    },
}


def _safe_text(value: object) -> str:
    return " ".join(str(value).strip().lower().split()) if pd.notna(value) else ""


def _dtype_family(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "text"


def _duplicate_rate(df: pd.DataFrame, key_columns: list[str]) -> float:
    if df.empty:
        return 0.0
    return float(df.duplicated(subset=key_columns).mean())


def _normalize_results_key(df: pd.DataFrame) -> pd.Series:
    fighters = df["BOUT"].fillna("").astype(str).str.split(" vs. ", n=1, expand=True)
    fighter_1 = fighters[0].fillna("").map(_safe_text)
    fighter_2 = fighters[1].fillna("").map(_safe_text) if fighters.shape[1] > 1 else pd.Series("", index=df.index)
    matchup_key = pd.DataFrame({"fighter_1": fighter_1, "fighter_2": fighter_2}).apply(
        lambda row: " || ".join(sorted([row["fighter_1"], row["fighter_2"]])),
        axis=1,
    )
    return df["EVENT"].map(_safe_text) + " | " + matchup_key


def _normalize_stats_key(df: pd.DataFrame) -> pd.Series:
    fighters = df["BOUT"].fillna("").astype(str).str.split(" vs. ", n=1, expand=True)
    fighter_1 = fighters[0].fillna("").map(_safe_text)
    fighter_2 = fighters[1].fillna("").map(_safe_text) if fighters.shape[1] > 1 else pd.Series("", index=df.index)
    matchup_key = pd.DataFrame({"fighter_1": fighter_1, "fighter_2": fighter_2}).apply(
        lambda row: " || ".join(sorted([row["fighter_1"], row["fighter_2"]])),
        axis=1,
    )
    round_value = df["ROUND"].astype(str).str.extract(r"(\d+)")[0].fillna(df["ROUND"].astype(str)).map(_safe_text)
    return (
        df["EVENT"].map(_safe_text)
        + " | "
        + matchup_key
        + " | "
        + round_value
        + " | "
        + df["FIGHTER"].map(_safe_text)
    )


def _normalize_fighter_key(df: pd.DataFrame, fallback_columns: list[str] | None = None) -> pd.Series:
    if "URL" in df.columns and df["URL"].notna().any():
        return df["URL"].fillna("").astype(str).map(_safe_text)
    fallback_columns = fallback_columns or []
    pieces = [df[column].astype(str).map(_safe_text) for column in fallback_columns]
    key = pieces[0] if pieces else pd.Series("", index=df.index)
    for piece in pieces[1:]:
        key = key + " | " + piece
    return key


def _stable_fight_key(df: pd.DataFrame, red_col: str, blue_col: str, date_col: str | None = None) -> pd.Series:
    red = df[red_col].map(_safe_text)
    blue = df[blue_col].map(_safe_text)
    pair = pd.DataFrame({"red": red, "blue": blue}).apply(lambda row: " || ".join(sorted([row["red"], row["blue"]])), axis=1)
    if date_col and date_col in df.columns:
        date_values = pd.to_datetime(df[date_col], errors="coerce").dt.date.astype(str).replace("NaT", "")
        return date_values.map(_safe_text) + " | " + pair
    return pair


def _discrepancy_summary(old_df: pd.DataFrame, new_df: pd.DataFrame, key_column: str, compare_columns: list[str], top_n: int = 10) -> dict[str, object]:
    overlap = old_df.merge(new_df, on=key_column, how="inner", suffixes=("_old", "_new"))
    discrepancies: list[dict[str, object]] = []
    for column in compare_columns:
        old_col = f"{column}_old"
        new_col = f"{column}_new"
        if old_col not in overlap.columns or new_col not in overlap.columns:
            continue
        mismatch_mask = overlap[old_col].fillna("<na>").astype(str) != overlap[new_col].fillna("<na>").astype(str)
        mismatch_count = int(mismatch_mask.sum())
        if mismatch_count == 0:
            continue
        sample_rows = overlap.loc[mismatch_mask, [key_column, old_col, new_col]].head(3)
        discrepancies.append(
            {
                "column": column,
                "mismatch_count": mismatch_count,
                "sample": sample_rows.to_dict("records"),
            }
        )
    discrepancies.sort(key=lambda item: item["mismatch_count"], reverse=True)
    return {"overlap_rows": int(len(overlap)), "columns": discrepancies[:top_n]}


def _top_null_rates(df: pd.DataFrame, top_n: int = 10) -> dict[str, float]:
    if df.empty:
        return {}
    null_rates = df.isna().mean().sort_values(ascending=False).head(top_n)
    return {column: float(value) for column, value in null_rates.items()}


def _numeric_distribution_report(old_df: pd.DataFrame, new_df: pd.DataFrame, overlap_cols: list[str], top_n: int = 15) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for col in overlap_cols:
        if not pd.api.types.is_numeric_dtype(old_df[col]) or not pd.api.types.is_numeric_dtype(new_df[col]):
            continue
        rows.append(
            {
                "column": col,
                "old_mean": float(old_df[col].mean()),
                "new_mean": float(new_df[col].mean()),
                "old_std": float(old_df[col].std(ddof=0)),
                "new_std": float(new_df[col].std(ddof=0)),
                "mean_abs_diff": float(abs(old_df[col].mean() - new_df[col].mean())),
                "null_rate_old": float(old_df[col].isna().mean()),
                "null_rate_new": float(new_df[col].isna().mean()),
            }
        )
    rows.sort(key=lambda item: item["mean_abs_diff"], reverse=True)
    return rows[:top_n]


def _raw_file_report(name: str, baseline_dir: Path, backfill_dir: Path) -> dict[str, object]:
    config = RAW_FILE_CONFIG[name]
    old_df = pd.read_csv(baseline_dir / config["filename"])
    new_df = pd.read_csv(backfill_dir / config["filename"])

    if name == "ufc_fight_stats":
        old_df = _collapse_stats_to_fight_level(old_df)
        new_df = _collapse_stats_to_fight_level(new_df)

    if name == "ufc_fight_results":
        old_df["stable_key"] = _normalize_results_key(old_df)
        new_df["stable_key"] = _normalize_results_key(new_df)
    elif name == "ufc_fight_stats":
        old_df["stable_key"] = (
            old_df["EVENT"].map(_safe_text) + " | " + old_df["BOUT"].map(_safe_text) + " | " + old_df["FIGHTER"].map(_safe_text)
        )
        new_df["stable_key"] = (
            new_df["EVENT"].map(_safe_text) + " | " + new_df["BOUT"].map(_safe_text) + " | " + new_df["FIGHTER"].map(_safe_text)
        )
    else:
        old_df["stable_key"] = _normalize_fighter_key(old_df, config.get("fallback_key_columns"))
        new_df["stable_key"] = _normalize_fighter_key(new_df, config.get("fallback_key_columns"))

    old_columns = set(old_df.columns) - {"stable_key"}
    new_columns = set(new_df.columns) - {"stable_key"}
    shared_keys = set(old_df["stable_key"]) & set(new_df["stable_key"])
    discrepancies = _discrepancy_summary(old_df, new_df, "stable_key", config["discrepancy_columns"])
    audit = LEGACY_DATA_AUDIT[f"{name}.csv"]
    required_columns = audit["unique_essential"]
    required_columns_present = [column for column in required_columns if column in new_df.columns]
    required_columns_missing = [column for column in required_columns if column not in new_df.columns]
    material_blocker = ""
    if name == "ufc_fight_stats" and len(shared_keys) == 0 and not required_columns_missing:
        material_blocker = (
            "The required stat columns exist, but the scraped rows do not match the row grain/semantics of the legacy contract. "
            "This is the main blocker for a self-owned rebuild."
        )
    elif name == "ufc_fight_stats" and float(len(shared_keys) / len(old_df)) < 0.95:
        material_blocker = "The required stats contract now exists, but sampled historical coverage is still too low to retire the imported stats file."
    elif name == "ufc_fight_results" and float(len(shared_keys) / len(old_df)) < 0.95:
        material_blocker = "Coverage is still too low to replace the imported results contract safely."
    elif name == "ufc_fighter_details" and float(len(shared_keys) / len(old_df)) >= 0.95:
        material_blocker = "No material blocker for the current feature pipeline."

    return {
        "baseline_shape": list(old_df.drop(columns=["stable_key"]).shape),
        "scraped_shape": list(new_df.drop(columns=["stable_key"]).shape),
        "exact_column_match": old_columns == new_columns,
        "only_in_baseline": sorted(old_columns - new_columns),
        "only_in_scraped": sorted(new_columns - old_columns),
        "baseline_duplicate_rate": _duplicate_rate(old_df, ["stable_key"]),
        "scraped_duplicate_rate": _duplicate_rate(new_df, ["stable_key"]),
        "baseline_null_rates_top10": _top_null_rates(old_df.drop(columns=["stable_key"])),
        "scraped_null_rates_top10": _top_null_rates(new_df.drop(columns=["stable_key"])),
        "overlap_rows": len(shared_keys),
        "baseline_overlap_rate": float(len(shared_keys) / len(old_df)) if len(old_df) else 0.0,
        "scraped_overlap_rate": float(len(shared_keys) / len(new_df)) if len(new_df) else 0.0,
        "required_columns": required_columns,
        "required_columns_present": required_columns_present,
        "required_columns_missing": required_columns_missing,
        "non_material_columns": audit["unused_by_current_feature_pipeline"],
        "material_blocker": material_blocker,
        "discrepancies": discrepancies,
    }


def _build_master_level_report(baseline_dir: Path, backfill_dir: Path) -> dict[str, object]:
    old_df = pd.read_csv(baseline_dir / "ufc-master.csv").copy()
    new_df = pd.read_csv(backfill_dir / "ufc_master_scraped.csv").copy()
    old_df["stable_key"] = _stable_fight_key(old_df, "R_fighter", "B_fighter", "date")
    new_df["stable_key"] = _stable_fight_key(new_df, "R_fighter", "B_fighter", "date")

    shared_columns = sorted((set(old_df.columns) & set(new_df.columns)) - {"stable_key"})
    dtype_mismatches = []
    for column in shared_columns:
        old_family = _dtype_family(old_df[column])
        new_family = _dtype_family(new_df[column])
        if old_family != new_family:
            dtype_mismatches.append({"column": column, "baseline": old_family, "scraped": new_family})

    overlap = old_df.merge(new_df, on="stable_key", how="inner", suffixes=("_old", "_new"))
    if len(overlap):
        old_winner = overlap["Winner_old"].fillna("").astype(str).str.lower()
        new_winner = overlap["Winner_new"].fillna("").astype(str).str.lower()
        winner_consistency = float((old_winner == new_winner).mean())
    else:
        winner_consistency = np.nan
    physical_columns = ["R_age", "B_age", "R_Height_cms", "B_Height_cms", "R_Reach_cms", "B_Reach_cms"]
    odds_columns = ["R_odds", "B_odds"]
    physical_coverage = {
        column: float(1.0 - new_df[column].isna().mean()) for column in physical_columns if column in new_df.columns
    }
    odds_coverage = {
        column: float(1.0 - new_df[column].isna().mean()) for column in odds_columns if column in new_df.columns
    }

    return {
        "baseline_shape": list(old_df.drop(columns=["stable_key"]).shape),
        "scraped_shape": list(new_df.drop(columns=["stable_key"]).shape),
        "shared_columns": len(shared_columns),
        "only_in_baseline": sorted((set(old_df.columns) - set(new_df.columns)) - {"stable_key"}),
        "only_in_scraped": sorted((set(new_df.columns) - set(old_df.columns)) - {"stable_key"}),
        "dtype_mismatches": dtype_mismatches[:25],
        "baseline_null_rates_top15": _top_null_rates(old_df.drop(columns=["stable_key"]), top_n=15),
        "scraped_null_rates_top15": _top_null_rates(new_df.drop(columns=["stable_key"]), top_n=15),
        "overlap_fights": int(len(overlap)),
        "baseline_overlap_rate": float(len(overlap) / len(old_df)) if len(old_df) else 0.0,
        "scraped_overlap_rate": float(len(overlap) / len(new_df)) if len(new_df) else 0.0,
        "winner_consistency": winner_consistency,
        "physical_coverage": physical_coverage,
        "odds_coverage": odds_coverage,
        "distribution_differences_top15": _numeric_distribution_report(old_df, new_df, shared_columns),
    }


def _build_training_feature_report(baseline_dir: Path, backfill_dir: Path) -> dict[str, object]:
    backfill_physical_path = backfill_dir / "historical_physical_features_scraped.csv"
    old_feature_df = build_feature_dataset(data_dir=baseline_dir, save=False, include_physical=True)
    new_feature_df = build_feature_dataset(
        data_dir=backfill_dir,
        save=False,
        include_physical=backfill_physical_path.exists(),
        physical_dataset_path=backfill_physical_path if backfill_physical_path.exists() else None,
    )

    old_cols = set(old_feature_df.columns)
    new_cols = set(new_feature_df.columns)
    shared_cols = sorted(old_cols & new_cols)

    def add_match_keys(df: pd.DataFrame) -> pd.DataFrame:
        keyed = df.copy()
        fighters = keyed["fight_id"].astype(str).str.split(" | ", n=1, expand=True, regex=False)[1]
        bout_parts = fighters.str.split(" vs. ", expand=True)
        keyed["fighter_1"] = bout_parts[0].str.strip().str.lower()
        keyed["fighter_2"] = bout_parts[1].str.strip().str.lower()
        keyed["match_key"] = keyed.apply(lambda row: " || ".join(sorted([row["fighter_1"], row["fighter_2"]])), axis=1)
        keyed["event_key"] = keyed["fight_id"].astype(str).str.split(" | ", n=1, expand=True, regex=False)[0].str.strip()
        keyed["comparison_key"] = keyed["event_key"] + " | " + keyed["match_key"]
        return keyed

    old_keyed = add_match_keys(old_feature_df)
    new_keyed = add_match_keys(new_feature_df)
    overlap = old_keyed.merge(new_keyed, on="comparison_key", how="inner", suffixes=("_old", "_new"))
    if len(overlap):
        same_orientation = overlap["fighter_1_old"] == overlap["fighter_1_new"]
        adjusted_new_target = np.where(same_orientation, overlap["target_A_win_new"], 1 - overlap["target_A_win_new"])
        target_consistency = float((overlap["target_A_win_old"] == adjusted_new_target).mean())
    else:
        target_consistency = np.nan

    return {
        "baseline_shape": list(old_feature_df.shape),
        "scraped_shape": list(new_feature_df.shape),
        "shared_columns": len(shared_cols),
        "only_in_baseline": sorted(old_cols - new_cols),
        "only_in_scraped": sorted(new_cols - old_cols),
        "overlap_fights": int(len(overlap)),
        "target_consistency": target_consistency,
        "baseline_total_null_cells": int(old_feature_df.isna().sum().sum()),
        "scraped_total_null_cells": int(new_feature_df.isna().sum().sum()),
        "baseline_null_rates_top10": _top_null_rates(old_feature_df),
        "scraped_null_rates_top10": _top_null_rates(new_feature_df),
        "distribution_differences_top15": _numeric_distribution_report(old_feature_df, new_feature_df, shared_cols),
    }


def _retirement_decisions(
    raw_reports: dict[str, dict[str, object]],
    master_report: dict[str, object],
    feature_report: dict[str, object],
) -> dict[str, dict[str, str | bool]]:
    decisions: dict[str, dict[str, str | bool]] = {}
    for name, report in raw_reports.items():
        baseline_rows = report["baseline_shape"][0]
        overlap_rate = report["baseline_overlap_rate"]
        required_columns_missing = report["required_columns_missing"]
        duplicate_rate = report["scraped_duplicate_rate"]
        ready = bool(not required_columns_missing and overlap_rate >= 0.95 and duplicate_rate == 0.0)
        if ready:
            reason = "Scraped replacement covers the file's unique required information well enough that remaining gaps are not material to the current feature pipeline."
        else:
            reason = (
                f"Not ready: overlap with imported baseline is {overlap_rate:.1%} on {baseline_rows} rows, "
                f"required columns missing={required_columns_missing}, scraped duplicate rate={duplicate_rate:.4f}."
            )
        decisions[f"{name}.csv"] = {"can_retire": ready, "reason": reason}

    master_overlap = master_report["baseline_overlap_rate"]
    winner_consistency = master_report["winner_consistency"]
    physical_coverage = min(master_report.get("physical_coverage", {}).values(), default=0.0)
    odds_coverage = min(master_report.get("odds_coverage", {}).values(), default=0.0)
    master_ready = bool(
        master_overlap >= 0.95
        and (np.isnan(winner_consistency) or winner_consistency >= 0.999)
        and physical_coverage >= 0.95
        and odds_coverage >= 0.95
    )
    raw_training_ready = all(
        decisions[filename]["can_retire"]
        for filename in ["ufc_fight_results.csv", "ufc_fight_stats.csv", "ufc_fighter_details.csv"]
    )
    target_consistency = feature_report.get("target_consistency", np.nan)
    scraped_total_null_cells = int(feature_report.get("scraped_total_null_cells", 1))
    master_training_ready = bool(
        raw_training_ready
        and not np.isnan(target_consistency)
        and target_consistency >= 0.999
        and scraped_total_null_cells == 0
    )
    if master_ready:
        master_reason = "Scraped assembled dataset matches the imported master closely enough to start replacing it."
    else:
        master_reason = (
            f"Not ready: imported-master overlap is {master_overlap:.1%}, "
            f"winner consistency is {winner_consistency if not np.isnan(winner_consistency) else 'n/a'}, "
            f"physical coverage floor is {physical_coverage:.1%}, odds coverage floor is {odds_coverage:.1%}."
        )
    decisions["ufc-master.csv"] = {"can_retire": master_ready, "reason": master_reason}
    if master_training_ready:
        training_reason = (
            "The self-owned raw results/stats/details contract is sufficient, the rebuilt feature dataset is null-free, "
            "and target consistency is at threshold; imported ufc-master is no longer required for training."
        )
    else:
        training_reason = (
            f"Not ready for training-only retirement: raw_training_ready={raw_training_ready}, "
            f"target_consistency={target_consistency if not np.isnan(target_consistency) else 'n/a'}, "
            f"scraped_feature_null_cells={scraped_total_null_cells}."
        )
    decisions["ufc-master.csv (training_only)"] = {"can_retire": master_training_ready, "reason": training_reason}
    decisions["ufc-master.csv (betting)"] = {"can_retire": master_ready, "reason": master_reason}
    return decisions


def compare_historical_datasets(
    baseline_data_dir: str | Path = DEFAULT_BASELINE_DIR,
    backfill_data_dir: str | Path = DEFAULT_BACKFILL_DIR,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    json_path: str | Path = DEFAULT_JSON_PATH,
) -> dict[str, object]:
    baseline_dir = Path(baseline_data_dir)
    backfill_dir = Path(backfill_data_dir)

    raw_reports = {name: _raw_file_report(name, baseline_dir, backfill_dir) for name in RAW_FILE_CONFIG}
    master_report = _build_master_level_report(baseline_dir, backfill_dir)
    feature_report = _build_training_feature_report(baseline_dir, backfill_dir)
    decisions = _retirement_decisions(raw_reports, master_report, feature_report)
    date_validation_path = backfill_dir / "historical_date_validation_report.json"
    date_validation = json.loads(date_validation_path.read_text(encoding="utf-8")) if date_validation_path.exists() else {}

    report = {
        "legacy_data_audit": LEGACY_DATA_AUDIT,
        "minimum_self_owned_contract": MINIMUM_SELF_OWNED_CONTRACT,
        "completion_thresholds": COMPLETION_THRESHOLDS,
        "raw_file_comparison": raw_reports,
        "assembled_master_comparison": master_report,
        "training_feature_comparison": feature_report,
        "historical_date_validation": date_validation,
        "retirement_decisions": decisions,
    }

    report_path = Path(report_path)
    json_path = Path(json_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    markdown_lines = [
        "# Historical Backfill Comparison",
        "",
        "## Purpose",
        "- Compare imported baseline contracts against the self-owned scraped backfill at both the raw-file level and the assembled historical dataset level.",
        "",
        "## Combined Legacy Contract Audit",
    ]
    for filename, audit in LEGACY_DATA_AUDIT.items():
        markdown_lines.extend(
            [
                "",
                f"### {filename}",
                f"- Unique required contribution: {audit['contribution']}",
                f"- Unique essential columns: `{audit['unique_essential']}`",
                f"- Duplicated but useful: `{audit['duplicated_but_useful']}`",
                f"- Duplicated and unnecessary: `{audit['duplicated_and_unnecessary']}`",
                f"- Unused by current feature pipeline: `{audit['unused_by_current_feature_pipeline']}`",
            ]
        )

    markdown_lines.extend(
        [
            "",
            "## Minimum Self-Owned Data Contract",
            f"- Required identifiers: `{MINIMUM_SELF_OWNED_CONTRACT['identifiers']}`",
            f"- Required result fields: `{MINIMUM_SELF_OWNED_CONTRACT['results']}`",
            f"- Required fighter bio fields: `{MINIMUM_SELF_OWNED_CONTRACT['fighter_bio']}`",
            f"- Required stats fields: `{MINIMUM_SELF_OWNED_CONTRACT['stats']}`",
            f"- Required time fields: `{MINIMUM_SELF_OWNED_CONTRACT['time_features']}`",
            f"- Required market fields only for EV/strategy use: `{MINIMUM_SELF_OWNED_CONTRACT['market_data']}`",
            f"- Not required because redundant or unused: `{MINIMUM_SELF_OWNED_CONTRACT['not_required_due_to_redundancy']}`",
            "",
            "## Completion Thresholds",
            f"- Training-data independence: `{COMPLETION_THRESHOLDS['training_data_independence']}`",
            f"- Betting-data independence: `{COMPLETION_THRESHOLDS['betting_data_independence']}`",
            "",
            "## Level A: Raw File Comparison",
        ]
    )
    for name, raw_report in raw_reports.items():
        markdown_lines.extend(
            [
                "",
                f"### {name}.csv",
                f"- Imported baseline shape: `{tuple(raw_report['baseline_shape'])}`",
                f"- Scraped shape: `{tuple(raw_report['scraped_shape'])}`",
                f"- Exact column match: `{raw_report['exact_column_match']}`",
                f"- Baseline overlap rate: `{raw_report['baseline_overlap_rate']:.1%}`",
                f"- Scraped overlap rate: `{raw_report['scraped_overlap_rate']:.1%}`",
                f"- Scraped duplicate rate: `{raw_report['scraped_duplicate_rate']:.4f}`",
                f"- Required columns for feature generation: `{raw_report['required_columns']}`",
                f"- Required columns missing in scraped build: `{raw_report['required_columns_missing']}`",
                f"- Missing columns that do not matter to the current feature set: `{raw_report['non_material_columns']}`",
                f"- Material blocker: {raw_report['material_blocker'] or 'none'}",
            ]
        )
        if raw_report["only_in_baseline"]:
            markdown_lines.append(f"- Only in imported baseline: `{raw_report['only_in_baseline']}`")
        if raw_report["only_in_scraped"]:
            markdown_lines.append(f"- Only in scraped build: `{raw_report['only_in_scraped']}`")
        if raw_report["discrepancies"]["columns"]:
            markdown_lines.append("- Key field discrepancies:")
            for discrepancy in raw_report["discrepancies"]["columns"][:5]:
                markdown_lines.append(f"  - `{discrepancy['column']}` mismatches: `{discrepancy['mismatch_count']}`")

    markdown_lines.extend(
        [
            "",
            "## Level B: Final Assembled Dataset Comparison",
            f"- Imported `ufc-master.csv` shape: `{tuple(master_report['baseline_shape'])}`",
            f"- Scraped `ufc_master_scraped.csv` shape: `{tuple(master_report['scraped_shape'])}`",
            f"- Shared columns: `{master_report['shared_columns']}`",
            f"- Overlapping fights by date + fighter pair key: `{master_report['overlap_fights']}`",
            f"- Winner consistency on overlap: `{master_report['winner_consistency']:.4f}`" if not np.isnan(master_report["winner_consistency"]) else "- Winner consistency on overlap: `n/a`",
            f"- Physical coverage: `{master_report['physical_coverage']}`",
            f"- Odds coverage: `{master_report['odds_coverage']}`",
            "",
            "## Training-Ready Feature Comparison",
            f"- Imported rebuilt feature shape: `{tuple(feature_report['baseline_shape'])}`",
            f"- Scraped rebuilt feature shape: `{tuple(feature_report['scraped_shape'])}`",
            f"- Overlapping fights by canonical event + matchup key: `{feature_report['overlap_fights']}`",
            f"- Target consistency on overlap: `{feature_report['target_consistency']:.4f}`" if not np.isnan(feature_report["target_consistency"]) else "- Target consistency on overlap: `n/a`",
            "",
            "## Historical Date Validation",
            f"- Date validation report: `{date_validation}`",
            "",
            "## Retirement Decisions",
        ]
    )
    for filename, decision in decisions.items():
        markdown_lines.append(f"- `{filename}`: `{'yes' if decision['can_retire'] else 'no'}` - {decision['reason']}")

    report_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare imported and scraped historical UFC datasets.")
    parser.add_argument("--baseline-data-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--backfill-data-dir", type=Path, default=DEFAULT_BACKFILL_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    args = parser.parse_args()
    report = compare_historical_datasets(
        baseline_data_dir=args.baseline_data_dir,
        backfill_data_dir=args.backfill_data_dir,
        report_path=args.report_path,
        json_path=args.json_path,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
