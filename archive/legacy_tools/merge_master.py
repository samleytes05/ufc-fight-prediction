from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MAIN_PATH = DATA_DIR / "ufc_fight_results.csv"
MASTER_PATH = DATA_DIR / "ufc-master.csv"
OUTPUT_PATH = DATA_DIR / "merged_dataset.csv"

LEAKAGE_COLUMNS = {
    "Winner",
    "finish",
    "finish_details",
    "finish_round",
    "finish_round_time",
    "total_fight_time_secs",
}


def normalize_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def build_main_dataset(main_path: Path) -> pd.DataFrame:
    df_main = pd.read_csv(main_path)
    fighters_split = df_main["BOUT"].astype(str).str.split(" vs. ", expand=True)
    df_main["fighter_A"] = fighters_split[0].fillna("").map(normalize_name)
    df_main["fighter_B"] = fighters_split[1].fillna("").map(normalize_name)
    df_main["main_row_id"] = range(len(df_main))
    return df_main


def build_master_dataset(master_path: Path) -> pd.DataFrame:
    df_master = pd.read_csv(master_path)
    helper_df = pd.DataFrame(
        {
            "R_fighter_norm": df_master["R_fighter"].map(normalize_name),
            "B_fighter_norm": df_master["B_fighter"].map(normalize_name),
        }
    )
    helper_df["pair_key_ordered"] = helper_df["R_fighter_norm"] + " || " + helper_df["B_fighter_norm"]
    helper_df["pair_key_unordered"] = helper_df.apply(
        lambda row: " || ".join(sorted([row["R_fighter_norm"], row["B_fighter_norm"]])),
        axis=1,
    )
    return pd.concat([df_master, helper_df], axis=1)


def classify_master_columns(df_master: pd.DataFrame) -> dict[str, list[str]]:
    helper_columns = {"R_fighter_norm", "B_fighter_norm", "pair_key_ordered", "pair_key_unordered"}
    master_columns = [col for col in df_master.columns if col not in helper_columns]

    odds_columns = [
        col
        for col in master_columns
        if "odds" in col.lower() or col.lower().endswith("_ev") or col in {"R_ev", "B_ev"}
    ]
    leakage_columns = [col for col in master_columns if col in LEAKAGE_COLUMNS]
    safe_prefight_columns = [
        col for col in master_columns if col not in set(odds_columns) and col not in set(leakage_columns)
    ]

    return {
        "safe_prefight": sorted(safe_prefight_columns),
        "odds_market": sorted(odds_columns),
        "postfight_leakage": sorted(leakage_columns),
    }


def prepare_master_for_join(df_master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_counts = (
        df_master.groupby("pair_key_unordered", as_index=False)
        .size()
        .rename(columns={"size": "master_pair_count"})
    )

    unique_master = df_master.merge(pair_counts, on="pair_key_unordered", how="left")
    ambiguous_master = unique_master[unique_master["master_pair_count"] > 1].copy()
    unique_master = unique_master[unique_master["master_pair_count"] == 1].copy()

    exact_join = unique_master.copy()
    exact_join["fighter_A"] = exact_join["R_fighter_norm"]
    exact_join["fighter_B"] = exact_join["B_fighter_norm"]
    exact_join["master_match_status"] = "exact"

    swapped_join = unique_master.copy()
    swapped_join["fighter_A"] = swapped_join["B_fighter_norm"]
    swapped_join["fighter_B"] = swapped_join["R_fighter_norm"]
    swapped_join["master_match_status"] = "swapped"

    join_columns = [
        "fighter_A",
        "fighter_B",
        "master_match_status",
        "master_pair_count",
    ] + [col for col in unique_master.columns if col not in {"R_fighter_norm", "B_fighter_norm", "pair_key_ordered"}]

    return exact_join[join_columns].copy(), swapped_join[join_columns].copy(), ambiguous_master.copy()


def merge_datasets(df_main: pd.DataFrame, exact_join: pd.DataFrame, swapped_join: pd.DataFrame) -> pd.DataFrame:
    exact_merged = df_main.merge(
        exact_join,
        on=["fighter_A", "fighter_B"],
        how="left",
    )
    swapped_merged = df_main.merge(
        swapped_join,
        on=["fighter_A", "fighter_B"],
        how="left",
        suffixes=("", "_swapped"),
    )

    supplemental_cols = [col for col in exact_join.columns if col not in {"fighter_A", "fighter_B"}]
    df_merged = exact_merged.copy()
    for col in supplemental_cols:
        swapped_col = f"{col}_swapped"
        if swapped_col in swapped_merged.columns:
            df_merged[col] = df_merged[col].combine_first(swapped_merged[swapped_col])

    df_merged["master_matched"] = df_merged["R_fighter"].notna()
    df_merged["master_match_status"] = df_merged["master_match_status"].fillna("unmatched")
    return df_merged


def build_summary(
    df_main: pd.DataFrame,
    df_merged: pd.DataFrame,
    ambiguous_master: pd.DataFrame,
    classified_columns: dict[str, list[str]],
) -> dict[str, object]:
    total_rows_before = len(df_main)
    total_rows_after = len(df_merged)
    matched_rows = int(df_merged["master_matched"].sum())
    unmatched_rows = total_rows_after - matched_rows
    match_rate = matched_rows / total_rows_before if total_rows_before else 0.0
    duplicates_created = int(df_merged["main_row_id"].duplicated().sum())

    key_merged_columns = [
        col
        for col in ["R_odds", "B_odds", "date", "location", "weight_class", "R_age", "B_age", "Winner"]
        if col in df_merged.columns
    ]
    missing_key_counts = {col: int(df_merged[col].isna().sum()) for col in key_merged_columns}

    matched_sample = df_merged[df_merged["master_matched"]][
        ["EVENT", "BOUT", "fighter_A", "fighter_B", "master_match_status", "date"]
    ].head(5)
    unmatched_sample = df_merged[~df_merged["master_matched"]][
        ["EVENT", "BOUT", "fighter_A", "fighter_B", "master_match_status"]
    ].head(5)

    return {
        "rows_before": total_rows_before,
        "rows_after": total_rows_after,
        "matched_rows": matched_rows,
        "unmatched_rows": unmatched_rows,
        "match_rate": match_rate,
        "duplicates_created": duplicates_created,
        "ambiguous_master_pairs": int(ambiguous_master["pair_key_unordered"].nunique()),
        "missing_key_counts": missing_key_counts,
        "matched_sample": matched_sample,
        "unmatched_sample": unmatched_sample,
        "classified_columns": classified_columns,
    }


def print_summary(summary: dict[str, object]) -> None:
    print("Merge Summary")
    print(f"Rows before merge: {summary['rows_before']}")
    print(f"Rows after merge: {summary['rows_after']}")
    print(f"Matched rows: {summary['matched_rows']}")
    print(f"Unmatched rows: {summary['unmatched_rows']}")
    print(f"Match rate: {summary['match_rate']:.2%}")
    print(f"Duplicate rows created: {summary['duplicates_created']}")
    print(f"Ambiguous master pairs excluded from matching: {summary['ambiguous_master_pairs']}")
    print("Date-aware matching:")
    print("  Main dataset does not include a fight date column, so merge used fighter-name keys only.")
    print()

    print("Merged Column Classification")
    print(f"Safe pre-fight features: {len(summary['classified_columns']['safe_prefight'])}")
    print(", ".join(summary["classified_columns"]["safe_prefight"][:20]))
    print(f"Odds / market data: {len(summary['classified_columns']['odds_market'])}")
    print(", ".join(summary["classified_columns"]["odds_market"]))
    print(f"Post-fight leakage: {len(summary['classified_columns']['postfight_leakage'])}")
    print(", ".join(summary["classified_columns"]["postfight_leakage"]))
    print()

    print("Missing Values For Key Merged Columns")
    for col, missing_count in summary["missing_key_counts"].items():
        print(f"  {col}: {missing_count}")
    print()

    print("Sample Matched Fights")
    if summary["matched_sample"].empty:
        print("  No matched fights found.")
    else:
        print(summary["matched_sample"].to_string(index=False))
    print()

    print("Sample Unmatched Fights")
    if summary["unmatched_sample"].empty:
        print("  No unmatched fights found.")
    else:
        print(summary["unmatched_sample"].to_string(index=False))


def main() -> None:
    df_main = build_main_dataset(MAIN_PATH)
    df_master = build_master_dataset(MASTER_PATH)
    classified_columns = classify_master_columns(df_master)
    exact_join, swapped_join, ambiguous_master = prepare_master_for_join(df_master)
    df_merged = merge_datasets(df_main, exact_join, swapped_join)
    df_merged.to_csv(OUTPUT_PATH, index=False)
    summary = build_summary(df_main, df_merged, ambiguous_master, classified_columns)
    print_summary(summary)
    print()
    print(f"Saved merged dataset: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
