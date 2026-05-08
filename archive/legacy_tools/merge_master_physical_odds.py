from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MAIN_PATH = DATA_DIR / "ufc_fight_results.csv"
MASTER_PATH = DATA_DIR / "ufc-master.csv"
OUTPUT_PATH = DATA_DIR / "merged_dataset_physical_odds.csv"

MASTER_FIELDS = [
    "R_odds",
    "B_odds",
    "R_age",
    "B_age",
    "R_Height_cms",
    "B_Height_cms",
    "R_Reach_cms",
    "B_Reach_cms",
]
MASTER_RENAME_MAP = {
    "R_odds": "master_R_odds",
    "B_odds": "master_B_odds",
    "R_age": "master_R_age",
    "B_age": "master_B_age",
    "R_Height_cms": "master_R_Height_cms",
    "B_Height_cms": "master_B_Height_cms",
    "R_Reach_cms": "master_R_Reach_cms",
    "B_Reach_cms": "master_B_Reach_cms",
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
    df_master = pd.read_csv(master_path, usecols=["R_fighter", "B_fighter"] + MASTER_FIELDS)
    df_master = df_master.rename(columns=MASTER_RENAME_MAP)
    helper_df = pd.DataFrame(
        {
            "R_fighter_norm": df_master["R_fighter"].map(normalize_name),
            "B_fighter_norm": df_master["B_fighter"].map(normalize_name),
        }
    )
    helper_df["pair_key_unordered"] = helper_df.apply(
        lambda row: " || ".join(sorted([row["R_fighter_norm"], row["B_fighter_norm"]])),
        axis=1,
    )
    return pd.concat([df_master, helper_df], axis=1)


def prepare_master_for_join(df_master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    pair_counts = (
        df_master.groupby("pair_key_unordered", as_index=False)
        .size()
        .rename(columns={"size": "master_pair_count"})
    )
    unique_master = df_master.merge(pair_counts, on="pair_key_unordered", how="left")
    ambiguous_pairs = int(unique_master.loc[unique_master["master_pair_count"] > 1, "pair_key_unordered"].nunique())
    unique_master = unique_master[unique_master["master_pair_count"] == 1].copy()

    exact_join = unique_master.copy()
    exact_join["fighter_A"] = exact_join["R_fighter_norm"]
    exact_join["fighter_B"] = exact_join["B_fighter_norm"]
    exact_join["master_match_status"] = "exact"

    swapped_join = unique_master.copy()
    swapped_join["fighter_A"] = swapped_join["B_fighter_norm"]
    swapped_join["fighter_B"] = swapped_join["R_fighter_norm"]
    swapped_join["master_match_status"] = "swapped"

    join_columns = ["fighter_A", "fighter_B", "master_match_status"] + list(MASTER_RENAME_MAP.values())
    return exact_join[join_columns].copy(), swapped_join[join_columns].copy(), ambiguous_pairs


def merge_datasets(df_main: pd.DataFrame, exact_join: pd.DataFrame, swapped_join: pd.DataFrame) -> pd.DataFrame:
    exact_merged = df_main.merge(exact_join, on=["fighter_A", "fighter_B"], how="left")
    swapped_merged = df_main.merge(swapped_join, on=["fighter_A", "fighter_B"], how="left", suffixes=("", "_swapped"))

    df_merged = exact_merged.copy()
    supplemental_cols = ["master_match_status"] + list(MASTER_RENAME_MAP.values())
    for col in supplemental_cols:
        swapped_col = f"{col}_swapped"
        if swapped_col in swapped_merged.columns:
            df_merged[col] = df_merged[col].combine_first(swapped_merged[swapped_col])

    df_merged["master_match_status"] = df_merged["master_match_status"].fillna("unmatched")
    df_merged["master_matched"] = df_merged["master_match_status"] != "unmatched"
    return df_merged


def add_oriented_columns(df_merged: pd.DataFrame) -> pd.DataFrame:
    """Create fighter_A-oriented physical and odds columns plus matchup diffs."""
    exact_mask = df_merged["master_match_status"] == "exact"

    oriented = df_merged.assign(
        A_odds=np.where(exact_mask, df_merged["master_R_odds"], df_merged["master_B_odds"]),
        B_odds=np.where(exact_mask, df_merged["master_B_odds"], df_merged["master_R_odds"]),
        A_age=np.where(exact_mask, df_merged["master_R_age"], df_merged["master_B_age"]),
        B_age=np.where(exact_mask, df_merged["master_B_age"], df_merged["master_R_age"]),
        A_height_cms=np.where(exact_mask, df_merged["master_R_Height_cms"], df_merged["master_B_Height_cms"]),
        B_height_cms=np.where(exact_mask, df_merged["master_B_Height_cms"], df_merged["master_R_Height_cms"]),
        A_reach_cms=np.where(exact_mask, df_merged["master_R_Reach_cms"], df_merged["master_B_Reach_cms"]),
        B_reach_cms=np.where(exact_mask, df_merged["master_B_Reach_cms"], df_merged["master_R_Reach_cms"]),
    ).copy()

    numeric_cols = [
        "A_odds",
        "B_odds",
        "A_age",
        "B_age",
        "A_height_cms",
        "B_height_cms",
        "A_reach_cms",
        "B_reach_cms",
    ]
    for col in numeric_cols:
        oriented[col] = pd.to_numeric(oriented[col], errors="coerce")

    oriented["age_diff"] = oriented["A_age"] - oriented["B_age"]
    oriented["height_diff"] = oriented["A_height_cms"] - oriented["B_height_cms"]
    oriented["reach_diff"] = oriented["A_reach_cms"] - oriented["B_reach_cms"]
    return oriented.copy()


def build_summary(df_main: pd.DataFrame, df_merged: pd.DataFrame, ambiguous_pairs: int) -> dict[str, object]:
    rows_before = len(df_main)
    rows_after = len(df_merged)
    matched_rows = int(df_merged["master_matched"].sum())
    match_rate = matched_rows / rows_before if rows_before else 0.0
    duplicates_created = int(df_merged["main_row_id"].duplicated().sum())

    merged_fields = [
        "master_R_odds",
        "master_B_odds",
        "master_R_age",
        "master_B_age",
        "master_R_Height_cms",
        "master_B_Height_cms",
        "master_R_Reach_cms",
        "master_B_Reach_cms",
        "A_odds",
        "B_odds",
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
    missing_counts = {col: int(df_merged[col].isna().sum()) for col in merged_fields}

    matched_sample = df_merged[df_merged["master_matched"]][
        [
            "EVENT",
            "BOUT",
            "master_match_status",
            "master_R_odds",
            "master_B_odds",
            "A_odds",
            "B_odds",
            "A_age",
            "B_age",
            "age_diff",
            "height_diff",
            "reach_diff",
        ]
    ].head(5)

    return {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "matched_rows": matched_rows,
        "match_rate": match_rate,
        "duplicates_created": duplicates_created,
        "ambiguous_pairs_excluded": ambiguous_pairs,
        "missing_counts": missing_counts,
        "matched_sample": matched_sample,
    }


def print_summary(summary: dict[str, object]) -> None:
    print("Physical + Odds Merge Summary")
    print(f"Rows before merge: {summary['rows_before']}")
    print(f"Rows after merge: {summary['rows_after']}")
    print(f"Matched rows: {summary['matched_rows']}")
    print(f"Match rate: {summary['match_rate']:.2%}")
    print(f"Duplicate rows created: {summary['duplicates_created']}")
    print(f"Ambiguous fighter-pair groups excluded: {summary['ambiguous_pairs_excluded']}")
    print("Date-aware matching:")
    print("  Main dataset does not include a fight date column, so the enrichment uses fighter-name keys only.")
    print()

    print("Missing Counts For Merged Fields")
    for col, count in summary["missing_counts"].items():
        print(f"  {col}: {count}")
    print()

    print("Example Matched Rows")
    if summary["matched_sample"].empty:
        print("  No matched rows found.")
    else:
        print(summary["matched_sample"].to_string(index=False))


def main() -> None:
    df_main = build_main_dataset(MAIN_PATH)
    df_master = build_master_dataset(MASTER_PATH)
    exact_join, swapped_join, ambiguous_pairs = prepare_master_for_join(df_master)
    df_merged = merge_datasets(df_main, exact_join, swapped_join)
    df_output = add_oriented_columns(df_merged)
    df_output.to_csv(OUTPUT_PATH, index=False)

    summary = build_summary(df_main, df_output, ambiguous_pairs)
    print_summary(summary)
    print()
    print(f"Saved output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
