from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scrapers.common import american_to_implied_probability, build_pair_key, normalize_fighter_name


LOGIT_PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "modeling" / "tuning" / "best_calibrated_logit_predictions.csv"
RF_PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "modeling" / "tuning" / "baseline_predictions_rf.csv"
FEATURE_DATASET_PATH = PROJECT_ROOT / "data" / "historical_backfill" / "ufc_rebuilt_features_scraped.csv"
EVENT_CATALOG_PATH = PROJECT_ROOT / "data" / "historical_backfill" / "historical_event_catalog_scraped.csv"
UFC_MASTER_PATH = PROJECT_ROOT / "data" / "legacy_betting" / "ufc-master.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "strategy" / "betting_ready.csv"
DATA_ALIAS_OUTPUT_PATH = PROJECT_ROOT / "data" / "betting_ready.csv"


def parse_fight_id(fight_id: str) -> tuple[str, str, str, str]:
    event_name, bout = str(fight_id).split(" | ", maxsplit=1)
    fighter_a, fighter_b = bout.split(" vs. ", maxsplit=1)
    return event_name.strip(), bout.strip(), fighter_a.strip(), fighter_b.strip()


def load_predictions(logit_path: Path, rf_path: Path) -> pd.DataFrame:
    logit_df = pd.read_csv(logit_path)
    logit_df = logit_df.rename(
        columns={
            "predicted_probability": "p_model_A",
            "y_true": "actual_outcome",
        }
    )
    metadata = logit_df["fight_id"].map(parse_fight_id).apply(pd.Series)
    metadata.columns = ["event_name", "bout", "fighter_A", "fighter_B"]
    logit_df = pd.concat([logit_df, metadata], axis=1)

    rf_df = pd.read_csv(rf_path)[["fight_id", "fold", "predicted_probability"]].rename(
        columns={"predicted_probability": "rf_probability"}
    )
    merged = logit_df.merge(rf_df, on=["fight_id", "fold"], how="left")
    merged["logit_pick"] = (merged["p_model_A"] >= 0.5).astype(int)
    merged["rf_pick"] = np.where(merged["rf_probability"].notna(), (merged["rf_probability"] >= 0.5).astype(int), np.nan)
    merged["model_agreement"] = np.where(
        merged["rf_pick"].notna(),
        merged["logit_pick"] == merged["rf_pick"],
        False,
    )
    return merged


def attach_event_dates(predictions_df: pd.DataFrame, event_catalog_path: Path) -> pd.DataFrame:
    event_catalog_df = pd.read_csv(event_catalog_path)[["EVENT", "DATE", "LOCATION"]].rename(
        columns={"EVENT": "event_name", "DATE": "event_date", "LOCATION": "event_location"}
    )
    merged = predictions_df.merge(event_catalog_df, on="event_name", how="left")
    merged["event_date"] = pd.to_datetime(merged["event_date"], errors="coerce")
    return merged


def load_master_odds(master_path: Path) -> pd.DataFrame:
    master_df = pd.read_csv(master_path)
    odds_df = master_df[
        ["R_fighter", "B_fighter", "date", "R_odds", "B_odds", "Winner", "weight_class", "title_bout"]
    ].copy()
    odds_df["event_date"] = pd.to_datetime(odds_df["date"], errors="coerce")
    odds_df["R_fighter_normalized"] = odds_df["R_fighter"].map(normalize_fighter_name)
    odds_df["B_fighter_normalized"] = odds_df["B_fighter"].map(normalize_fighter_name)
    odds_df["pair_key"] = odds_df.apply(
        lambda row: build_pair_key(row["R_fighter_normalized"], row["B_fighter_normalized"]),
        axis=1,
    )
    odds_df["odds_row_id"] = np.arange(len(odds_df))
    return odds_df


def merge_predictions_with_odds(predictions_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    merged = predictions_df.copy()
    merged["fighter_A_normalized"] = merged["fighter_A"].map(normalize_fighter_name)
    merged["fighter_B_normalized"] = merged["fighter_B"].map(normalize_fighter_name)
    merged["pair_key"] = merged.apply(
        lambda row: build_pair_key(row["fighter_A_normalized"], row["fighter_B_normalized"]),
        axis=1,
    )

    odds_primary = odds_df.drop_duplicates(subset=["pair_key", "event_date"], keep="last").copy()
    merged = merged.merge(
        odds_primary[
            [
                "pair_key",
                "event_date",
                "R_fighter",
                "B_fighter",
                "R_fighter_normalized",
                "B_fighter_normalized",
                "R_odds",
                "B_odds",
                "Winner",
                "weight_class",
                "title_bout",
                "odds_row_id",
            ]
        ],
        on=["pair_key", "event_date"],
        how="left",
    )

    exact_orientation = (
        (merged["fighter_A_normalized"] == merged["R_fighter_normalized"])
        & (merged["fighter_B_normalized"] == merged["B_fighter_normalized"])
    )
    swapped_orientation = (
        (merged["fighter_A_normalized"] == merged["B_fighter_normalized"])
        & (merged["fighter_B_normalized"] == merged["R_fighter_normalized"])
    )

    merged["merge_method"] = np.where(
        merged["odds_row_id"].notna(),
        "pair_key_plus_date",
        "unmatched",
    )
    merged["orientation_method"] = np.select(
        [exact_orientation, swapped_orientation],
        ["exact", "swapped"],
        default="unresolved",
    )

    merged["odds_A"] = np.where(exact_orientation, merged["R_odds"], np.where(swapped_orientation, merged["B_odds"], np.nan))
    merged["odds_B"] = np.where(exact_orientation, merged["B_odds"], np.where(swapped_orientation, merged["R_odds"], np.nan))
    merged["winner_aligned_to_A"] = np.where(
        exact_orientation,
        merged["Winner"],
        np.where(swapped_orientation, np.where(merged["Winner"] == "Red", "Blue", np.where(merged["Winner"] == "Blue", "Red", merged["Winner"])), np.nan),
    )
    return merged


def finalize_betting_dataset(predictions_df: pd.DataFrame) -> pd.DataFrame:
    final_df = predictions_df.copy()
    final_df["fight_order"] = pd.to_numeric(final_df["fight_order"], errors="coerce")
    final_df["actual_outcome"] = pd.to_numeric(final_df["actual_outcome"], errors="coerce")
    final_df["implied_prob_A"] = final_df["odds_A"].map(american_to_implied_probability)
    final_df["implied_prob_B"] = final_df["odds_B"].map(american_to_implied_probability)
    final_df["edge_A"] = final_df["p_model_A"] - final_df["implied_prob_A"]
    final_df["edge_B"] = (1.0 - final_df["p_model_A"]) - final_df["implied_prob_B"]
    final_df["has_valid_odds"] = final_df["odds_A"].notna() & final_df["odds_B"].notna()
    final_df = final_df.sort_values(["fight_order", "fight_id"]).reset_index(drop=True)

    ordered_columns = [
        "fight_id",
        "fight_order",
        "fold",
        "event_name",
        "event_date",
        "event_location",
        "bout",
        "fighter_A",
        "fighter_B",
        "fighter_A_normalized",
        "fighter_B_normalized",
        "actual_outcome",
        "p_model_A",
        "rf_probability",
        "logit_pick",
        "rf_pick",
        "model_agreement",
        "R_fighter",
        "B_fighter",
        "odds_A",
        "odds_B",
        "implied_prob_A",
        "implied_prob_B",
        "edge_A",
        "edge_B",
        "weight_class",
        "title_bout",
        "merge_method",
        "orientation_method",
        "has_valid_odds",
    ]
    remaining_columns = [column for column in final_df.columns if column not in ordered_columns]
    return final_df[ordered_columns + remaining_columns]


def build_betting_dataset(
    *,
    logit_predictions_path: Path = LOGIT_PREDICTIONS_PATH,
    rf_predictions_path: Path = RF_PREDICTIONS_PATH,
    event_catalog_path: Path = EVENT_CATALOG_PATH,
    master_odds_path: Path = UFC_MASTER_PATH,
    output_path: Path = OUTPUT_PATH,
    data_alias_output_path: Path | None = DATA_ALIAS_OUTPUT_PATH,
) -> pd.DataFrame:
    predictions_df = load_predictions(logit_predictions_path, rf_predictions_path)
    predictions_df = attach_event_dates(predictions_df, event_catalog_path)
    odds_df = load_master_odds(master_odds_path)
    merged_df = merge_predictions_with_odds(predictions_df, odds_df)
    final_df = finalize_betting_dataset(merged_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    if data_alias_output_path is not None:
        data_alias_output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(data_alias_output_path, index=False)
    return final_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 3 betting-ready dataset.")
    parser.add_argument("--logit-predictions", type=Path, default=LOGIT_PREDICTIONS_PATH)
    parser.add_argument("--rf-predictions", type=Path, default=RF_PREDICTIONS_PATH)
    parser.add_argument("--event-catalog", type=Path, default=EVENT_CATALOG_PATH)
    parser.add_argument("--master-odds", type=Path, default=UFC_MASTER_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--data-output-alias", type=Path, default=DATA_ALIAS_OUTPUT_PATH)
    args = parser.parse_args()

    final_df = build_betting_dataset(
        logit_predictions_path=args.logit_predictions,
        rf_predictions_path=args.rf_predictions,
        event_catalog_path=args.event_catalog,
        master_odds_path=args.master_odds,
        output_path=args.output,
        data_alias_output_path=args.data_output_alias,
    )
    valid_odds_rate = final_df["has_valid_odds"].mean() if len(final_df) else 0.0
    print(f"Saved betting-ready dataset: {args.output}")
    print(f"Rows: {len(final_df)}")
    print(f"Fights with valid odds: {int(final_df['has_valid_odds'].sum())} ({valid_odds_rate:.1%})")


if __name__ == "__main__":
    main()
