from __future__ import annotations

"""Feature-engineering pipeline for UFC outcome modeling."""

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from src.scrapers.common import normalize_fighter_name as shared_normalize_fighter_name
except ImportError:  # pragma: no cover
    shared_normalize_fighter_name = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "historical_backfill"
DEFAULT_OUTPUT_PATH = DEFAULT_DATA_DIR / "ufc_rebuilt_features_scraped.csv"
DEFAULT_PHYSICAL_ODDS_PATH = DEFAULT_DATA_DIR / "historical_physical_features_scraped.csv"

STAT_PLACEHOLDERS = ["---", "--", "nan", "None", ""]
BASE_IDENTIFIER_COLUMNS = [
    "fight_id",
    "fight_order",
    "target_A_win",
]
BASELINE_FEATURE_COLUMNS = [
    "career_win_rate_diff",
    "career_fights_diff",
    "win_streak_diff",
    "l3_sig_str_acc_diff",
    "career_striking_efficiency_diff",
    "l3_striking_efficiency_diff",
    "career_sig_str_absorbed_per_round_diff",
    "l3_sig_str_absorbed_per_round_diff",
    "career_defensive_efficiency_diff",
    "career_damage_efficiency_ratio_diff",
    "career_offense_defense_ratio_diff",
    "l3_offense_defense_ratio_diff",
    "l3_net_striking_diff",
    "career_td_landed_per_round_diff",
    "career_td_acc_diff",
    "l3_td_acc_diff",
    "l3_td_def_diff",
    "career_td_path_diff",
    "career_control_share_diff",
    "career_adjusted_sig_strike_diff",
    "l3_adjusted_sig_strike_diff",
    "l3_pace_diff",
    "career_finish_rate_diff",
    "career_ko_rate_diff",
    "career_sub_rate_diff",
    "career_early_finish_rate_diff",
    "career_avg_fight_time_diff",
    "career_opponent_adjusted_damage_ratio_diff",
    "experience_advantage_ratio_log",
    "A_elo",
    "B_elo",
    "elo_diff",
    "A_avg_elo_last3_opp",
    "B_avg_elo_last3_opp",
    "avg_elo_last3_opp_diff",
]
PHYSICAL_FEATURE_COLUMNS = [
    "age_diff",
    "height_diff",
    "reach_diff",
]
FINAL_FEATURE_COLUMNS = BASE_IDENTIFIER_COLUMNS + BASELINE_FEATURE_COLUMNS
PREFIGHT_STATE_COLUMNS = [
    "career_fights",
    "career_wins",
    "career_win_rate",
    "career_sig_str_landed_per_round",
    "career_sig_str_attempted_per_round",
    "career_total_str_landed_per_round",
    "career_total_str_attempted_per_round",
    "career_td_landed_per_round",
    "career_td_attempted_per_round",
    "career_sub_att_per_round",
    "career_ctrl_seconds_per_round",
    "career_sig_str_acc",
    "career_td_acc",
    "career_sig_str_absorbed_per_round",
    "career_damage_absorbed_ratio",
    "career_defensive_efficiency",
    "career_net_damage_per_round",
    "career_offense_defense_ratio",
    "career_td_def",
    "career_control_share",
    "career_grappling_pressure",
    "career_style_index",
    "career_pace",
    "career_damage_ratio",
    "career_striking_efficiency",
    "career_finish_rate",
    "career_ko_rate",
    "career_sub_rate",
    "career_early_finish_rate",
    "career_avg_fight_time_seconds",
    "win_streak",
    "l3_fights",
    "l3_wins",
    "l3_win_rate",
    "l3_sig_str_landed_per_round",
    "l3_sig_str_attempted_per_round",
    "l3_total_str_landed_per_round",
    "l3_total_str_attempted_per_round",
    "l3_td_landed_per_round",
    "l3_td_attempted_per_round",
    "l3_sub_att_per_round",
    "l3_ctrl_seconds_per_round",
    "l3_sig_str_acc",
    "l3_td_acc",
    "l3_sig_str_absorbed_per_round",
    "l3_damage_absorbed_ratio",
    "l3_defensive_efficiency",
    "l3_net_damage_per_round",
    "l3_offense_defense_ratio",
    "l3_td_def",
    "l3_control_share",
    "l3_grappling_pressure",
    "l3_style_index",
    "l3_pace",
    "l3_damage_ratio",
    "l3_striking_efficiency",
]
HISTORY_SUM_COLUMNS = [
    "KD",
    "sig_str_landed",
    "sig_str_attempted",
    "total_str_landed",
    "total_str_attempted",
    "td_landed",
    "td_attempted",
    "sub_att",
    "ctrl_seconds",
    "rounds_fought",
    "opp_sig_str_landed",
    "opp_sig_str_attempted",
    "opp_td_landed",
    "opp_td_attempted",
    "total_fight_time_seconds",
    "finish_win",
    "ko_win",
    "sub_win",
    "early_finish_win",
]


def split_of_stat(col: pd.Series) -> tuple[pd.Series, pd.Series]:
    extracted = col.astype(str).str.extract(r"(\d+)\s+of\s+(\d+)")
    landed = pd.to_numeric(extracted[0], errors="coerce")
    attempted = pd.to_numeric(extracted[1], errors="coerce")
    return landed, attempted


def pct_to_float(col: pd.Series) -> pd.Series:
    return pd.to_numeric(col.astype(str).str.replace("%", "", regex=False), errors="coerce") / 100


def time_to_seconds(col: pd.Series) -> pd.Series:
    parts = col.astype(str).str.extract(r"(\d+):(\d+)")
    mins = pd.to_numeric(parts[0], errors="coerce")
    secs = pd.to_numeric(parts[1], errors="coerce")
    return mins * 60 + secs


def safe_divide(num: pd.Series, den: pd.Series) -> np.ndarray:
    return np.where(den > 0, num / den, 0)


def normalize_fighter_name(name: str) -> str:
    if shared_normalize_fighter_name is not None:
        return shared_normalize_fighter_name(name)
    return str(name).strip().lower()


def load_physical_differences(physical_dataset_path: str | Path = DEFAULT_PHYSICAL_ODDS_PATH) -> pd.DataFrame:
    physical_path = Path(physical_dataset_path)
    df_physical = pd.read_csv(physical_path)
    required_cols = ["EVENT", "BOUT"] + PHYSICAL_FEATURE_COLUMNS
    missing_cols = [col for col in required_cols if col not in df_physical.columns]
    if missing_cols:
        raise ValueError(f"Physical dataset missing required columns: {missing_cols}")

    df_physical["fight_id"] = (
        df_physical["EVENT"].astype(str).str.strip() + " | " + df_physical["BOUT"].astype(str).str.strip()
    )
    df_physical = df_physical[["fight_id"] + PHYSICAL_FEATURE_COLUMNS].copy()
    df_physical = df_physical.drop_duplicates(subset=["fight_id"], keep="first")
    return df_physical


def add_physical_difference_features(
    df_final: pd.DataFrame,
    physical_dataset_path: str | Path = DEFAULT_PHYSICAL_ODDS_PATH,
) -> pd.DataFrame:
    """Attach optional pre-fight physical matchup differences by fight_id."""
    df_physical = load_physical_differences(physical_dataset_path)
    df_enriched = df_final.merge(df_physical, on="fight_id", how="left")
    for col in PHYSICAL_FEATURE_COLUMNS:
        df_enriched[col] = pd.to_numeric(df_enriched[col], errors="coerce").fillna(0.0)
    return df_enriched


def load_raw_data(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict[str, pd.DataFrame]:
    data_path = Path(data_dir)
    return {
        "results": pd.read_csv(data_path / "ufc_fight_results.csv"),
        "stats": pd.read_csv(data_path / "ufc_fight_stats.csv"),
        "fighters": pd.read_csv(data_path / "ufc_fighter_details.csv"),
    }


def clean_stats_table(df_stats: pd.DataFrame) -> pd.DataFrame:
    df_stats = df_stats.copy()
    df_stats["FIGHTER"] = df_stats["FIGHTER"].astype(str).str.strip().str.lower()
    df_stats["BOUT"] = df_stats["BOUT"].astype(str).str.strip()
    df_stats = df_stats.replace(STAT_PLACEHOLDERS, np.nan)

    df_stats["sig_str_landed"], df_stats["sig_str_attempted"] = split_of_stat(df_stats["SIG.STR."])
    df_stats["sig_str_acc"] = pct_to_float(df_stats["SIG.STR. %"])
    df_stats["total_str_landed"], df_stats["total_str_attempted"] = split_of_stat(df_stats["TOTAL STR."])
    df_stats["td_landed"], df_stats["td_attempted"] = split_of_stat(df_stats["TD"])
    df_stats["td_acc"] = pct_to_float(df_stats["TD %"])
    df_stats["ctrl_seconds"] = time_to_seconds(df_stats["CTRL"])

    for col in ["HEAD", "BODY", "LEG", "DISTANCE", "CLINCH", "GROUND"]:
        landed, attempted = split_of_stat(df_stats[col])
        df_stats[f"{col.lower()}_landed"] = landed
        df_stats[f"{col.lower()}_attempted"] = attempted

    df_stats["ROUND"] = pd.to_numeric(df_stats["ROUND"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

    for col in ["KD", "SUB.ATT", "REV."]:
        df_stats[col] = pd.to_numeric(df_stats[col], errors="coerce")

    cols_to_keep = [
        "EVENT",
        "BOUT",
        "ROUND",
        "FIGHTER",
        "KD",
        "SUB.ATT",
        "REV.",
        "sig_str_landed",
        "sig_str_attempted",
        "sig_str_acc",
        "total_str_landed",
        "total_str_attempted",
        "td_landed",
        "td_attempted",
        "td_acc",
        "ctrl_seconds",
        "head_landed",
        "head_attempted",
        "body_landed",
        "body_attempted",
        "leg_landed",
        "leg_attempted",
        "distance_landed",
        "distance_attempted",
        "clinch_landed",
        "clinch_attempted",
        "ground_landed",
        "ground_attempted",
    ]
    df_stats = df_stats[cols_to_keep].copy()
    df_stats = df_stats.dropna(subset=["FIGHTER", "ROUND", "sig_str_landed", "sig_str_attempted"]).copy()
    return df_stats


def aggregate_fight_level_stats(df_stats: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["EVENT", "BOUT", "FIGHTER"]
    df_fight_level = df_stats.groupby(group_cols, as_index=False).agg(
        {
            "ROUND": "max",
            "KD": "sum",
            "SUB.ATT": "sum",
            "REV.": "sum",
            "sig_str_landed": "sum",
            "sig_str_attempted": "sum",
            "total_str_landed": "sum",
            "total_str_attempted": "sum",
            "td_landed": "sum",
            "td_attempted": "sum",
            "ctrl_seconds": "sum",
            "head_landed": "sum",
            "head_attempted": "sum",
            "body_landed": "sum",
            "body_attempted": "sum",
            "leg_landed": "sum",
            "leg_attempted": "sum",
            "distance_landed": "sum",
            "distance_attempted": "sum",
            "clinch_landed": "sum",
            "clinch_attempted": "sum",
            "ground_landed": "sum",
            "ground_attempted": "sum",
        }
    )

    df_fight_level = df_fight_level.rename(columns={"ROUND": "rounds_fought", "SUB.ATT": "sub_att", "REV.": "rev"})
    df_fight_level["sig_str_acc"] = np.where(
        df_fight_level["sig_str_attempted"] > 0,
        df_fight_level["sig_str_landed"] / df_fight_level["sig_str_attempted"],
        0,
    )
    df_fight_level["td_acc"] = np.where(
        df_fight_level["td_attempted"] > 0,
        df_fight_level["td_landed"] / df_fight_level["td_attempted"],
        0,
    )

    for col in [
        "KD",
        "sig_str_landed",
        "sig_str_attempted",
        "total_str_landed",
        "total_str_attempted",
        "td_landed",
        "td_attempted",
        "sub_att",
        "ctrl_seconds",
    ]:
        df_fight_level[f"{col.lower()}_per_round"] = np.where(
            df_fight_level["rounds_fought"] > 0,
            df_fight_level[col] / df_fight_level["rounds_fought"],
            0,
        )
    return df_fight_level


def clean_results_table(df_results: pd.DataFrame) -> pd.DataFrame:
    df_results = df_results.copy()
    df_results["BOUT"] = df_results["BOUT"].astype(str).str.strip()
    df_results["EVENT"] = df_results["EVENT"].astype(str).str.strip()
    df_results["OUTCOME"] = df_results["OUTCOME"].astype(str).str.strip()
    df_results = df_results.iloc[::-1].reset_index(drop=True)

    fighters_split = df_results["BOUT"].str.split(" vs. ", expand=True)
    df_results["fighter_A"] = fighters_split[0].str.strip().str.lower()
    df_results["fighter_B"] = fighters_split[1].str.strip().str.lower()
    df_results["target_A_win"] = np.where(
        df_results["OUTCOME"] == "W/L",
        1,
        np.where(df_results["OUTCOME"] == "L/W", 0, np.nan),
    )
    df_results["fight_order"] = range(len(df_results))
    df_results["fight_id"] = df_results["EVENT"].astype(str).str.strip() + " | " + df_results["BOUT"].astype(str).str.strip()

    df_fights = df_results[
        [
            "fight_id",
            "fight_order",
            "EVENT",
            "BOUT",
            "fighter_A",
            "fighter_B",
            "target_A_win",
            "WEIGHTCLASS",
            "METHOD",
            "ROUND",
            "TIME",
        ]
    ].copy()
    df_fights = df_fights[df_fights["target_A_win"].isin([0, 1])].copy()
    df_fights["target_A_win"] = df_fights["target_A_win"].astype(int)
    return df_fights


def build_model_base(df_fights: pd.DataFrame, df_fight_level: pd.DataFrame) -> pd.DataFrame:
    df_fight_level = df_fight_level.copy()
    df_fight_level["fight_id"] = df_fight_level["EVENT"].astype(str).str.strip() + " | " + df_fight_level["BOUT"].astype(str).str.strip()

    df_pair = df_fight_level.merge(df_fight_level, on=["EVENT", "BOUT"], suffixes=("", "_opp"))
    df_pair = df_pair[df_pair["FIGHTER"] != df_pair["FIGHTER_opp"]].copy()
    df_pair = df_pair.rename(
        columns={
            "FIGHTER": "fighter",
            "FIGHTER_opp": "opponent",
            "sig_str_landed_opp": "opp_sig_str_landed",
            "sig_str_attempted_opp": "opp_sig_str_attempted",
            "td_landed_opp": "opp_td_landed",
            "td_attempted_opp": "opp_td_attempted",
            "ctrl_seconds_opp": "opp_ctrl_seconds",
        }
    )
    df_pair["total_fight_time_seconds"] = df_pair["rounds_fought"] * 300

    pair_keep_cols = [
        "fight_id",
        "EVENT",
        "BOUT",
        "fighter",
        "opponent",
        "rounds_fought",
        "KD",
        "sub_att",
        "rev",
        "sig_str_landed",
        "sig_str_attempted",
        "total_str_landed",
        "total_str_attempted",
        "td_landed",
        "td_attempted",
        "ctrl_seconds",
        "opp_sig_str_landed",
        "opp_sig_str_attempted",
        "opp_td_landed",
        "opp_td_attempted",
        "opp_ctrl_seconds",
        "total_fight_time_seconds",
    ]
    df_pair = df_pair[pair_keep_cols].copy()

    a_stats = df_pair.copy().rename(columns={"fighter": "fighter_A"})
    a_stats = a_stats.rename(
        columns={col: f"A_{col}" for col in a_stats.columns if col not in ["fight_id", "fighter_A", "EVENT", "BOUT"]}
    )

    b_stats = df_pair.copy().rename(columns={"fighter": "fighter_B"})
    b_stats = b_stats.rename(
        columns={col: f"B_{col}" for col in b_stats.columns if col not in ["fight_id", "fighter_B", "EVENT", "BOUT"]}
    )

    df_model_base = df_fights.merge(a_stats.drop(columns=["EVENT", "BOUT"]), on=["fight_id", "fighter_A"], how="left")
    df_model_base = df_model_base.merge(b_stats.drop(columns=["EVENT", "BOUT"]), on=["fight_id", "fighter_B"], how="left")
    df_model_base = df_model_base.dropna(subset=["A_rounds_fought", "B_rounds_fought"]).reset_index(drop=True)
    return df_model_base


def add_result_based_finish_flags(df_model_base: pd.DataFrame) -> pd.DataFrame:
    df_model_base = df_model_base.copy()
    df_model_base["method_clean"] = df_model_base["METHOD"].astype(str).str.strip().str.lower()
    df_model_base["is_finish"] = (~df_model_base["method_clean"].str.contains("decision", na=False)).astype(int)
    df_model_base["fight_round_end"] = pd.to_numeric(df_model_base["ROUND"], errors="coerce")
    df_model_base["A_finish_win"] = ((df_model_base["target_A_win"] == 1) & (df_model_base["is_finish"] == 1)).astype(int)
    df_model_base["A_ko_win"] = (
        (df_model_base["target_A_win"] == 1)
        & (df_model_base["is_finish"] == 1)
        & (df_model_base["method_clean"].str.contains("ko|tko", regex=True, na=False))
    ).astype(int)
    df_model_base["A_sub_win"] = (
        (df_model_base["target_A_win"] == 1)
        & (df_model_base["is_finish"] == 1)
        & (df_model_base["method_clean"].str.contains("submission", na=False))
    ).astype(int)
    df_model_base["A_early_finish_win"] = (
        (df_model_base["target_A_win"] == 1) & (df_model_base["is_finish"] == 1) & (df_model_base["fight_round_end"] <= 2)
    ).astype(int)
    return df_model_base


def build_long_fighter_history(df_model_base: pd.DataFrame) -> pd.DataFrame:
    a_exclude = {"A_finish_win", "A_ko_win", "A_sub_win", "A_early_finish_win"}
    b_exclude: set[str] = set()
    a_cols = [col for col in df_model_base.columns if col.startswith("A_") and col not in a_exclude]
    b_cols = [col for col in df_model_base.columns if col.startswith("B_") and col not in b_exclude]

    df_a = df_model_base[
        ["fight_id", "fight_order", "fighter_A", "target_A_win", "A_finish_win", "A_ko_win", "A_sub_win", "A_early_finish_win"]
        + a_cols
    ].copy()
    df_a = df_a.rename(
        columns={
            "fighter_A": "fighter",
            "target_A_win": "win",
            "A_finish_win": "finish_win",
            "A_ko_win": "ko_win",
            "A_sub_win": "sub_win",
            "A_early_finish_win": "early_finish_win",
        }
    )
    df_a = df_a.rename(columns={col: col.replace("A_", "") for col in a_cols})

    df_b = df_model_base[
        ["fight_id", "fight_order", "fighter_B", "target_A_win", "is_finish", "fight_round_end", "method_clean"] + b_cols
    ].copy()
    df_b = df_b.rename(columns={"fighter_B": "fighter"})
    df_b["win"] = 1 - df_b["target_A_win"]
    df_b["finish_win"] = ((df_b["win"] == 1) & (df_b["is_finish"] == 1)).astype(int)
    df_b["ko_win"] = (
        (df_b["win"] == 1) & (df_b["method_clean"].str.contains("ko|tko", regex=True, na=False))
    ).astype(int)
    df_b["sub_win"] = ((df_b["win"] == 1) & (df_b["method_clean"].str.contains("submission", na=False))).astype(int)
    df_b["early_finish_win"] = (
        (df_b["win"] == 1) & (df_b["is_finish"] == 1) & (df_b["fight_round_end"] <= 2)
    ).astype(int)
    df_b = df_b.drop(columns=["target_A_win", "is_finish", "fight_round_end", "method_clean"])
    df_b = df_b.rename(columns={col: col.replace("B_", "") for col in b_cols})

    df_a = df_a.loc[:, ~df_a.columns.duplicated()].copy()
    df_b = df_b.loc[:, ~df_b.columns.duplicated()].copy()
    df_long = pd.concat([df_a, df_b], axis=0, ignore_index=True)
    df_long = df_long.sort_values(["fighter", "fight_order"]).reset_index(drop=True)
    return df_long


def add_per_fight_engineered_stats(df_long: pd.DataFrame) -> pd.DataFrame:
    df_long = df_long.copy()
    df_long["sig_str_absorbed_per_round_fight"] = np.where(
        df_long["rounds_fought"] > 0,
        df_long["opp_sig_str_landed"] / df_long["rounds_fought"],
        0,
    )
    df_long["td_def_fight"] = np.where(
        df_long["opp_td_attempted"] > 0,
        1 - (df_long["opp_td_landed"] / df_long["opp_td_attempted"]),
        0,
    )
    df_long["control_share_fight"] = np.where(
        df_long["total_fight_time_seconds"] > 0,
        df_long["ctrl_seconds"] / df_long["total_fight_time_seconds"],
        0,
    )
    df_long["grappling_pressure_fight"] = np.where(
        df_long["rounds_fought"] > 0,
        (df_long["td_landed"] / df_long["rounds_fought"]) * (df_long["ctrl_seconds"] / df_long["rounds_fought"]),
        0,
    )
    df_long["style_index_fight"] = np.where(
        df_long["rounds_fought"] > 0,
        (df_long["sig_str_landed"] / df_long["rounds_fought"])
        - ((df_long["td_landed"] / df_long["rounds_fought"]) + (df_long["ctrl_seconds"] / df_long["rounds_fought"]) / 60.0),
        0,
    )
    df_long["pace_fight"] = np.where(
        df_long["rounds_fought"] > 0,
        (df_long["sig_str_attempted"] + df_long["td_attempted"]) / df_long["rounds_fought"],
        0,
    )
    df_long["damage_ratio_fight"] = np.where(
        df_long["opp_sig_str_landed"] > 0,
        df_long["sig_str_landed"] / df_long["opp_sig_str_landed"],
        np.where(df_long["sig_str_landed"] > 0, 5.0, 1.0),
    )
    df_long["striking_efficiency_fight"] = np.where(
        df_long["sig_str_attempted"] > 0,
        (df_long["sig_str_landed"] ** 2) / df_long["sig_str_attempted"],
        0,
    )
    return df_long


def add_career_and_recent_features(df_long: pd.DataFrame) -> pd.DataFrame:
    df_long = df_long.copy()
    sum_cols = [
        "KD",
        "sig_str_landed",
        "sig_str_attempted",
        "total_str_landed",
        "total_str_attempted",
        "td_landed",
        "td_attempted",
        "sub_att",
        "ctrl_seconds",
        "rounds_fought",
        "opp_sig_str_landed",
        "opp_sig_str_attempted",
        "opp_td_landed",
        "opp_td_attempted",
        "total_fight_time_seconds",
        "finish_win",
        "ko_win",
        "sub_win",
        "early_finish_win",
    ]

    for col in sum_cols:
        df_long[f"career_{col}"] = df_long.groupby("fighter")[col].transform(lambda s: s.cumsum().shift(1)).fillna(0)

    df_long["career_fights"] = df_long.groupby("fighter").cumcount()
    df_long["career_wins"] = df_long.groupby("fighter")["win"].transform(lambda s: s.cumsum().shift(1)).fillna(0)

    streak_values = np.zeros(len(df_long), dtype=int)
    for _, idx in df_long.groupby("fighter").groups.items():
        streak = 0
        for i in list(idx):
            streak_values[i] = streak
            streak = streak + 1 if df_long.loc[i, "win"] == 1 else 0
    df_long["win_streak"] = streak_values

    rolling_cols = sum_cols + ["win"]
    for col in rolling_cols:
        df_long[f"l3_{col}"] = (
            df_long.groupby("fighter")[col].rolling(3, min_periods=1).sum().groupby(level=0).shift(1).reset_index(level=0, drop=True)
        )
    df_long["l3_fights"] = df_long.groupby("fighter").cumcount().clip(upper=3)
    df_long["l3_wins"] = df_long["l3_win"].fillna(0)

    for prefix in ["career", "l3"]:
        fights_col = f"{prefix}_fights"
        rounds_col = f"{prefix}_rounds_fought"
        wins_col = f"{prefix}_wins"

        df_long[f"{prefix}_win_rate"] = safe_divide(df_long[wins_col], df_long[fights_col])
        df_long[f"{prefix}_sig_str_landed_per_round"] = safe_divide(df_long[f"{prefix}_sig_str_landed"], df_long[rounds_col])
        df_long[f"{prefix}_sig_str_attempted_per_round"] = safe_divide(
            df_long[f"{prefix}_sig_str_attempted"], df_long[rounds_col]
        )
        df_long[f"{prefix}_total_str_landed_per_round"] = safe_divide(
            df_long[f"{prefix}_total_str_landed"], df_long[rounds_col]
        )
        df_long[f"{prefix}_total_str_attempted_per_round"] = safe_divide(
            df_long[f"{prefix}_total_str_attempted"], df_long[rounds_col]
        )
        df_long[f"{prefix}_td_landed_per_round"] = safe_divide(df_long[f"{prefix}_td_landed"], df_long[rounds_col])
        df_long[f"{prefix}_td_attempted_per_round"] = safe_divide(df_long[f"{prefix}_td_attempted"], df_long[rounds_col])
        df_long[f"{prefix}_sub_att_per_round"] = safe_divide(df_long[f"{prefix}_sub_att"], df_long[rounds_col])
        df_long[f"{prefix}_ctrl_seconds_per_round"] = safe_divide(df_long[f"{prefix}_ctrl_seconds"], df_long[rounds_col])
        df_long[f"{prefix}_sig_str_absorbed_per_round"] = safe_divide(
            df_long[f"{prefix}_opp_sig_str_landed"], df_long[rounds_col]
        )
        df_long[f"{prefix}_sig_str_acc"] = safe_divide(
            df_long[f"{prefix}_sig_str_landed"], df_long[f"{prefix}_sig_str_attempted"]
        )
        df_long[f"{prefix}_td_acc"] = safe_divide(df_long[f"{prefix}_td_landed"], df_long[f"{prefix}_td_attempted"])
        df_long[f"{prefix}_td_def"] = np.where(
            df_long[f"{prefix}_opp_td_attempted"] > 0,
            1 - (df_long[f"{prefix}_opp_td_landed"] / df_long[f"{prefix}_opp_td_attempted"]),
            0,
        )
        df_long[f"{prefix}_control_share"] = safe_divide(
            df_long[f"{prefix}_ctrl_seconds"], df_long[f"{prefix}_total_fight_time_seconds"]
        )
        df_long[f"{prefix}_grappling_pressure"] = (
            df_long[f"{prefix}_td_landed_per_round"] * df_long[f"{prefix}_ctrl_seconds_per_round"]
        )
        df_long[f"{prefix}_style_index"] = (
            df_long[f"{prefix}_sig_str_landed_per_round"]
            - (df_long[f"{prefix}_td_landed_per_round"] + (df_long[f"{prefix}_ctrl_seconds_per_round"] / 60.0))
        )
        df_long[f"{prefix}_pace"] = safe_divide(
            df_long[f"{prefix}_sig_str_attempted"] + df_long[f"{prefix}_td_attempted"],
            df_long[rounds_col],
        )
        df_long[f"{prefix}_damage_ratio"] = np.where(
            df_long[f"{prefix}_opp_sig_str_landed"] > 0,
            df_long[f"{prefix}_sig_str_landed"] / df_long[f"{prefix}_opp_sig_str_landed"],
            np.where(df_long[f"{prefix}_sig_str_landed"] > 0, 5.0, 1.0),
        )
        df_long[f"{prefix}_striking_efficiency"] = safe_divide(
            df_long[f"{prefix}_sig_str_landed"] ** 2,
            df_long[f"{prefix}_sig_str_attempted"],
        )
        df_long[f"{prefix}_damage_absorbed_ratio"] = safe_divide(
            df_long[f"{prefix}_opp_sig_str_landed"],
            df_long[f"{prefix}_opp_sig_str_attempted"],
        )
        df_long[f"{prefix}_defensive_efficiency"] = 1 - df_long[f"{prefix}_damage_absorbed_ratio"]
        df_long[f"{prefix}_net_damage_per_round"] = (
            df_long[f"{prefix}_sig_str_landed_per_round"] - df_long[f"{prefix}_sig_str_absorbed_per_round"]
        )
        df_long[f"{prefix}_offense_defense_ratio"] = safe_divide(
            df_long[f"{prefix}_sig_str_landed_per_round"],
            df_long[f"{prefix}_sig_str_absorbed_per_round"],
        )

    df_long["career_finish_rate"] = safe_divide(df_long["career_finish_win"], df_long["career_wins"])
    df_long["career_ko_rate"] = safe_divide(df_long["career_ko_win"], df_long["career_wins"])
    df_long["career_sub_rate"] = safe_divide(df_long["career_sub_win"], df_long["career_wins"])
    df_long["career_early_finish_rate"] = safe_divide(df_long["career_early_finish_win"], df_long["career_fights"])
    df_long["career_avg_fight_time_seconds"] = safe_divide(
        df_long["career_total_fight_time_seconds"], df_long["career_fights"]
    )
    df_long = df_long.replace([np.inf, -np.inf], 0).fillna(0)
    return df_long


def build_prefight_model_table(df_fights: pd.DataFrame, df_long: pd.DataFrame) -> pd.DataFrame:
    prefight_cols = ["fight_id", "fight_order", "fighter"] + PREFIGHT_STATE_COLUMNS
    df_prefight = df_long[prefight_cols].copy()

    df_a_prefight = df_prefight.rename(columns={"fighter": "fighter_A"})
    df_a_prefight = df_a_prefight.rename(
        columns={col: f"A_{col}" for col in df_a_prefight.columns if col not in ["fight_id", "fight_order", "fighter_A"]}
    )
    df_b_prefight = df_prefight.rename(columns={"fighter": "fighter_B"})
    df_b_prefight = df_b_prefight.rename(
        columns={col: f"B_{col}" for col in df_b_prefight.columns if col not in ["fight_id", "fight_order", "fighter_B"]}
    )

    df_model = df_fights.merge(df_a_prefight.drop(columns=["fight_order"]), on=["fight_id", "fighter_A"], how="left")
    df_model = df_model.merge(df_b_prefight.drop(columns=["fight_order"]), on=["fight_id", "fighter_B"], how="left")
    df_model = df_model.replace([np.inf, -np.inf], 0)
    return df_model


def dynamic_k(n_fights: float) -> float:
    return float(60 - (40 / (1 + np.exp(-n_fights / 5))))


def add_elo_features(df_model: pd.DataFrame) -> pd.DataFrame:
    df_model = df_model.sort_values("fight_order").reset_index(drop=True).copy()
    elo_dict: defaultdict[str, float] = defaultdict(lambda: 1500.0)
    opponent_elo_history: defaultdict[str, list[float]] = defaultdict(list)

    a_elo_list: list[float] = []
    b_elo_list: list[float] = []
    a_avg_elo_last3_opp_list: list[float] = []
    b_avg_elo_last3_opp_list: list[float] = []

    for _, row in df_model.iterrows():
        fighter_a = row["fighter_A"]
        fighter_b = row["fighter_B"]
        elo_a = float(elo_dict[fighter_a])
        elo_b = float(elo_dict[fighter_b])

        a_elo_list.append(elo_a)
        b_elo_list.append(elo_b)

        a_prev_opp_elos = opponent_elo_history[fighter_a][-3:]
        b_prev_opp_elos = opponent_elo_history[fighter_b][-3:]
        a_avg_elo_last3_opp_list.append(float(np.mean(a_prev_opp_elos)) if a_prev_opp_elos else 1500.0)
        b_avg_elo_last3_opp_list.append(float(np.mean(b_prev_opp_elos)) if b_prev_opp_elos else 1500.0)

        result_a = row["target_A_win"]
        result_b = 1 - result_a
        a_fights = row["A_career_fights"] if pd.notna(row["A_career_fights"]) else 0
        b_fights = row["B_career_fights"] if pd.notna(row["B_career_fights"]) else 0
        k_a = dynamic_k(a_fights)
        k_b = dynamic_k(b_fights)
        p_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        p_b = 1 - p_a

        elo_dict[fighter_a] = elo_a + k_a * (result_a - p_a)
        elo_dict[fighter_b] = elo_b + k_b * (result_b - p_b)
        opponent_elo_history[fighter_a].append(elo_b)
        opponent_elo_history[fighter_b].append(elo_a)

    df_model["A_elo"] = a_elo_list
    df_model["B_elo"] = b_elo_list
    df_model["elo_diff"] = df_model["A_elo"] - df_model["B_elo"]
    df_model["A_avg_elo_last3_opp"] = a_avg_elo_last3_opp_list
    df_model["B_avg_elo_last3_opp"] = b_avg_elo_last3_opp_list
    df_model["avg_elo_last3_opp_diff"] = df_model["A_avg_elo_last3_opp"] - df_model["B_avg_elo_last3_opp"]
    return df_model


def compute_current_elo_state(df_model: pd.DataFrame) -> tuple[dict[str, float], dict[str, list[float]]]:
    df_model = df_model.sort_values("fight_order").reset_index(drop=True)
    elo_dict: defaultdict[str, float] = defaultdict(lambda: 1500.0)
    opponent_elo_history: defaultdict[str, list[float]] = defaultdict(list)

    for _, row in df_model.iterrows():
        fighter_a = row["fighter_A"]
        fighter_b = row["fighter_B"]
        elo_a = float(elo_dict[fighter_a])
        elo_b = float(elo_dict[fighter_b])
        result_a = row["target_A_win"]
        result_b = 1 - result_a
        a_fights = row["A_career_fights"] if pd.notna(row["A_career_fights"]) else 0
        b_fights = row["B_career_fights"] if pd.notna(row["B_career_fights"]) else 0
        k_a = dynamic_k(a_fights)
        k_b = dynamic_k(b_fights)
        p_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        p_b = 1 - p_a
        elo_dict[fighter_a] = elo_a + k_a * (result_a - p_a)
        elo_dict[fighter_b] = elo_b + k_b * (result_b - p_b)
        opponent_elo_history[fighter_a].append(elo_b)
        opponent_elo_history[fighter_b].append(elo_a)

    return dict(elo_dict), {fighter: list(history) for fighter, history in opponent_elo_history.items()}


def _summarize_history_window(history_df: pd.DataFrame, prefix: str) -> dict[str, float]:
    history_df = history_df.copy()
    fights = float(len(history_df))
    wins = float(history_df["win"].sum()) if fights > 0 else 0.0
    totals = {col: float(history_df[col].sum()) if fights > 0 else 0.0 for col in HISTORY_SUM_COLUMNS}
    rounds = totals["rounds_fought"]
    result = {
        f"{prefix}_fights": fights,
        f"{prefix}_wins": wins,
        f"{prefix}_win_rate": float(wins / fights) if fights > 0 else 0.0,
        f"{prefix}_sig_str_landed_per_round": float(totals["sig_str_landed"] / rounds) if rounds > 0 else 0.0,
        f"{prefix}_sig_str_attempted_per_round": float(totals["sig_str_attempted"] / rounds) if rounds > 0 else 0.0,
        f"{prefix}_total_str_landed_per_round": float(totals["total_str_landed"] / rounds) if rounds > 0 else 0.0,
        f"{prefix}_total_str_attempted_per_round": float(totals["total_str_attempted"] / rounds) if rounds > 0 else 0.0,
        f"{prefix}_td_landed_per_round": float(totals["td_landed"] / rounds) if rounds > 0 else 0.0,
        f"{prefix}_td_attempted_per_round": float(totals["td_attempted"] / rounds) if rounds > 0 else 0.0,
        f"{prefix}_sub_att_per_round": float(totals["sub_att"] / rounds) if rounds > 0 else 0.0,
        f"{prefix}_ctrl_seconds_per_round": float(totals["ctrl_seconds"] / rounds) if rounds > 0 else 0.0,
        f"{prefix}_sig_str_acc": float(totals["sig_str_landed"] / totals["sig_str_attempted"]) if totals["sig_str_attempted"] > 0 else 0.0,
        f"{prefix}_td_acc": float(totals["td_landed"] / totals["td_attempted"]) if totals["td_attempted"] > 0 else 0.0,
        f"{prefix}_sig_str_absorbed_per_round": float(totals["opp_sig_str_landed"] / rounds) if rounds > 0 else 0.0,
        f"{prefix}_damage_absorbed_ratio": float(totals["opp_sig_str_landed"] / totals["opp_sig_str_attempted"]) if totals["opp_sig_str_attempted"] > 0 else 0.0,
        f"{prefix}_td_def": 1 - float(totals["opp_td_landed"] / totals["opp_td_attempted"]) if totals["opp_td_attempted"] > 0 else 0.0,
        f"{prefix}_control_share": float(totals["ctrl_seconds"] / totals["total_fight_time_seconds"]) if totals["total_fight_time_seconds"] > 0 else 0.0,
        f"{prefix}_grappling_pressure": 0.0,
        f"{prefix}_style_index": 0.0,
        f"{prefix}_pace": float((totals["sig_str_attempted"] + totals["td_attempted"]) / rounds) if rounds > 0 else 0.0,
        f"{prefix}_damage_ratio": float(totals["sig_str_landed"] / totals["opp_sig_str_landed"]) if totals["opp_sig_str_landed"] > 0 else (5.0 if totals["sig_str_landed"] > 0 else 1.0),
        f"{prefix}_striking_efficiency": float((totals["sig_str_landed"] ** 2) / totals["sig_str_attempted"]) if totals["sig_str_attempted"] > 0 else 0.0,
        f"{prefix}_defensive_efficiency": 0.0,
        f"{prefix}_net_damage_per_round": 0.0,
        f"{prefix}_offense_defense_ratio": 0.0,
    }
    result[f"{prefix}_defensive_efficiency"] = 1 - result[f"{prefix}_damage_absorbed_ratio"]
    result[f"{prefix}_net_damage_per_round"] = (
        result[f"{prefix}_sig_str_landed_per_round"] - result[f"{prefix}_sig_str_absorbed_per_round"]
    )
    result[f"{prefix}_offense_defense_ratio"] = (
        result[f"{prefix}_sig_str_landed_per_round"] / result[f"{prefix}_sig_str_absorbed_per_round"]
        if result[f"{prefix}_sig_str_absorbed_per_round"] > 0
        else 0.0
    )
    result[f"{prefix}_grappling_pressure"] = (
        result[f"{prefix}_td_landed_per_round"] * result[f"{prefix}_ctrl_seconds_per_round"]
    )
    result[f"{prefix}_style_index"] = (
        result[f"{prefix}_sig_str_landed_per_round"]
        - (result[f"{prefix}_td_landed_per_round"] + (result[f"{prefix}_ctrl_seconds_per_round"] / 60.0))
    )
    return result


def _current_win_streak(history_df: pd.DataFrame) -> int:
    streak = 0
    for win in history_df["win"].iloc[::-1]:
        if int(win) == 1:
            streak += 1
        else:
            break
    return streak


def build_current_fighter_states(df_long: pd.DataFrame) -> pd.DataFrame:
    fighter_states: list[dict[str, float | str]] = []
    sorted_long = df_long.sort_values(["fighter", "fight_order"]).reset_index(drop=True)

    for fighter, history_df in sorted_long.groupby("fighter"):
        history_df = history_df.sort_values("fight_order").copy()
        state: dict[str, float | str] = {"fighter": fighter}
        state.update(_summarize_history_window(history_df, prefix="career"))
        state.update(_summarize_history_window(history_df.tail(3), prefix="l3"))
        state["career_finish_rate"] = (
            float(history_df["finish_win"].sum() / history_df["win"].sum()) if history_df["win"].sum() > 0 else 0.0
        )
        state["career_ko_rate"] = (
            float(history_df["ko_win"].sum() / history_df["win"].sum()) if history_df["win"].sum() > 0 else 0.0
        )
        state["career_sub_rate"] = (
            float(history_df["sub_win"].sum() / history_df["win"].sum()) if history_df["win"].sum() > 0 else 0.0
        )
        state["career_early_finish_rate"] = float(history_df["early_finish_win"].sum() / len(history_df)) if len(history_df) > 0 else 0.0
        state["career_avg_fight_time_seconds"] = (
            float(history_df["total_fight_time_seconds"].sum() / len(history_df)) if len(history_df) > 0 else 0.0
        )
        state["win_streak"] = float(_current_win_streak(history_df))
        fighter_states.append(state)

    fighter_states_df = pd.DataFrame(fighter_states)
    if fighter_states_df.empty:
        fighter_states_df = pd.DataFrame(columns=["fighter"] + PREFIGHT_STATE_COLUMNS)
    for col in PREFIGHT_STATE_COLUMNS:
        if col not in fighter_states_df.columns:
            fighter_states_df[col] = 0.0
    fighter_states_df = fighter_states_df[["fighter"] + PREFIGHT_STATE_COLUMNS].copy()
    fighter_states_df = fighter_states_df.replace([np.inf, -np.inf], 0).fillna(0)
    return fighter_states_df


def build_future_matchup_features(
    matchups_df: pd.DataFrame,
    df_long: pd.DataFrame,
    df_model: pd.DataFrame,
) -> pd.DataFrame:
    fighter_states = build_current_fighter_states(df_long)
    elo_state, opponent_elo_history = compute_current_elo_state(df_model)
    matchups_df = matchups_df.copy()
    matchups_df["fighter_A_display"] = matchups_df["fighter_A"]
    matchups_df["fighter_B_display"] = matchups_df["fighter_B"]
    matchups_df["fighter_A"] = matchups_df["fighter_A"].map(normalize_fighter_name)
    matchups_df["fighter_B"] = matchups_df["fighter_B"].map(normalize_fighter_name)
    if "fight_id" not in matchups_df.columns:
        matchups_df["fight_id"] = matchups_df.apply(
            lambda row: f"future_{row.name + 1} | {row['fighter_A']} vs. {row['fighter_B']}",
            axis=1,
        )
    matchups_df["fight_order"] = range(len(df_model), len(df_model) + len(matchups_df))

    df_a_state = fighter_states.rename(columns={"fighter": "fighter_A"})
    df_a_state = df_a_state.rename(columns={col: f"A_{col}" for col in PREFIGHT_STATE_COLUMNS})
    df_b_state = fighter_states.rename(columns={"fighter": "fighter_B"})
    df_b_state = df_b_state.rename(columns={col: f"B_{col}" for col in PREFIGHT_STATE_COLUMNS})

    future_df = matchups_df.merge(df_a_state, on="fighter_A", how="left")
    future_df = future_df.merge(df_b_state, on="fighter_B", how="left")

    required_state_cols = [f"A_{col}" for col in PREFIGHT_STATE_COLUMNS] + [f"B_{col}" for col in PREFIGHT_STATE_COLUMNS]
    missing_state_cols = [col for col in required_state_cols if col not in future_df.columns]
    if missing_state_cols:
        future_df = pd.concat(
            [future_df, pd.DataFrame(0.0, index=future_df.index, columns=missing_state_cols)],
            axis=1,
        )
    numeric_cols = [col for col in required_state_cols if col in future_df.columns]
    future_df[numeric_cols] = future_df[numeric_cols].fillna(0)
    future_df = future_df.copy()
    future_df = future_df.assign(
        A_elo=future_df["fighter_A"].map(lambda fighter: float(elo_state.get(fighter, 1500.0))),
        B_elo=future_df["fighter_B"].map(lambda fighter: float(elo_state.get(fighter, 1500.0))),
        A_avg_elo_last3_opp=future_df["fighter_A"].map(
            lambda fighter: float(np.mean(opponent_elo_history.get(fighter, [])[-3:])) if opponent_elo_history.get(fighter) else 1500.0
        ),
        B_avg_elo_last3_opp=future_df["fighter_B"].map(
            lambda fighter: float(np.mean(opponent_elo_history.get(fighter, [])[-3:])) if opponent_elo_history.get(fighter) else 1500.0
        ),
    )

    future_df = create_differential_features(future_df)
    future_df = future_df.replace([np.inf, -np.inf], 0).fillna(0)
    return future_df


def create_differential_features(df_model: pd.DataFrame) -> pd.DataFrame:
    """Build A-minus-B matchup features from pre-fight fighter states."""
    df_model = df_model.copy()
    eps = 1e-6
    diff_pairs = [
        ("career_win_rate_diff", "A_career_win_rate", "B_career_win_rate"),
        ("l3_win_rate_diff", "A_l3_win_rate", "B_l3_win_rate"),
        ("career_fights_diff", "A_career_fights", "B_career_fights"),
        ("career_wins_diff", "A_career_wins", "B_career_wins"),
        ("win_streak_diff", "A_win_streak", "B_win_streak"),
        ("career_sig_str_landed_per_round_diff", "A_career_sig_str_landed_per_round", "B_career_sig_str_landed_per_round"),
        ("l3_sig_str_landed_per_round_diff", "A_l3_sig_str_landed_per_round", "B_l3_sig_str_landed_per_round"),
        ("career_sig_str_acc_diff", "A_career_sig_str_acc", "B_career_sig_str_acc"),
        ("l3_sig_str_acc_diff", "A_l3_sig_str_acc", "B_l3_sig_str_acc"),
        ("career_striking_efficiency_diff", "A_career_striking_efficiency", "B_career_striking_efficiency"),
        ("l3_striking_efficiency_diff", "A_l3_striking_efficiency", "B_l3_striking_efficiency"),
        ("career_sig_str_absorbed_per_round_diff", "A_career_sig_str_absorbed_per_round", "B_career_sig_str_absorbed_per_round"),
        ("l3_sig_str_absorbed_per_round_diff", "A_l3_sig_str_absorbed_per_round", "B_l3_sig_str_absorbed_per_round"),
        ("career_damage_ratio_diff", "A_career_damage_ratio", "B_career_damage_ratio"),
        ("l3_damage_ratio_diff", "A_l3_damage_ratio", "B_l3_damage_ratio"),
        ("career_td_landed_per_round_diff", "A_career_td_landed_per_round", "B_career_td_landed_per_round"),
        ("l3_td_landed_per_round_diff", "A_l3_td_landed_per_round", "B_l3_td_landed_per_round"),
        ("career_td_acc_diff", "A_career_td_acc", "B_career_td_acc"),
        ("l3_td_acc_diff", "A_l3_td_acc", "B_l3_td_acc"),
        ("career_td_def_diff", "A_career_td_def", "B_career_td_def"),
        ("l3_td_def_diff", "A_l3_td_def", "B_l3_td_def"),
        ("career_control_share_diff", "A_career_control_share", "B_career_control_share"),
        ("l3_control_share_diff", "A_l3_control_share", "B_l3_control_share"),
        ("career_grappling_pressure_diff", "A_career_grappling_pressure", "B_career_grappling_pressure"),
        ("l3_grappling_pressure_diff", "A_l3_grappling_pressure", "B_l3_grappling_pressure"),
        ("career_style_index_diff", "A_career_style_index", "B_career_style_index"),
        ("l3_style_index_diff", "A_l3_style_index", "B_l3_style_index"),
        ("career_pace_diff", "A_career_pace", "B_career_pace"),
        ("l3_pace_diff", "A_l3_pace", "B_l3_pace"),
        ("career_finish_rate_diff", "A_career_finish_rate", "B_career_finish_rate"),
        ("career_ko_rate_diff", "A_career_ko_rate", "B_career_ko_rate"),
        ("career_sub_rate_diff", "A_career_sub_rate", "B_career_sub_rate"),
        ("career_early_finish_rate_diff", "A_career_early_finish_rate", "B_career_early_finish_rate"),
        ("career_avg_fight_time_diff", "A_career_avg_fight_time_seconds", "B_career_avg_fight_time_seconds"),
    ]
    for new_col, a_col, b_col in diff_pairs:
        df_model[new_col] = df_model[a_col] - df_model[b_col]

    df_model["career_damage_efficiency_ratio_diff"] = np.log(
        (df_model["A_career_sig_str_landed_per_round"] + eps) / (df_model["A_career_sig_str_absorbed_per_round"] + eps)
    ) - np.log((df_model["B_career_sig_str_landed_per_round"] + eps) / (df_model["B_career_sig_str_absorbed_per_round"] + eps))
    df_model["l3_damage_efficiency_ratio_diff"] = np.log(
        (df_model["A_l3_sig_str_landed_per_round"] + eps) / (df_model["A_l3_sig_str_absorbed_per_round"] + eps)
    ) - np.log((df_model["B_l3_sig_str_landed_per_round"] + eps) / (df_model["B_l3_sig_str_absorbed_per_round"] + eps))

    df_model["career_net_striking_diff"] = (
        (df_model["A_career_sig_str_landed_per_round"] - df_model["A_career_sig_str_absorbed_per_round"])
        - (df_model["B_career_sig_str_landed_per_round"] - df_model["B_career_sig_str_absorbed_per_round"])
    )
    df_model["l3_net_striking_diff"] = (
        (df_model["A_l3_sig_str_landed_per_round"] - df_model["A_l3_sig_str_absorbed_per_round"])
        - (df_model["B_l3_sig_str_landed_per_round"] - df_model["B_l3_sig_str_absorbed_per_round"])
    )

    df_model["career_td_path_diff"] = (
        (df_model["A_career_td_acc"] * (1 - df_model["B_career_td_def"]))
        - (df_model["B_career_td_acc"] * (1 - df_model["A_career_td_def"]))
    )
    df_model["l3_td_path_diff"] = (
        (df_model["A_l3_td_acc"] * (1 - df_model["B_l3_td_def"]))
        - (df_model["B_l3_td_acc"] * (1 - df_model["A_l3_td_def"]))
    )

    df_model["career_opponent_adjusted_damage_ratio_diff"] = (
        df_model["A_career_damage_ratio"] * (df_model["A_avg_elo_last3_opp"] / 1500.0)
        - df_model["B_career_damage_ratio"] * (df_model["B_avg_elo_last3_opp"] / 1500.0)
    )
    df_model["career_damage_absorbed_ratio_diff"] = (
        df_model["A_career_damage_absorbed_ratio"] - df_model["B_career_damage_absorbed_ratio"]
    )
    df_model["l3_damage_absorbed_ratio_diff"] = df_model["A_l3_damage_absorbed_ratio"] - df_model["B_l3_damage_absorbed_ratio"]
    df_model["career_defensive_efficiency_diff"] = (
        df_model["A_career_defensive_efficiency"] - df_model["B_career_defensive_efficiency"]
    )
    df_model["l3_defensive_efficiency_diff"] = (
        df_model["A_l3_defensive_efficiency"] - df_model["B_l3_defensive_efficiency"]
    )
    df_model["career_net_damage_per_round_diff"] = (
        df_model["A_career_net_damage_per_round"] - df_model["B_career_net_damage_per_round"]
    )
    df_model["career_offense_defense_ratio_diff"] = np.log(
        (df_model["A_career_offense_defense_ratio"] + eps) / (df_model["B_career_offense_defense_ratio"] + eps)
    )
    df_model["l3_offense_defense_ratio_diff"] = np.log(
        (df_model["A_l3_offense_defense_ratio"] + eps) / (df_model["B_l3_offense_defense_ratio"] + eps)
    )
    df_model["career_adjusted_sig_strike_diff"] = (
        (df_model["A_career_sig_str_landed_per_round"] + eps) / (df_model["B_career_sig_str_absorbed_per_round"] + eps)
        - (df_model["B_career_sig_str_landed_per_round"] + eps) / (df_model["A_career_sig_str_absorbed_per_round"] + eps)
    )
    df_model["l3_adjusted_sig_strike_diff"] = (
        (df_model["A_l3_sig_str_landed_per_round"] + eps) / (df_model["B_l3_sig_str_absorbed_per_round"] + eps)
        - (df_model["B_l3_sig_str_landed_per_round"] + eps) / (df_model["A_l3_sig_str_absorbed_per_round"] + eps)
    )
    df_model["career_adjusted_grappling_diff"] = (
        (df_model["A_career_td_landed_per_round"] + eps) / (1 - df_model["B_career_td_def"] + eps)
        - (df_model["B_career_td_landed_per_round"] + eps) / (1 - df_model["A_career_td_def"] + eps)
    )
    df_model["l3_adjusted_grappling_diff"] = (
        (df_model["A_l3_td_landed_per_round"] + eps) / (1 - df_model["B_l3_td_def"] + eps)
        - (df_model["B_l3_td_landed_per_round"] + eps) / (1 - df_model["A_l3_td_def"] + eps)
    )
    df_model["experience_advantage_ratio_log"] = np.log(
        (df_model["A_career_fights"] + 1.0) / (df_model["B_career_fights"] + 1.0)
    )
    df_model["win_advantage_ratio_log"] = np.log((df_model["A_career_wins"] + 1.0) / (df_model["B_career_wins"] + 1.0))

    df_model["elo_diff"] = df_model["A_elo"] - df_model["B_elo"]
    df_model["avg_elo_last3_opp_diff"] = df_model["A_avg_elo_last3_opp"] - df_model["B_avg_elo_last3_opp"]
    df_model = df_model.replace([np.inf, -np.inf], 0).fillna(0)
    return df_model


def finalize_model_dataset(df_model: pd.DataFrame) -> pd.DataFrame:
    df_final = df_model[FINAL_FEATURE_COLUMNS].copy()
    df_final = df_final.replace([np.inf, -np.inf], 0).fillna(0)
    return df_final


def run_leakage_checks(df_long: pd.DataFrame, df_final: pd.DataFrame) -> dict[str, Any]:
    first_fights = df_long.groupby("fighter").head(1).copy()
    career_cols = [
        "career_fights",
        "career_wins",
        "career_sig_str_landed",
        "career_sig_str_attempted",
        "career_td_landed",
        "career_td_attempted",
        "career_ctrl_seconds",
        "career_finish_win",
    ]
    l3_cols = [
        "l3_fights",
        "l3_wins",
        "l3_sig_str_landed",
        "l3_sig_str_attempted",
        "l3_td_landed",
        "l3_td_attempted",
        "l3_ctrl_seconds",
    ]
    return {
        "final_nulls": int(df_final.isna().sum().sum()),
        "duplicate_fight_id": bool(df_final["fight_id"].duplicated().any()),
        "first_fight_career_nonzero": {col: int((first_fights[col] != 0).sum()) for col in career_cols},
        "first_fight_l3_nonzero": {col: int((first_fights[col] != 0).sum()) for col in l3_cols},
    }


def build_feature_dataset(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_path: str | Path | None = None,
    save: bool = False,
    return_intermediates: bool = False,
    include_physical: bool = True,
    physical_dataset_path: str | Path = DEFAULT_PHYSICAL_ODDS_PATH,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, pd.DataFrame | dict[str, Any]]]:
    """Build the final modeling dataset from raw data and optional physical enrichments."""
    raw_data = load_raw_data(data_dir=data_dir)
    df_stats = clean_stats_table(raw_data["stats"])
    df_fight_level = aggregate_fight_level_stats(df_stats)
    df_fights = clean_results_table(raw_data["results"])
    df_model_base = build_model_base(df_fights, df_fight_level)
    df_model_base = add_result_based_finish_flags(df_model_base)
    df_long = build_long_fighter_history(df_model_base)
    df_long = add_per_fight_engineered_stats(df_long)
    df_long = add_career_and_recent_features(df_long)
    df_model = build_prefight_model_table(df_fights, df_long)
    df_model = add_elo_features(df_model)
    df_model = create_differential_features(df_model)
    df_final = finalize_model_dataset(df_model)
    if include_physical:
        df_final = add_physical_difference_features(df_final, physical_dataset_path=physical_dataset_path)
    leakage_checks = run_leakage_checks(df_long, df_final)

    if save:
        destination = Path(output_path) if output_path else DEFAULT_OUTPUT_PATH
        destination.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(destination, index=False)

    if return_intermediates:
        artifacts: dict[str, pd.DataFrame | dict[str, Any]] = {
            "results": raw_data["results"],
            "fighters": raw_data["fighters"],
            "stats_clean": df_stats,
            "fight_level": df_fight_level,
            "fights": df_fights,
            "model_base": df_model_base,
            "fighter_long": df_long,
            "model_table": df_model,
            "leakage_checks": leakage_checks,
        }
        return df_final, artifacts

    return df_final


if __name__ == "__main__":
    final_df, artifacts = build_feature_dataset(save=True, return_intermediates=True)
    print(f"Built dataset with shape: {final_df.shape}")
    print(f"Saved to: {DEFAULT_OUTPUT_PATH}")
    print(f"Nulls in final: {artifacts['leakage_checks']['final_nulls']}")
    print(f"Duplicate fight_id: {artifacts['leakage_checks']['duplicate_fight_id']}")
