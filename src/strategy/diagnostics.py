from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BETTING_READY_PATH = PROJECT_ROOT / "outputs" / "strategy" / "betting_ready.csv"
BET_LOG_PATH = PROJECT_ROOT / "outputs" / "strategy" / "strategy_bet_log.csv"
FOLD_RESULTS_PATH = PROJECT_ROOT / "outputs" / "strategy" / "fold_results.csv"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "strategy" / "reports"
EDGE_BUCKET_ANALYSIS_PATH = REPORTS_DIR / "edge_bucket_analysis.csv"
FOLD_STABILITY_SUMMARY_PATH = REPORTS_DIR / "fold_stability_summary.csv"
FOLD_LEVEL_ROI_PATH = REPORTS_DIR / "fold_level_roi.csv"
EDGE_BUCKET_PLOT_PATH = REPORTS_DIR / "edge_bucket_plot.png"
FLAT_STAKE = 100.0


EDGE_BUCKET_BINS = [0.0, 0.02, 0.05, 0.08, np.inf]
EDGE_BUCKET_LABELS = ["0-2%", "2-5%", "5-8%", "8%+"]

MPL_CONFIG_DIR = PROJECT_ROOT / "outputs" / "strategy" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_edge_analysis_source(betting_ready_path: Path, bet_log_path: Path) -> pd.DataFrame:
    betting_ready_df = pd.read_csv(betting_ready_path)
    if "bet_flag" in betting_ready_df.columns:
        edge_df = betting_ready_df.copy()
        edge_df["bet_flag"] = pd.to_numeric(edge_df["bet_flag"], errors="coerce").fillna(0).astype(int)
        edge_df["strategy_name"] = edge_df.get("strategy_name", "unknown")
        edge_df["target_A_win"] = pd.to_numeric(
            edge_df.get("target_A_win", edge_df.get("actual_outcome")),
            errors="coerce",
        )
        edge_df["profit"] = pd.to_numeric(edge_df.get("profit", edge_df.get("bet_profit", 0.0)), errors="coerce").fillna(0.0)
        edge_df["stake"] = pd.to_numeric(edge_df.get("stake", FLAT_STAKE), errors="coerce").fillna(FLAT_STAKE)
        return edge_df

    bet_log_df = pd.read_csv(bet_log_path)
    bet_log_df["bet_flag"] = pd.to_numeric(bet_log_df.get("bet_placed", 0), errors="coerce").fillna(0).astype(int)
    bet_log_df["target_A_win"] = pd.to_numeric(
        bet_log_df.get("target_A_win", bet_log_df.get("actual_outcome")),
        errors="coerce",
    )
    bet_log_df["profit"] = pd.to_numeric(bet_log_df.get("bet_profit", 0.0), errors="coerce").fillna(0.0)
    bet_log_df["stake"] = pd.to_numeric(bet_log_df.get("stake", FLAT_STAKE), errors="coerce").fillna(FLAT_STAKE)
    return bet_log_df


def build_edge_bucket_analysis(edge_source_df: pd.DataFrame) -> pd.DataFrame:
    placed_df = edge_source_df[edge_source_df["bet_flag"] == 1].copy()
    placed_df = placed_df[placed_df["edge_A"].notna()].copy()
    placed_df = placed_df[placed_df["edge_A"] >= 0].copy()
    placed_df["edge_bucket"] = pd.cut(
        placed_df["edge_A"],
        bins=EDGE_BUCKET_BINS,
        labels=EDGE_BUCKET_LABELS,
        include_lowest=True,
        right=False,
        ordered=True,
    )
    placed_df = placed_df[placed_df["edge_bucket"].notna()].copy()

    strategy_bucket_df = (
        placed_df.groupby(["strategy_name", "edge_bucket"], observed=False, as_index=False)
        .agg(
            number_of_bets=("bet_flag", "sum"),
            win_rate=("target_A_win", "mean"),
            average_edge=("edge_A", "mean"),
            total_profit=("profit", "sum"),
            total_staked=("stake", "sum"),
        )
        .assign(roi=lambda frame: np.where(frame["total_staked"] > 0, frame["total_profit"] / frame["total_staked"], 0.0))
    )

    overall_df = (
        placed_df.groupby("edge_bucket", observed=False, as_index=False)
        .agg(
            number_of_bets=("bet_flag", "sum"),
            win_rate=("target_A_win", "mean"),
            average_edge=("edge_A", "mean"),
            total_profit=("profit", "sum"),
            total_staked=("stake", "sum"),
        )
        .assign(
            strategy_name="overall",
            roi=lambda frame: np.where(frame["total_staked"] > 0, frame["total_profit"] / frame["total_staked"], 0.0),
        )
    )

    combined_df = pd.concat([overall_df, strategy_bucket_df], ignore_index=True)
    combined_df["edge_bucket"] = pd.Categorical(combined_df["edge_bucket"], categories=EDGE_BUCKET_LABELS, ordered=True)
    combined_df["win_rate"] = combined_df["win_rate"].fillna(0.0)
    combined_df["average_edge"] = combined_df["average_edge"].fillna(0.0)
    combined_df["total_profit"] = combined_df["total_profit"].fillna(0.0)
    combined_df["total_staked"] = combined_df["total_staked"].fillna(0.0)
    combined_df["roi"] = combined_df["roi"].fillna(0.0)
    combined_df = combined_df.sort_values(["strategy_name", "edge_bucket"]).reset_index(drop=True)
    return combined_df


def build_fold_stability_analysis(fold_results_path: Path, stake: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_df = pd.read_csv(fold_results_path)
    strategy_column = "strategy_name" if "strategy_name" in fold_df.columns else "strategy"
    profit_column = "total_profit" if "total_profit" in fold_df.columns else "profit"
    bets_column = "bets"

    fold_level_roi_df = fold_df.copy()
    fold_level_roi_df["recomputed_roi"] = np.where(
        pd.to_numeric(fold_level_roi_df[bets_column], errors="coerce").fillna(0) > 0,
        pd.to_numeric(fold_level_roi_df[profit_column], errors="coerce").fillna(0.0)
        / (pd.to_numeric(fold_level_roi_df[bets_column], errors="coerce").fillna(0) * stake),
        0.0,
    )
    fold_level_roi_df["recomputed_roi"] = fold_level_roi_df["recomputed_roi"].fillna(0.0)

    fold_stability_df = (
        fold_level_roi_df.groupby(strategy_column, as_index=False)
        .agg(
            mean_roi=("recomputed_roi", "mean"),
            roi_std=("recomputed_roi", "std"),
            min_roi=("recomputed_roi", "min"),
            max_roi=("recomputed_roi", "max"),
            total_bets=(bets_column, "sum"),
        )
        .rename(columns={strategy_column: "strategy_name"})
        .sort_values(["mean_roi", "total_bets"], ascending=[False, False])
        .reset_index(drop=True)
    )
    fold_stability_df["roi_std"] = fold_stability_df["roi_std"].fillna(0.0)
    return fold_level_roi_df, fold_stability_df


def save_edge_bucket_plot(edge_bucket_df: pd.DataFrame, output_path: Path) -> None:
    overall_df = edge_bucket_df[edge_bucket_df["strategy_name"] == "overall"].copy()
    overall_df = overall_df.sort_values("edge_bucket")
    if overall_df.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(overall_df["edge_bucket"].astype(str), overall_df["roi"], marker="o")
    plt.xlabel("Edge Bucket")
    plt.ylabel("ROI")
    plt.title("ROI by Edge Bucket")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_diagnostics(
    betting_ready_path: Path = BETTING_READY_PATH,
    bet_log_path: Path = BET_LOG_PATH,
    fold_results_path: Path = FOLD_RESULTS_PATH,
    reports_dir: Path = REPORTS_DIR,
    stake: float = FLAT_STAKE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reports_dir.mkdir(parents=True, exist_ok=True)

    edge_source_df = load_edge_analysis_source(betting_ready_path, bet_log_path)
    edge_bucket_df = build_edge_bucket_analysis(edge_source_df)
    edge_bucket_df.to_csv(EDGE_BUCKET_ANALYSIS_PATH, index=False)

    fold_level_roi_df, fold_stability_df = build_fold_stability_analysis(fold_results_path, stake=stake)
    fold_level_roi_df.to_csv(FOLD_LEVEL_ROI_PATH, index=False)
    fold_stability_df.to_csv(FOLD_STABILITY_SUMMARY_PATH, index=False)

    save_edge_bucket_plot(edge_bucket_df, EDGE_BUCKET_PLOT_PATH)
    return edge_bucket_df, fold_level_roi_df, fold_stability_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 edge bucket and fold stability diagnostics.")
    parser.add_argument("--betting-ready", type=Path, default=BETTING_READY_PATH)
    parser.add_argument("--bet-log", type=Path, default=BET_LOG_PATH)
    parser.add_argument("--fold-results", type=Path, default=FOLD_RESULTS_PATH)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--stake", type=float, default=FLAT_STAKE)
    args = parser.parse_args()

    edge_bucket_df, _, fold_stability_df = run_diagnostics(
        betting_ready_path=args.betting_ready,
        bet_log_path=args.bet_log,
        fold_results_path=args.fold_results,
        reports_dir=args.reports_dir,
        stake=args.stake,
    )

    print("Edge bucket analysis")
    print(
        edge_bucket_df.sort_values(["strategy_name", "roi"], ascending=[True, False]).to_string(index=False)
    )
    print()
    print("Fold stability ranking")
    print(fold_stability_df.to_string(index=False))


if __name__ == "__main__":
    main()
