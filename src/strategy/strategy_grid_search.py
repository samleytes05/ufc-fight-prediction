from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BETTING_READY_PATH = PROJECT_ROOT / "outputs" / "strategy" / "betting_ready.csv"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "strategy" / "reports"
GRID_RESULTS_PATH = REPORTS_DIR / "strategy_grid_results.csv"
GRID_TOP_PATH = REPORTS_DIR / "strategy_grid_top.csv"
HEATMAP_CSV_PATH = REPORTS_DIR / "edge_conf_heatmap.csv"
HEATMAP_PNG_PATH = REPORTS_DIR / "edge_conf_heatmap.png"
FLAT_STAKE = 100.0

EDGE_THRESHOLDS = [0.02, 0.03, 0.04, 0.05, 0.06]
CONFIDENCE_THRESHOLDS = [0.55, 0.60, 0.65, 0.70]
MIN_SAMPLE_SIZE = 100

MPL_CONFIG_DIR = PROJECT_ROOT / "outputs" / "strategy" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def american_profit(odds: float, stake: float = FLAT_STAKE) -> float:
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds > 0:
        return float(stake * (odds / 100.0))
    return float(stake * (100.0 / abs(odds)))


def load_betting_ready(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["p_model_A", "implied_prob_A", "edge_A", "odds_A", "fight_order"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in betting_ready.csv: {missing}")

    target_column = "target_A_win" if "target_A_win" in df.columns else "actual_outcome"
    if target_column not in df.columns:
        raise ValueError("betting_ready.csv must include either target_A_win or actual_outcome")

    df = df.copy()
    df["target_A_win"] = pd.to_numeric(df[target_column], errors="coerce")
    df["fight_order"] = pd.to_numeric(df["fight_order"], errors="coerce")
    df["p_model_A"] = pd.to_numeric(df["p_model_A"], errors="coerce")
    df["implied_prob_A"] = pd.to_numeric(df["implied_prob_A"], errors="coerce")
    df["edge_A"] = pd.to_numeric(df["edge_A"], errors="coerce")
    df["odds_A"] = pd.to_numeric(df["odds_A"], errors="coerce")
    if "has_valid_odds" in df.columns:
        df["has_valid_odds"] = df["has_valid_odds"].fillna(False).astype(bool)
    else:
        df["has_valid_odds"] = df["odds_A"].notna() & df["implied_prob_A"].notna()
    df = df[df["has_valid_odds"]].copy()
    df = df.sort_values(["fight_order", "fight_id"]).reset_index(drop=True)
    return df


def segment_filter(df: pd.DataFrame, segment: str) -> pd.DataFrame:
    if segment == "all":
        return df
    if segment == "favorite":
        return df[df["odds_A"] < 0].copy()
    if segment == "underdog":
        return df[df["odds_A"] > 0].copy()
    raise ValueError(f"Unsupported segment: {segment}")


def evaluate_subset(df: pd.DataFrame, edge_threshold: float, confidence_threshold: float, segment: str) -> dict[str, object]:
    filtered = segment_filter(df, segment)
    filtered = filtered[
        (filtered["edge_A"] > edge_threshold)
        & (filtered["p_model_A"] >= confidence_threshold)
    ].copy()

    if filtered.empty:
        return {
            "edge_threshold": edge_threshold,
            "confidence_threshold": confidence_threshold,
            "segment": segment,
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_profit": 0.0,
            "roi": 0.0,
        }

    payout_if_win = filtered["odds_A"].map(american_profit)
    profits = np.where(filtered["target_A_win"] == 1, payout_if_win, -FLAT_STAKE)
    bets = int(len(filtered))
    wins = int((filtered["target_A_win"] == 1).sum())
    losses = int((filtered["target_A_win"] == 0).sum())
    total_profit = float(np.nansum(profits))
    total_staked = bets * FLAT_STAKE

    return {
        "edge_threshold": edge_threshold,
        "confidence_threshold": confidence_threshold,
        "segment": segment,
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "win_rate": float(wins / bets) if bets > 0 else 0.0,
        "total_profit": total_profit,
        "roi": float(total_profit / total_staked) if total_staked > 0 else 0.0,
    }


def run_grid_search(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for edge_threshold in EDGE_THRESHOLDS:
        for confidence_threshold in CONFIDENCE_THRESHOLDS:
            for segment in ["all", "favorite", "underdog"]:
                rows.append(
                    evaluate_subset(
                        df=df,
                        edge_threshold=edge_threshold,
                        confidence_threshold=confidence_threshold,
                        segment=segment,
                    )
                )
    results_df = pd.DataFrame(rows)
    results_df["roi"] = results_df["roi"].fillna(0.0)
    results_df["win_rate"] = results_df["win_rate"].fillna(0.0)
    return results_df.sort_values(
        ["segment", "roi", "bets", "edge_threshold", "confidence_threshold"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)


def build_top_strategies(results_df: pd.DataFrame) -> pd.DataFrame:
    top_df = results_df[results_df["bets"] >= MIN_SAMPLE_SIZE].copy()
    return top_df.sort_values(
        ["roi", "bets"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_heatmap_table(results_df: pd.DataFrame) -> pd.DataFrame:
    all_segment_df = results_df[results_df["segment"] == "all"].copy()
    heatmap_df = all_segment_df.pivot(
        index="edge_threshold",
        columns="confidence_threshold",
        values="roi",
    )
    heatmap_df = heatmap_df.reindex(index=EDGE_THRESHOLDS, columns=CONFIDENCE_THRESHOLDS)
    heatmap_df = heatmap_df.fillna(0.0)
    return heatmap_df


def save_heatmap_plot(heatmap_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(heatmap_df.to_numpy(), aspect="auto")
    ax.set_xticks(np.arange(len(heatmap_df.columns)))
    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_xticklabels([f"{value:.2f}" for value in heatmap_df.columns])
    ax.set_yticklabels([f"{value:.2f}" for value in heatmap_df.index])
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Edge Threshold")
    ax.set_title("ROI Heatmap")
    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def run_strategy_grid_search(
    betting_ready_path: Path = BETTING_READY_PATH,
    reports_dir: Path = REPORTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    df = load_betting_ready(betting_ready_path)
    results_df = run_grid_search(df)
    top_df = build_top_strategies(results_df)
    heatmap_df = build_heatmap_table(results_df)

    results_df.to_csv(GRID_RESULTS_PATH, index=False)
    top_df.to_csv(GRID_TOP_PATH, index=False)
    heatmap_df.to_csv(HEATMAP_CSV_PATH)
    save_heatmap_plot(heatmap_df, HEATMAP_PNG_PATH)
    return results_df, top_df, heatmap_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 edge/confidence grid search.")
    parser.add_argument("--betting-ready", type=Path, default=BETTING_READY_PATH)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    results_df, top_df, heatmap_df = run_strategy_grid_search(
        betting_ready_path=args.betting_ready,
        reports_dir=args.reports_dir,
    )

    print("Top grid strategies")
    print(top_df.head(20).to_string(index=False))
    print()
    print("All-bets ROI heatmap")
    print(heatmap_df.to_string())


if __name__ == "__main__":
    main()
