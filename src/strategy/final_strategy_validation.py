from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BETTING_READY_PATH = PROJECT_ROOT / "outputs" / "strategy" / "betting_ready.csv"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "strategy" / "reports"
EQUITY_CURVES_DIR = PROJECT_ROOT / "outputs" / "strategy" / "equity_curves"
PHASE3_RESULTS_MD = PROJECT_ROOT / "docs" / "strategy" / "BETTING_STRATEGY_RESULTS.md"

FINAL_SUMMARY_PATH = REPORTS_DIR / "final_strategy_summary.csv"
FINAL_FOLD_BREAKDOWN_PATH = REPORTS_DIR / "final_strategy_fold_breakdown.csv"
FINAL_DRAWDOWN_PATH = REPORTS_DIR / "final_strategy_drawdown.csv"
FINAL_EDGE_MONOTONICITY_PATH = REPORTS_DIR / "final_strategy_edge_monotonicity.csv"
FINAL_TIME_SPLIT_PATH = REPORTS_DIR / "final_strategy_time_split.csv"

FLAT_STAKE = 100.0
STARTING_BANKROLL = 10_000.0
MIN_ROI = 0.08
MIN_BETS = 200
MIN_PROFITABLE_FOLDS_PCT = 0.65
MAX_DRAWDOWN_TO_PROFIT_RATIO = 1.0
MAX_TOP5_PROFIT_SHARE = 0.50


def american_profit(odds: float, stake: float = FLAT_STAKE) -> float:
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds > 0:
        return float(stake * (odds / 100.0))
    return float(stake * (100.0 / abs(odds)))


def load_betting_ready(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = ["p_model_A", "implied_prob_A", "edge_A", "odds_A", "fight_order"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in betting_ready.csv: {missing}")

    target_column = "target_A_win" if "target_A_win" in df.columns else "actual_outcome"
    if target_column not in df.columns:
        raise ValueError("betting_ready.csv must include either target_A_win or actual_outcome")

    df = df.copy()
    df["target_A_win"] = pd.to_numeric(df[target_column], errors="coerce")
    df["fight_order"] = pd.to_numeric(df["fight_order"], errors="coerce")
    df["p_model_A"] = pd.to_numeric(df["p_model_A"], errors="coerce")
    df["edge_A"] = pd.to_numeric(df["edge_A"], errors="coerce")
    df["odds_A"] = pd.to_numeric(df["odds_A"], errors="coerce")
    if "fold" in df.columns:
        df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
    else:
        df["fold"] = np.nan
    if "has_valid_odds" in df.columns:
        df["has_valid_odds"] = df["has_valid_odds"].fillna(False).astype(bool)
    else:
        df["has_valid_odds"] = df["odds_A"].notna()
    return df[df["has_valid_odds"]].sort_values(["fight_order", "fight_id"]).reset_index(drop=True)


def build_folds(df: pd.DataFrame, fold_count: int = 5) -> pd.DataFrame:
    if df["fold"].notna().any():
        return df.copy()

    result = df.copy()
    result["fold"] = pd.qcut(result["fight_order"].rank(method="first"), q=fold_count, labels=False, duplicates="drop") + 1
    return result


def strategy_definitions() -> list[dict[str, object]]:
    return [
        {
            "strategy_name": "V2_Core",
            "edge_threshold": 0.04,
            "confidence_threshold": 0.65,
            "favorites_only": False,
        },
        {
            "strategy_name": "V2_Fav",
            "edge_threshold": 0.04,
            "confidence_threshold": 0.65,
            "favorites_only": True,
        },
    ]


def apply_strategy(df: pd.DataFrame, strategy: dict[str, object]) -> pd.DataFrame:
    result = df.copy()
    eligible = (result["edge_A"] > float(strategy["edge_threshold"])) & (
        result["p_model_A"] >= float(strategy["confidence_threshold"])
    )
    if bool(strategy["favorites_only"]):
        eligible &= result["odds_A"] < 0

    result["strategy_name"] = str(strategy["strategy_name"])
    result["bet_placed"] = eligible.astype(int)
    result["payout_if_win"] = result["odds_A"].map(american_profit)
    result["profit"] = np.where(
        result["bet_placed"].eq(1) & result["target_A_win"].eq(1),
        result["payout_if_win"],
        np.where(result["bet_placed"].eq(1) & result["target_A_win"].eq(0), -FLAT_STAKE, 0.0),
    )
    result["bet_return"] = np.where(result["bet_placed"].eq(1), result["profit"] / FLAT_STAKE, np.nan)
    result["cumulative_profit"] = result["profit"].cumsum()
    result["bankroll"] = STARTING_BANKROLL + result["cumulative_profit"]
    result["running_peak_bankroll"] = result["bankroll"].cummax()
    result["drawdown"] = result["bankroll"] - result["running_peak_bankroll"]
    result["bet_number"] = result["bet_placed"].cumsum()

    losing_streak = 0
    losing_streaks: list[int] = []
    for row in result.itertuples():
        if int(row.bet_placed) == 1 and float(row.profit) < 0:
            losing_streak += 1
        elif int(row.bet_placed) == 1:
            losing_streak = 0
        losing_streaks.append(losing_streak)
    result["losing_streak"] = losing_streaks
    return result


def fold_breakdown(strategy_df: pd.DataFrame) -> pd.DataFrame:
    placed_df = strategy_df[strategy_df["bet_placed"] == 1].copy()
    grouped = (
        placed_df.groupby("fold", as_index=False)
        .agg(
            bets=("bet_placed", "sum"),
            wins=("target_A_win", "sum"),
            total_profit=("profit", "sum"),
        )
        .assign(
            losses=lambda frame: frame["bets"] - frame["wins"],
            win_rate=lambda frame: np.where(frame["bets"] > 0, frame["wins"] / frame["bets"], 0.0),
            roi=lambda frame: np.where(frame["bets"] > 0, frame["total_profit"] / (frame["bets"] * FLAT_STAKE), 0.0),
        )
    )
    grouped["strategy_name"] = strategy_df["strategy_name"].iloc[0]
    return grouped[["strategy_name", "fold", "bets", "wins", "losses", "win_rate", "total_profit", "roi"]]


def edge_monotonicity(strategy_df: pd.DataFrame) -> pd.DataFrame:
    placed_df = strategy_df[strategy_df["bet_placed"] == 1].copy()
    placed_df["edge_bucket"] = pd.cut(
        placed_df["edge_A"],
        bins=[0.04, 0.05, 0.06, np.inf],
        labels=["4-5%", "5-6%", "6%+"],
        include_lowest=True,
        right=False,
        ordered=True,
    )
    grouped = (
        placed_df.groupby("edge_bucket", observed=False, as_index=False)
        .agg(
            bets=("bet_placed", "sum"),
            total_profit=("profit", "sum"),
        )
        .assign(roi=lambda frame: np.where(frame["bets"] > 0, frame["total_profit"] / (frame["bets"] * FLAT_STAKE), 0.0))
    )
    grouped["strategy_name"] = strategy_df["strategy_name"].iloc[0]
    return grouped[["strategy_name", "edge_bucket", "bets", "total_profit", "roi"]]


def time_split(strategy_df: pd.DataFrame) -> pd.DataFrame:
    placed_df = strategy_df[strategy_df["bet_placed"] == 1].copy()
    if placed_df.empty:
        return pd.DataFrame(
            [{"strategy_name": strategy_df["strategy_name"].iloc[0], "time_split": label, "bets": 0, "roi": 0.0, "win_rate": 0.0} for label in ["early_50", "recent_50"]]
        )

    midpoint = strategy_df["fight_order"].median()
    segments = {
        "early_50": placed_df[placed_df["fight_order"] <= midpoint],
        "recent_50": placed_df[placed_df["fight_order"] > midpoint],
    }
    rows: list[dict[str, object]] = []
    for label, segment_df in segments.items():
        bets = int(len(segment_df))
        profit = float(segment_df["profit"].sum()) if bets else 0.0
        rows.append(
            {
                "strategy_name": strategy_df["strategy_name"].iloc[0],
                "time_split": label,
                "bets": bets,
                "roi": float(profit / (bets * FLAT_STAKE)) if bets > 0 else 0.0,
                "win_rate": float(segment_df["target_A_win"].mean()) if bets > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def summary_row(strategy_df: pd.DataFrame, fold_df: pd.DataFrame) -> dict[str, object]:
    placed_df = strategy_df[strategy_df["bet_placed"] == 1].copy()
    total_bets = int(len(placed_df))
    total_profit = float(placed_df["profit"].sum()) if total_bets else 0.0
    roi = float(total_profit / (total_bets * FLAT_STAKE)) if total_bets > 0 else 0.0
    win_rate = float(placed_df["target_A_win"].mean()) if total_bets > 0 else 0.0
    max_drawdown = float(abs(strategy_df["drawdown"].min())) if not strategy_df.empty else 0.0
    negative_drawdowns = strategy_df.loc[strategy_df["drawdown"] < 0, "drawdown"].abs()
    average_drawdown = float(negative_drawdowns.mean()) if not negative_drawdowns.empty else 0.0
    longest_losing_streak = int(strategy_df["losing_streak"].max()) if not strategy_df.empty else 0
    std_dev_returns = float(placed_df["bet_return"].std(ddof=0)) if total_bets > 0 else 0.0
    profitable_folds_pct = float((fold_df["roi"] > 0).mean()) if not fold_df.empty else 0.0
    top5_profit_share = (
        float(placed_df["profit"].nlargest(5).sum() / total_profit)
        if total_bets > 0 and total_profit > 0
        else np.inf if total_profit <= 0 else 0.0
    )
    drawdown_to_profit_ratio = float(max_drawdown / total_profit) if total_profit > 0 else np.inf

    accept = (
        roi >= MIN_ROI
        and total_bets >= MIN_BETS
        and profitable_folds_pct >= MIN_PROFITABLE_FOLDS_PCT
        and drawdown_to_profit_ratio <= MAX_DRAWDOWN_TO_PROFIT_RATIO
        and top5_profit_share <= MAX_TOP5_PROFIT_SHARE
    )

    return {
        "strategy_name": strategy_df["strategy_name"].iloc[0],
        "total_bets": total_bets,
        "total_profit": total_profit,
        "roi": roi,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "average_drawdown": average_drawdown,
        "longest_losing_streak": longest_losing_streak,
        "std_dev_returns": std_dev_returns,
        "profitable_folds_pct": profitable_folds_pct,
        "drawdown_to_profit_ratio": drawdown_to_profit_ratio,
        "top5_profit_share": top5_profit_share,
        "decision": "ACCEPT" if accept else "REJECT",
    }


def next_experiment_number(markdown_path: Path) -> int:
    if not markdown_path.exists():
        return 1
    return markdown_path.read_text(encoding="utf-8").count("## Experiment ") + 1


def append_log(summary_df: pd.DataFrame, markdown_path: Path) -> None:
    experiment_number = next_experiment_number(markdown_path)
    lines = [
        f"## Experiment {experiment_number}: Final Strategy Validation",
        f"Date: {date.today().isoformat()}",
        "",
    ]
    for row in summary_df.itertuples():
        lines.extend(
            [
                f"- strategy tested: `{row.strategy_name}`",
                f"- ROI: {row.roi:.4f}",
                f"- bets: {int(row.total_bets)}",
                f"- drawdown: {row.max_drawdown:.2f}",
                f"- fold stability: {row.profitable_folds_pct:.1%} profitable folds",
                f"- decision: {row.decision}",
                "",
            ]
        )
    with markdown_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def run_validation(
    betting_ready_path: Path = BETTING_READY_PATH,
    reports_dir: Path = REPORTS_DIR,
    equity_curves_dir: Path = EQUITY_CURVES_DIR,
    markdown_path: Path = PHASE3_RESULTS_MD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    equity_curves_dir.mkdir(parents=True, exist_ok=True)

    betting_df = build_folds(load_betting_ready(betting_ready_path))
    summary_rows: list[dict[str, object]] = []
    fold_frames: list[pd.DataFrame] = []
    drawdown_frames: list[pd.DataFrame] = []
    monotonicity_frames: list[pd.DataFrame] = []
    time_split_frames: list[pd.DataFrame] = []

    for strategy in strategy_definitions():
        strategy_df = apply_strategy(betting_df, strategy)
        fold_df = fold_breakdown(strategy_df)
        summary_rows.append(summary_row(strategy_df, fold_df))
        fold_frames.append(fold_df)
        monotonicity_frames.append(edge_monotonicity(strategy_df))
        time_split_frames.append(time_split(strategy_df))

        equity_name = "V2_Core.csv" if strategy["strategy_name"] == "V2_Core" else "V2_Fav.csv"
        strategy_df.to_csv(equity_curves_dir / equity_name, index=False)
        drawdown_frames.append(
            strategy_df[
                [
                    "strategy_name",
                    "fight_id",
                    "fight_order",
                    "bet_placed",
                    "profit",
                    "cumulative_profit",
                    "bankroll",
                    "running_peak_bankroll",
                    "drawdown",
                    "losing_streak",
                ]
            ].copy()
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["decision", "roi", "total_bets"], ascending=[True, False, False]).reset_index(drop=True)
    fold_breakdown_df = pd.concat(fold_frames, ignore_index=True)
    drawdown_df = pd.concat(drawdown_frames, ignore_index=True)
    monotonicity_df = pd.concat(monotonicity_frames, ignore_index=True)
    time_split_df = pd.concat(time_split_frames, ignore_index=True)

    summary_df.to_csv(FINAL_SUMMARY_PATH, index=False)
    fold_breakdown_df.to_csv(FINAL_FOLD_BREAKDOWN_PATH, index=False)
    drawdown_df.to_csv(FINAL_DRAWDOWN_PATH, index=False)
    monotonicity_df.to_csv(FINAL_EDGE_MONOTONICITY_PATH, index=False)
    time_split_df.to_csv(FINAL_TIME_SPLIT_PATH, index=False)
    append_log(summary_df, markdown_path)

    return summary_df, fold_breakdown_df, drawdown_df, monotonicity_df, time_split_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final Phase 3 strategy validation.")
    parser.add_argument("--betting-ready", type=Path, default=BETTING_READY_PATH)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--equity-curves-dir", type=Path, default=EQUITY_CURVES_DIR)
    parser.add_argument("--results-markdown", type=Path, default=PHASE3_RESULTS_MD)
    args = parser.parse_args()

    summary_df, fold_breakdown_df, _, monotonicity_df, time_split_df = run_validation(
        betting_ready_path=args.betting_ready,
        reports_dir=args.reports_dir,
        equity_curves_dir=args.equity_curves_dir,
        markdown_path=args.results_markdown,
    )
    print("Final strategy summary")
    print(summary_df.to_string(index=False))
    print()
    print("Fold breakdown preview")
    print(fold_breakdown_df.to_string(index=False))
    print()
    print("Edge monotonicity")
    print(monotonicity_df.to_string(index=False))
    print()
    print("Time split")
    print(time_split_df.to_string(index=False))


if __name__ == "__main__":
    main()
