from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BETTING_READY_PATH = PROJECT_ROOT / "outputs" / "strategy" / "betting_ready.csv"
BET_LOG_PATH = PROJECT_ROOT / "outputs" / "strategy" / "strategy_bet_log.csv"
STRATEGY_RESULTS_PATH = PROJECT_ROOT / "outputs" / "strategy" / "strategy_results.csv"
FOLD_RESULTS_PATH = PROJECT_ROOT / "outputs" / "strategy" / "fold_results.csv"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "strategy" / "reports"
PHASE3_RESULTS_MD = PROJECT_ROOT / "docs" / "strategy" / "BETTING_STRATEGY_RESULTS.md"


def max_drawdown_from_curve(curve: pd.Series) -> float:
    if curve.empty:
        return 0.0
    running_peak = curve.cummax()
    drawdown = curve - running_peak
    return float(drawdown.min())


def compute_strategy_results(bet_log_df: pd.DataFrame, betting_ready_df: pd.DataFrame) -> pd.DataFrame:
    coverage = {
        "total_fights": int(len(betting_ready_df)),
        "fights_with_odds": int(betting_ready_df["has_valid_odds"].sum()),
        "odds_coverage_pct": float(betting_ready_df["has_valid_odds"].mean()) if len(betting_ready_df) else 0.0,
    }

    rows: list[dict[str, object]] = []
    for strategy_name, strategy_df in bet_log_df.groupby("strategy_name", sort=True):
        bets_df = strategy_df[strategy_df["bet_placed"] == 1].copy()
        total_stake = float(bets_df["stake"].sum())
        total_profit = float(bets_df["bet_profit"].sum())
        roi = float(total_profit / total_stake) if total_stake > 0 else 0.0
        max_drawdown = max_drawdown_from_curve(strategy_df["cumulative_profit"])
        fold_roi = (
            bets_df.groupby("fold", as_index=False)
            .agg(
                stake=("stake", "sum"),
                profit=("bet_profit", "sum"),
            )
            .assign(roi=lambda frame: np.where(frame["stake"] > 0, frame["profit"] / frame["stake"], np.nan))
        )
        rows.append(
            {
                "strategy_name": strategy_name,
                "edge_threshold": bets_df["edge_threshold"].iloc[0] if not bets_df.empty else strategy_df["edge_threshold"].iloc[0],
                "confidence_threshold": strategy_df["confidence_threshold"].iloc[0],
                "require_agreement": bool(strategy_df["require_agreement"].iloc[0]),
                "max_positive_odds": strategy_df["max_positive_odds"].iloc[0],
                "stake": float(strategy_df["stake"].iloc[0]),
                **coverage,
                "total_bets": int(bets_df["bet_placed"].sum()),
                "wins": int(bets_df["won_bet"].sum()),
                "losses": int(bets_df["lost_bet"].sum()),
                "win_rate": float(bets_df["won_bet"].mean()) if not bets_df.empty else 0.0,
                "total_profit": total_profit,
                "total_stake": total_stake,
                "roi": roi,
                "average_edge": float(bets_df["edge_A"].mean()) if not bets_df.empty else 0.0,
                "max_drawdown": max_drawdown,
                "roi_variance_across_folds": float(fold_roi["roi"].var(ddof=0)) if not fold_roi.empty else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("roi", ascending=False).reset_index(drop=True)


def compute_fold_results(bet_log_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (strategy_name, fold), fold_df in bet_log_df.groupby(["strategy_name", "fold"], sort=True):
        bets_df = fold_df[fold_df["bet_placed"] == 1].copy()
        total_stake = float(bets_df["stake"].sum())
        total_profit = float(bets_df["bet_profit"].sum())
        rows.append(
            {
                "strategy_name": strategy_name,
                "fold": int(fold),
                "bets": int(bets_df["bet_placed"].sum()),
                "wins": int(bets_df["won_bet"].sum()),
                "losses": int(bets_df["lost_bet"].sum()),
                "win_rate": float(bets_df["won_bet"].mean()) if not bets_df.empty else 0.0,
                "total_profit": total_profit,
                "total_stake": total_stake,
                "roi": float(total_profit / total_stake) if total_stake > 0 else 0.0,
                "average_edge": float(bets_df["edge_A"].mean()) if not bets_df.empty else 0.0,
                "max_drawdown": max_drawdown_from_curve(fold_df["cumulative_profit"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["strategy_name", "fold"]).reset_index(drop=True)


def bucket_summary(
    bet_log_df: pd.DataFrame,
    bucket_column: str,
    output_name: str,
    reports_dir: Path,
) -> None:
    placed_df = bet_log_df[bet_log_df["bet_placed"] == 1].copy()
    if placed_df.empty:
        pd.DataFrame().to_csv(reports_dir / output_name, index=False)
        return
    summary_df = (
        placed_df.groupby(["strategy_name", bucket_column], dropna=False, as_index=False)
        .agg(
            bets=("bet_placed", "sum"),
            wins=("won_bet", "sum"),
            total_profit=("bet_profit", "sum"),
            total_stake=("stake", "sum"),
            average_edge=("edge_A", "mean"),
        )
        .assign(
            win_rate=lambda frame: np.where(frame["bets"] > 0, frame["wins"] / frame["bets"], np.nan),
            roi=lambda frame: np.where(frame["total_stake"] > 0, frame["total_profit"] / frame["total_stake"], np.nan),
        )
    )
    summary_df.to_csv(reports_dir / output_name, index=False)


def create_diagnostics(bet_log_df: pd.DataFrame, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_df = bet_log_df.copy()
    diagnostics_df["edge_bucket"] = pd.cut(
        diagnostics_df["edge_A"],
        bins=[-np.inf, 0.0, 0.03, 0.05, 0.10, np.inf],
        labels=["<=0", "0-0.03", "0.03-0.05", "0.05-0.10", "0.10+"],
    )
    diagnostics_df["confidence_bucket"] = pd.cut(
        diagnostics_df["p_model_A"],
        bins=[0.0, 0.55, 0.60, 0.70, 1.0],
        labels=["<=0.55", "0.55-0.60", "0.60-0.70", "0.70+"],
        include_lowest=True,
    )
    diagnostics_df["price_bucket"] = np.select(
        [
            diagnostics_df["odds_A"] < 0,
            diagnostics_df["odds_A"].between(0, 150, inclusive="both"),
            diagnostics_df["odds_A"] > 150,
        ],
        ["favorite", "short_underdog", "long_underdog"],
        default="missing",
    )
    diagnostics_df["agreement_bucket"] = np.where(diagnostics_df["model_agreement"], "agreement", "disagreement")

    bucket_summary(diagnostics_df, "edge_bucket", "edge_bucket_summary.csv", reports_dir)
    bucket_summary(diagnostics_df, "confidence_bucket", "confidence_bucket_summary.csv", reports_dir)
    bucket_summary(diagnostics_df, "price_bucket", "price_bucket_summary.csv", reports_dir)
    bucket_summary(diagnostics_df, "agreement_bucket", "agreement_summary.csv", reports_dir)


def next_experiment_number(markdown_path: Path) -> int:
    if not markdown_path.exists():
        return 1
    existing_text = markdown_path.read_text(encoding="utf-8")
    return existing_text.count("## Experiment ") + 1


def append_results_log(
    strategy_results_df: pd.DataFrame,
    betting_ready_df: pd.DataFrame,
    markdown_path: Path,
) -> None:
    experiment_number = next_experiment_number(markdown_path)
    top_row = strategy_results_df.iloc[0]
    header = "# Phase 3 - Betting Strategy Results\n\n" if not markdown_path.exists() else ""
    coverage_pct = float(betting_ready_df["has_valid_odds"].mean()) if len(betting_ready_df) else 0.0

    lines = [
        header,
        f"## Experiment {experiment_number}: Initial Phase 3 Strategy Suite",
        f"Date: {date.today().isoformat()}",
        "",
        "### Strategy Definition",
        f"- edge threshold: strategy batch from 0.00 to 0.05+",
        "- confidence threshold: none, 0.60, and 0.70 variants",
        "- filters: +150 A-side underdog cap, optional RF agreement filter",
        "- stake: flat 100",
        "",
        "### Data Coverage",
        f"- total fights: {len(betting_ready_df)}",
        f"- fights with odds: {int(betting_ready_df['has_valid_odds'].sum())}",
        f"- % coverage: {coverage_pct:.1%}",
        "",
        "### Results",
        f"- ROI: {top_row['roi']:.4f}",
        f"- total profit: {top_row['total_profit']:.2f}",
        f"- total bets: {int(top_row['total_bets'])}",
        f"- win rate: {top_row['win_rate']:.4f}",
        f"- max drawdown: {top_row['max_drawdown']:.2f}",
        "",
        "### Observations",
        f"- what worked: highest ROI strategy in this batch was `{top_row['strategy_name']}`",
        "- what failed: lower-edge or low-coverage variants may still be noisy and need fold review",
        "- anomalies: historical odds coverage is limited to the portion matched through the legacy odds file",
        "",
        "### Next Adjustments",
        "- test edge bucket refinements and confidence filters against fold stability",
        "- consider fractional Kelly only after the flat-stake baseline is accepted",
        "",
    ]
    with markdown_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def evaluate(
    betting_ready_path: Path = BETTING_READY_PATH,
    bet_log_path: Path = BET_LOG_PATH,
    strategy_results_path: Path = STRATEGY_RESULTS_PATH,
    fold_results_path: Path = FOLD_RESULTS_PATH,
    reports_dir: Path = REPORTS_DIR,
    phase3_results_md: Path = PHASE3_RESULTS_MD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    betting_ready_df = pd.read_csv(betting_ready_path)
    bet_log_df = pd.read_csv(bet_log_path)
    strategy_results_df = compute_strategy_results(bet_log_df, betting_ready_df)
    fold_results_df = compute_fold_results(bet_log_df)

    strategy_results_df.to_csv(strategy_results_path, index=False)
    fold_results_df.to_csv(fold_results_path, index=False)
    create_diagnostics(bet_log_df, reports_dir)
    append_results_log(strategy_results_df, betting_ready_df, phase3_results_md)
    return strategy_results_df, fold_results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 3 strategy backtest outputs.")
    parser.add_argument("--betting-ready", type=Path, default=BETTING_READY_PATH)
    parser.add_argument("--bet-log", type=Path, default=BET_LOG_PATH)
    parser.add_argument("--strategy-results", type=Path, default=STRATEGY_RESULTS_PATH)
    parser.add_argument("--fold-results", type=Path, default=FOLD_RESULTS_PATH)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--results-markdown", type=Path, default=PHASE3_RESULTS_MD)
    args = parser.parse_args()

    strategy_results_df, fold_results_df = evaluate(
        betting_ready_path=args.betting_ready,
        bet_log_path=args.bet_log,
        strategy_results_path=args.strategy_results,
        fold_results_path=args.fold_results,
        reports_dir=args.reports_dir,
        phase3_results_md=args.results_markdown,
    )
    print(f"Saved strategy summary: {args.strategy_results}")
    print(f"Saved fold summary: {args.fold_results}")
    print(strategy_results_df.to_string(index=False))
    print()
    print("Fold ROI preview")
    print(fold_results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
