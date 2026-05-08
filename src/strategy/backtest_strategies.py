from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BETTING_READY_PATH = PROJECT_ROOT / "outputs" / "strategy" / "betting_ready.csv"
BET_LOG_PATH = PROJECT_ROOT / "outputs" / "strategy" / "strategy_bet_log.csv"
EQUITY_CURVES_DIR = PROJECT_ROOT / "outputs" / "strategy" / "equity_curves"

FLAT_STAKE = 100.0


def american_profit(odds: float, stake: float = FLAT_STAKE) -> float:
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds > 0:
        return float(stake * (odds / 100.0))
    return float(stake * (100.0 / abs(odds)))


def strategy_definitions() -> list[dict[str, object]]:
    return [
        {
            "strategy_name": "v1_base",
            "edge_threshold": 0.0,
            "confidence_threshold": None,
            "require_agreement": False,
            "max_positive_odds": 150.0,
            "stake": FLAT_STAKE,
        },
        {
            "strategy_name": "s1_edge_gt_0p03",
            "edge_threshold": 0.03,
            "confidence_threshold": None,
            "require_agreement": False,
            "max_positive_odds": 150.0,
            "stake": FLAT_STAKE,
        },
        {
            "strategy_name": "s2_edge_gt_0p05",
            "edge_threshold": 0.05,
            "confidence_threshold": None,
            "require_agreement": False,
            "max_positive_odds": 150.0,
            "stake": FLAT_STAKE,
        },
        {
            "strategy_name": "s3_edge_gt_0p05_p_ge_0p60",
            "edge_threshold": 0.05,
            "confidence_threshold": 0.60,
            "require_agreement": False,
            "max_positive_odds": 150.0,
            "stake": FLAT_STAKE,
        },
        {
            "strategy_name": "s4_edge_gt_0p05_p_ge_0p70",
            "edge_threshold": 0.05,
            "confidence_threshold": 0.70,
            "require_agreement": False,
            "max_positive_odds": 150.0,
            "stake": FLAT_STAKE,
        },
        {
            "strategy_name": "s5_agreement_edge_gt_0p05",
            "edge_threshold": 0.05,
            "confidence_threshold": None,
            "require_agreement": True,
            "max_positive_odds": 150.0,
            "stake": FLAT_STAKE,
        },
    ]


def apply_strategy(df: pd.DataFrame, strategy: dict[str, object]) -> pd.DataFrame:
    result = df.copy()
    edge_threshold = float(strategy["edge_threshold"])
    confidence_threshold = strategy["confidence_threshold"]
    require_agreement = bool(strategy["require_agreement"])
    max_positive_odds = strategy["max_positive_odds"]
    stake = float(strategy["stake"])

    eligible = result["has_valid_odds"].fillna(False)
    eligible &= result["edge_A"].fillna(-np.inf) > edge_threshold
    if confidence_threshold is not None:
        eligible &= result["p_model_A"].fillna(-np.inf) >= float(confidence_threshold)
    if require_agreement:
        eligible &= result["model_agreement"].fillna(False)
    if max_positive_odds is not None:
        eligible &= ~(result["odds_A"].gt(float(max_positive_odds)))

    profit_if_win = result["odds_A"].map(lambda odds: american_profit(odds, stake=stake))
    bet_placed = eligible.astype(int)
    won_bet = bet_placed.eq(1) & result["actual_outcome"].eq(1)
    lost_bet = bet_placed.eq(1) & result["actual_outcome"].eq(0)

    result["strategy_name"] = str(strategy["strategy_name"])
    result["edge_threshold"] = edge_threshold
    result["confidence_threshold"] = confidence_threshold
    result["require_agreement"] = require_agreement
    result["max_positive_odds"] = max_positive_odds
    result["stake"] = stake
    result["bet_placed"] = bet_placed
    result["won_bet"] = won_bet.astype(int)
    result["lost_bet"] = lost_bet.astype(int)
    result["potential_profit_if_win"] = profit_if_win
    result["bet_profit"] = np.where(
        won_bet,
        profit_if_win,
        np.where(lost_bet, -stake, 0.0),
    )
    result["cumulative_profit"] = result["bet_profit"].cumsum()
    result["equity_peak"] = result["cumulative_profit"].cummax()
    result["drawdown"] = result["cumulative_profit"] - result["equity_peak"]
    result["bet_number"] = result["bet_placed"].cumsum()
    return result


def run_backtests(
    betting_ready_path: Path = BETTING_READY_PATH,
    bet_log_path: Path = BET_LOG_PATH,
    equity_curves_dir: Path = EQUITY_CURVES_DIR,
) -> pd.DataFrame:
    betting_df = pd.read_csv(betting_ready_path)
    betting_df["event_date"] = pd.to_datetime(betting_df["event_date"], errors="coerce")
    betting_df = betting_df.sort_values(["fight_order", "fight_id"]).reset_index(drop=True)

    strategy_frames: list[pd.DataFrame] = []
    equity_curves_dir.mkdir(parents=True, exist_ok=True)
    for strategy in strategy_definitions():
        strategy_df = apply_strategy(betting_df, strategy)
        strategy_df.to_csv(equity_curves_dir / f"{strategy['strategy_name']}.csv", index=False)
        strategy_frames.append(strategy_df)

    all_bets_df = pd.concat(strategy_frames, ignore_index=True)
    bet_log_path.parent.mkdir(parents=True, exist_ok=True)
    all_bets_df.to_csv(bet_log_path, index=False)
    return all_bets_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 betting strategy backtests.")
    parser.add_argument("--betting-ready", type=Path, default=BETTING_READY_PATH)
    parser.add_argument("--bet-log-output", type=Path, default=BET_LOG_PATH)
    parser.add_argument("--equity-curves-dir", type=Path, default=EQUITY_CURVES_DIR)
    args = parser.parse_args()

    all_bets_df = run_backtests(
        betting_ready_path=args.betting_ready,
        bet_log_path=args.bet_log_output,
        equity_curves_dir=args.equity_curves_dir,
    )
    summary = (
        all_bets_df.groupby("strategy_name", as_index=False)
        .agg(
            bets=("bet_placed", "sum"),
            profit=("bet_profit", "sum"),
        )
        .sort_values("strategy_name")
    )
    print(f"Saved bet log: {args.bet_log_output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
