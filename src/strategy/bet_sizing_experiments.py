from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BETTING_READY_PATH = PROJECT_ROOT / "outputs" / "strategy" / "betting_ready.csv"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "strategy" / "reports"
EQUITY_CURVES_DIR = PROJECT_ROOT / "outputs" / "strategy" / "equity_curves"
SUMMARY_PATH = REPORTS_DIR / "bet_sizing_summary.csv"

INITIAL_BANKROLL = 10_000.0
V2_CORE_EDGE_THRESHOLD = 0.04
V2_CORE_CONFIDENCE_THRESHOLD = 0.65
PERCENT_BANKROLL_FRACTION = 0.01
KELLY_CAP_FRACTION = 0.02
MAX_ACCEPTABLE_DRAWDOWN_PCT = 0.40
MAX_ACCEPTABLE_STAKE_PCT = 0.05


def american_profit(odds: float, stake: float) -> float:
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds > 0:
        return float(stake * (odds / 100.0))
    return float(stake * (100.0 / abs(odds)))


def payout_multiple(odds: float) -> float:
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds > 0:
        return float(odds / 100.0)
    return float(100.0 / abs(odds))


def load_v2_core_bets(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = ["fight_order", "fight_id", "p_model_A", "edge_A", "odds_A"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in betting_ready.csv: {missing}")

    target_column = "target_A_win" if "target_A_win" in df.columns else "actual_outcome"
    if target_column not in df.columns:
        raise ValueError("betting_ready.csv must include either target_A_win or actual_outcome")

    result = df.copy()
    result["fight_order"] = pd.to_numeric(result["fight_order"], errors="coerce")
    result["p_model_A"] = pd.to_numeric(result["p_model_A"], errors="coerce")
    result["edge_A"] = pd.to_numeric(result["edge_A"], errors="coerce")
    result["odds_A"] = pd.to_numeric(result["odds_A"], errors="coerce")
    result["target_A_win"] = pd.to_numeric(result[target_column], errors="coerce")
    if "has_valid_odds" in result.columns:
        result["has_valid_odds"] = result["has_valid_odds"].fillna(False).astype(bool)
    else:
        result["has_valid_odds"] = result["odds_A"].notna()

    eligible = (
        result["has_valid_odds"]
        & result["edge_A"].gt(V2_CORE_EDGE_THRESHOLD)
        & result["p_model_A"].ge(V2_CORE_CONFIDENCE_THRESHOLD)
        & result["target_A_win"].isin([0, 1])
    )
    result = result.loc[eligible].copy()
    result["payout_multiple"] = result["odds_A"].map(payout_multiple)
    return result.sort_values(["fight_order", "fight_id"]).reset_index(drop=True)


def strategy_definitions() -> list[dict[str, object]]:
    return [
        {
            "strategy_name": "flat_100",
            "strategy_label": "Flat $100",
            "sizing_type": "flat",
        },
        {
            "strategy_name": "pct_bankroll_1pct",
            "strategy_label": "1% Bankroll",
            "sizing_type": "percent_bankroll",
            "fraction": PERCENT_BANKROLL_FRACTION,
        },
        {
            "strategy_name": "kelly_full",
            "strategy_label": "Full Kelly",
            "sizing_type": "kelly",
            "kelly_multiplier": 1.0,
        },
        {
            "strategy_name": "kelly_half",
            "strategy_label": "Half Kelly",
            "sizing_type": "kelly",
            "kelly_multiplier": 0.5,
        },
        {
            "strategy_name": "kelly_quarter",
            "strategy_label": "Quarter Kelly",
            "sizing_type": "kelly",
            "kelly_multiplier": 0.25,
        },
        {
            "strategy_name": "kelly_capped_2pct",
            "strategy_label": "Kelly Capped 2%",
            "sizing_type": "capped_kelly",
            "kelly_multiplier": 1.0,
            "cap_fraction": KELLY_CAP_FRACTION,
        },
    ]


def kelly_fraction(probability: float, odds: float) -> float:
    b = payout_multiple(odds)
    if pd.isna(probability) or pd.isna(b) or b <= 0:
        return 0.0
    q = 1.0 - float(probability)
    fraction = ((b * float(probability)) - q) / b
    return max(float(fraction), 0.0)


def stake_for_strategy(
    bankroll: float,
    probability: float,
    odds: float,
    strategy: dict[str, object],
) -> tuple[float, float]:
    sizing_type = str(strategy["sizing_type"])
    raw_kelly = kelly_fraction(probability, odds)

    if bankroll <= 0:
        return 0.0, raw_kelly

    if sizing_type == "flat":
        stake = 100.0
    elif sizing_type == "percent_bankroll":
        stake = bankroll * float(strategy["fraction"])
    elif sizing_type == "kelly":
        stake = bankroll * raw_kelly * float(strategy["kelly_multiplier"])
    elif sizing_type == "capped_kelly":
        capped_fraction = min(raw_kelly * float(strategy["kelly_multiplier"]), float(strategy["cap_fraction"]))
        stake = bankroll * capped_fraction
    else:
        raise ValueError(f"Unsupported sizing type: {sizing_type}")

    stake = max(float(stake), 0.0)
    stake = min(stake, bankroll)
    return stake, raw_kelly


def simulate_strategy(bets_df: pd.DataFrame, strategy: dict[str, object]) -> tuple[pd.DataFrame, dict[str, object]]:
    bankroll = INITIAL_BANKROLL
    running_peak = INITIAL_BANKROLL
    losing_streak = 0
    rows: list[dict[str, object]] = []

    for bet_number, row in enumerate(bets_df.itertuples(index=False), start=1):
        bankroll_before = bankroll
        stake, raw_kelly = stake_for_strategy(
            bankroll=bankroll_before,
            probability=float(row.p_model_A),
            odds=float(row.odds_A),
            strategy=strategy,
        )
        placed_bet = stake > 0
        payout_if_win = american_profit(float(row.odds_A), stake) if placed_bet else 0.0

        if placed_bet and int(row.target_A_win) == 1:
            profit = float(payout_if_win)
            bankroll = bankroll_before + profit
            losing_streak = 0
        elif placed_bet:
            profit = -stake
            bankroll = bankroll_before + profit
            losing_streak += 1
        else:
            profit = 0.0
            bankroll = bankroll_before

        running_peak = max(running_peak, bankroll)
        drawdown = bankroll - running_peak
        simple_return = (profit / stake) if stake > 0 else np.nan
        bankroll_return = (profit / bankroll_before) if bankroll_before > 0 else np.nan

        rows.append(
            {
                "strategy_name": strategy["strategy_name"],
                "strategy_label": strategy["strategy_label"],
                "fight_id": row.fight_id,
                "fight_order": row.fight_order,
                "event_name": getattr(row, "event_name", None),
                "event_date": getattr(row, "event_date", None),
                "bout": getattr(row, "bout", None),
                "fighter_A": getattr(row, "fighter_A", None),
                "fighter_B": getattr(row, "fighter_B", None),
                "fold": getattr(row, "fold", np.nan),
                "odds_A": float(row.odds_A),
                "p_model_A": float(row.p_model_A),
                "edge_A": float(row.edge_A),
                "target_A_win": int(row.target_A_win),
                "bet_number": bet_number,
                "bankroll_before": bankroll_before,
                "stake": stake,
                "stake_pct_bankroll": (stake / bankroll_before) if bankroll_before > 0 else 0.0,
                "raw_kelly_fraction": raw_kelly,
                "payout_if_win": payout_if_win,
                "profit": profit,
                "simple_return": simple_return,
                "bankroll_return": bankroll_return,
                "bankroll_after": bankroll,
                "running_peak_bankroll": running_peak,
                "drawdown": drawdown,
                "losing_streak": losing_streak,
                "bet_placed": int(placed_bet),
            }
        )

    equity_df = pd.DataFrame(rows)
    summary = summarize_strategy(equity_df)
    return equity_df, summary


def summarize_strategy(equity_df: pd.DataFrame) -> dict[str, object]:
    strategy_name = equity_df["strategy_name"].iloc[0]
    strategy_label = equity_df["strategy_label"].iloc[0]
    final_bankroll = float(equity_df["bankroll_after"].iloc[-1]) if not equity_df.empty else INITIAL_BANKROLL
    total_profit = final_bankroll - INITIAL_BANKROLL
    total_staked = float(equity_df["stake"].sum()) if not equity_df.empty else 0.0
    roi = (total_profit / INITIAL_BANKROLL) if INITIAL_BANKROLL > 0 else 0.0
    turnover_roi = (total_profit / total_staked) if total_staked > 0 else 0.0
    max_drawdown = float(abs(equity_df["drawdown"].min())) if not equity_df.empty else 0.0
    max_drawdown_pct = max_drawdown / INITIAL_BANKROLL if INITIAL_BANKROLL > 0 else 0.0
    std_dev_returns = float(equity_df["bankroll_return"].dropna().std(ddof=0)) if not equity_df.empty else 0.0
    longest_losing_streak = int(equity_df["losing_streak"].max()) if not equity_df.empty else 0
    max_stake = float(equity_df["stake"].max()) if not equity_df.empty else 0.0
    max_stake_pct = float(equity_df["stake_pct_bankroll"].max()) if not equity_df.empty else 0.0
    avg_stake = float(equity_df["stake"].mean()) if not equity_df.empty else 0.0
    min_bankroll = float(equity_df["bankroll_after"].min()) if not equity_df.empty else INITIAL_BANKROLL
    bets_placed = int(equity_df["bet_placed"].sum()) if not equity_df.empty else 0
    wins = int(((equity_df["bet_placed"] == 1) & (equity_df["profit"] > 0)).sum()) if not equity_df.empty else 0
    losses = int(((equity_df["bet_placed"] == 1) & (equity_df["profit"] < 0)).sum()) if not equity_df.empty else 0
    win_rate = wins / bets_placed if bets_placed > 0 else 0.0
    blew_up_bankroll = min_bankroll <= 0 or final_bankroll <= INITIAL_BANKROLL * 0.5
    extreme_bet_spike = max_stake_pct > MAX_ACCEPTABLE_STAKE_PCT
    unstable_drawdown = max_drawdown_pct > MAX_ACCEPTABLE_DRAWDOWN_PCT
    decision = "REJECT" if blew_up_bankroll or extreme_bet_spike or unstable_drawdown else "ACCEPT"

    return {
        "strategy_name": strategy_name,
        "strategy_label": strategy_label,
        "initial_bankroll": INITIAL_BANKROLL,
        "final_bankroll": final_bankroll,
        "total_profit": total_profit,
        "roi": roi,
        "turnover_roi": turnover_roi,
        "total_staked": total_staked,
        "bets_placed": bets_placed,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "std_dev_returns": std_dev_returns,
        "longest_losing_streak": longest_losing_streak,
        "avg_stake": avg_stake,
        "max_stake": max_stake,
        "max_stake_pct": max_stake_pct,
        "min_bankroll": min_bankroll,
        "blew_up_bankroll": blew_up_bankroll,
        "extreme_bet_spike": extreme_bet_spike,
        "unstable_drawdown": unstable_drawdown,
        "decision": decision,
    }


def run_experiments(
    betting_ready_path: Path = BETTING_READY_PATH,
    reports_dir: Path = REPORTS_DIR,
    equity_curves_dir: Path = EQUITY_CURVES_DIR,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    equity_curves_dir.mkdir(parents=True, exist_ok=True)

    v2_core_bets = load_v2_core_bets(betting_ready_path)
    summary_rows: list[dict[str, object]] = []
    equity_curves: dict[str, pd.DataFrame] = {}

    for strategy in strategy_definitions():
        equity_df, summary = simulate_strategy(v2_core_bets, strategy)
        equity_curves[str(strategy["strategy_name"])] = equity_df
        summary_rows.append(summary)
        equity_df.to_csv(equity_curves_dir / f"{strategy['strategy_name']}_sizing.csv", index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["final_bankroll", "roi", "max_drawdown"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    return summary_df, equity_curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bet sizing experiments on validated V2_Core bets.")
    parser.add_argument("--betting-ready", type=Path, default=BETTING_READY_PATH)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--equity-curves-dir", type=Path, default=EQUITY_CURVES_DIR)
    args = parser.parse_args()

    summary_df, _ = run_experiments(
        betting_ready_path=args.betting_ready,
        reports_dir=args.reports_dir,
        equity_curves_dir=args.equity_curves_dir,
    )
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
