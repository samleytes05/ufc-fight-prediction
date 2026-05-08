from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_TRACKING_DIR = PROJECT_ROOT / "outputs" / "strategy" / "live_tracking"
MPL_CONFIG_DIR = PROJECT_ROOT / "outputs" / "strategy" / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIVE_TRACKING_PATH = LIVE_TRACKING_DIR / "live_bet_tracking.csv"
FINAL_STATUS_PATH = PROJECT_ROOT / "docs" / "strategy" / "BETTING_STRATEGY_STATUS.md"

PENDING_BETS_PATH = LIVE_TRACKING_DIR / "pending_live_bets.csv"
SETTLED_BETS_PATH = LIVE_TRACKING_DIR / "settled_live_bets.csv"
PERFORMANCE_SUMMARY_PATH = LIVE_TRACKING_DIR / "live_performance_summary.csv"
CLV_SUMMARY_PATH = LIVE_TRACKING_DIR / "live_clv_summary.csv"
EVENT_SUMMARY_PATH = LIVE_TRACKING_DIR / "live_event_summary.csv"
BACKTEST_COMPARISON_PATH = LIVE_TRACKING_DIR / "live_backtest_comparison.csv"
MARKDOWN_REPORT_PATH = LIVE_TRACKING_DIR / "LIVE_SHADOW_EVIDENCE.md"
EQUITY_CURVE_PATH = LIVE_TRACKING_DIR / "live_equity_curve.png"
EVENT_PROFIT_CHART_PATH = LIVE_TRACKING_DIR / "live_event_profit.png"

BACKTEST_V2_CORE_ROI = 0.1121
BACKTEST_V2_CORE_WIN_RATE = 0.6031
BACKTEST_V2_CORE_PROFITABLE_FOLD_RATE = 0.80

EXPECTED_COLUMNS = [
    "run_timestamp",
    "event",
    "fight",
    "fighter_A",
    "fighter_B",
    "selected_fighter",
    "odds_A",
    "odds_at_pick",
    "closing_odds",
    "p_model_A",
    "implied_prob_A",
    "edge_A",
    "stake_flat_100",
    "stake_bankroll_1pct",
    "stake_kelly_capped_2pct",
    "expected_profit_flat_100",
    "expected_profit_bankroll_1pct",
    "expected_profit_kelly_capped_2pct",
    "result",
    "profit_flat_100",
    "profit_bankroll_1pct",
    "profit_kelly_capped_2pct",
    "line_movement",
    "beat_closing_line_flag",
    "notes",
]

SIZING_PROFILES = [
    {
        "profile": "flat_100",
        "stake_col": "stake_flat_100",
        "profit_col": "profit_flat_100",
        "expected_col": "expected_profit_flat_100",
    },
    {
        "profile": "bankroll_1pct",
        "stake_col": "stake_bankroll_1pct",
        "profit_col": "profit_bankroll_1pct",
        "expected_col": "expected_profit_bankroll_1pct",
    },
    {
        "profile": "kelly_capped_2pct",
        "stake_col": "stake_kelly_capped_2pct",
        "profit_col": "profit_kelly_capped_2pct",
        "expected_col": "expected_profit_kelly_capped_2pct",
    },
]


def load_live_tracking(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=EXPECTED_COLUMNS)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    for column in EXPECTED_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    numeric_columns = [
        "odds_A",
        "odds_at_pick",
        "closing_odds",
        "p_model_A",
        "implied_prob_A",
        "edge_A",
        "stake_flat_100",
        "stake_bankroll_1pct",
        "stake_kelly_capped_2pct",
        "expected_profit_flat_100",
        "expected_profit_bankroll_1pct",
        "expected_profit_kelly_capped_2pct",
        "profit_flat_100",
        "profit_bankroll_1pct",
        "profit_kelly_capped_2pct",
        "line_movement",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["run_timestamp"] = pd.to_datetime(df["run_timestamp"], errors="coerce", utc=True)
    df["notes"] = df["notes"].fillna("").astype(str)
    df["result"] = df["result"].fillna("").astype(str).str.strip().str.lower()
    return df


def split_pending_and_settled(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    profit_present = df[["profit_flat_100", "profit_bankroll_1pct", "profit_kelly_capped_2pct"]].notna().any(axis=1)
    pending_mask = df["result"].eq("") | df["notes"].str.contains("pending", case=False, na=False)
    settled_mask = df["result"].ne("") & profit_present
    pending_df = df.loc[pending_mask].copy()
    settled_df = df.loc[settled_mask].copy()
    return pending_df, settled_df


def max_drawdown_from_profits(profits: pd.Series) -> float:
    if profits.empty:
        return 0.0
    cumulative = profits.fillna(0.0).cumsum()
    drawdown = cumulative - cumulative.cummax()
    return float(abs(drawdown.min()))


def longest_losing_streak_from_results(results: pd.Series) -> int:
    longest = 0
    current = 0
    for value in results.fillna("").astype(str).str.lower():
        if value == "loss":
            current += 1
            longest = max(longest, current)
        elif value == "win":
            current = 0
    return int(longest)


def summarize_performance(settled_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    wins = int(settled_df["result"].eq("win").sum())
    losses = int(settled_df["result"].eq("loss").sum())
    settled_bets = int(len(settled_df))
    win_rate = float(wins / settled_bets) if settled_bets > 0 else 0.0

    ordered = settled_df.sort_values(["run_timestamp", "event", "fight"]).reset_index(drop=True)

    for profile in SIZING_PROFILES:
        stake_col = profile["stake_col"]
        profit_col = profile["profit_col"]
        expected_col = profile["expected_col"]
        total_stake = float(ordered[stake_col].fillna(0.0).sum())
        total_profit = float(ordered[profit_col].fillna(0.0).sum())
        expected_total_profit = float(ordered[expected_col].fillna(0.0).sum())
        average_expected_profit = float(ordered[expected_col].fillna(0.0).mean()) if settled_bets > 0 else 0.0
        rows.append(
            {
                "sizing_profile": profile["profile"],
                "settled_bets": settled_bets,
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "total_stake": total_stake,
                "total_profit": total_profit,
                "roi": float(total_profit / total_stake) if total_stake > 0 else 0.0,
                "average_expected_profit": average_expected_profit,
                "expected_total_profit": expected_total_profit,
                "actual_minus_expected_profit": total_profit - expected_total_profit,
                "max_drawdown": max_drawdown_from_profits(ordered[profit_col]),
                "longest_losing_streak": longest_losing_streak_from_results(ordered["result"]),
            }
        )
    return pd.DataFrame(rows)


def derive_beat_closing_flag(df: pd.DataFrame) -> pd.Series:
    existing = df["beat_closing_line_flag"]
    existing_bool = existing.map(
        lambda value: True
        if str(value).strip().lower() in {"true", "1", "yes"}
        else False
        if str(value).strip().lower() in {"false", "0", "no"}
        else np.nan
    )

    implied_pick = np.where(df["odds_at_pick"].notna(), np.where(df["odds_at_pick"] > 0, 100.0 / (df["odds_at_pick"] + 100.0), np.abs(df["odds_at_pick"]) / (np.abs(df["odds_at_pick"]) + 100.0)), np.nan)
    implied_close = np.where(df["closing_odds"].notna(), np.where(df["closing_odds"] > 0, 100.0 / (df["closing_odds"] + 100.0), np.abs(df["closing_odds"]) / (np.abs(df["closing_odds"]) + 100.0)), np.nan)
    derived = pd.Series(np.where(~np.isnan(implied_pick) & ~np.isnan(implied_close), implied_pick < implied_close, np.nan), index=df.index)
    return existing_bool.combine_first(derived)


def summarize_clv(settled_df: pd.DataFrame) -> pd.DataFrame:
    clv_df = settled_df.copy()
    clv_df["beat_closing_line_flag_derived"] = derive_beat_closing_flag(clv_df)
    with_close = clv_df[clv_df["closing_odds"].notna()].copy()
    if with_close.empty:
        return pd.DataFrame(
            [
                {
                    "rows_with_closing_odds": 0,
                    "average_line_movement": 0.0,
                    "beat_closing_line_count": 0,
                    "beat_closing_line_rate": 0.0,
                }
            ]
        )

    if "line_movement" not in with_close.columns or with_close["line_movement"].isna().all():
        with_close["line_movement"] = with_close["closing_odds"] - with_close["odds_at_pick"]

    beat_count = int(with_close["beat_closing_line_flag_derived"].fillna(False).sum())
    row = {
        "rows_with_closing_odds": int(len(with_close)),
        "average_line_movement": float(with_close["line_movement"].fillna(0.0).mean()),
        "beat_closing_line_count": beat_count,
        "beat_closing_line_rate": float(beat_count / len(with_close)) if len(with_close) > 0 else 0.0,
    }
    return pd.DataFrame([row])


def summarize_events(df: pd.DataFrame, settled_df: pd.DataFrame) -> pd.DataFrame:
    all_events = sorted(set(df["event"].fillna("").tolist()))
    settled_lookup = settled_df.copy()
    settled_lookup["beat_closing_line_flag_derived"] = derive_beat_closing_flag(settled_lookup)
    rows: list[dict[str, object]] = []

    for event_name in all_events:
        event_all = df[df["event"].fillna("") == event_name].copy()
        event_settled = settled_lookup[settled_lookup["event"].fillna("") == event_name].copy()
        total_recs = int(len(event_all))
        settled_recs = int(len(event_settled))
        pending_recs = total_recs - settled_recs
        wins = int(event_settled["result"].eq("win").sum())
        losses = int(event_settled["result"].eq("loss").sum())

        profit_flat = float(event_settled["profit_flat_100"].fillna(0.0).sum())
        profit_pct = float(event_settled["profit_bankroll_1pct"].fillna(0.0).sum())
        profit_kelly = float(event_settled["profit_kelly_capped_2pct"].fillna(0.0).sum())
        stake_flat = float(event_settled["stake_flat_100"].fillna(0.0).sum())
        stake_pct = float(event_settled["stake_bankroll_1pct"].fillna(0.0).sum())
        stake_kelly = float(event_settled["stake_kelly_capped_2pct"].fillna(0.0).sum())

        clv_rows = event_settled[event_settled["closing_odds"].notna()]
        beat_rate = float(clv_rows["beat_closing_line_flag_derived"].fillna(False).mean()) if len(clv_rows) > 0 else np.nan

        rows.append(
            {
                "event": event_name,
                "total_recommendations": total_recs,
                "settled_recommendations": settled_recs,
                "pending_recommendations": pending_recs,
                "wins": wins,
                "losses": losses,
                "profit_flat_100": profit_flat,
                "profit_bankroll_1pct": profit_pct,
                "profit_kelly_capped_2pct": profit_kelly,
                "ROI_flat_100": float(profit_flat / stake_flat) if stake_flat > 0 else 0.0,
                "ROI_bankroll_1pct": float(profit_pct / stake_pct) if stake_pct > 0 else 0.0,
                "ROI_kelly_capped_2pct": float(profit_kelly / stake_kelly) if stake_kelly > 0 else 0.0,
                "beat_closing_line_rate": beat_rate,
            }
        )
    return pd.DataFrame(rows)


def sample_size_warning(settled_bets: int) -> str:
    if settled_bets < 25:
        return "VERY EARLY - do not judge"
    if settled_bets < 75:
        return "EARLY - directional only"
    if settled_bets < 150:
        return "MODERATE - useful signal"
    return "STRONGER - compare seriously"


def compare_backtest(performance_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in performance_df.itertuples(index=False):
        rows.append(
            {
                "sizing_profile": row.sizing_profile,
                "settled_bets": int(row.settled_bets),
                "live_roi": float(row.roi),
                "backtest_roi": BACKTEST_V2_CORE_ROI,
                "live_roi_vs_backtest_roi": float(row.roi - BACKTEST_V2_CORE_ROI),
                "live_win_rate": float(row.win_rate),
                "backtest_win_rate": BACKTEST_V2_CORE_WIN_RATE,
                "live_win_rate_vs_backtest_win_rate": float(row.win_rate - BACKTEST_V2_CORE_WIN_RATE),
                "backtest_profitable_fold_rate": BACKTEST_V2_CORE_PROFITABLE_FOLD_RATE,
                "sample_size_warning": sample_size_warning(int(row.settled_bets)),
            }
        )
    return pd.DataFrame(rows)


def interpretation_label(comparison_df: pd.DataFrame) -> str:
    if comparison_df.empty:
        return "Too early to judge"
    row = comparison_df.loc[comparison_df["sizing_profile"] == "flat_100"]
    if row.empty:
        row = comparison_df.iloc[[0]]
    row = row.iloc[0]
    warning = str(row["sample_size_warning"])
    roi_delta = float(row["live_roi_vs_backtest_roi"])
    if warning.startswith("VERY EARLY") or warning.startswith("EARLY"):
        return "Too early to judge"
    if roi_delta >= -0.02:
        return "Tracking well"
    if roi_delta < -0.05:
        return "Possible model/market drift"
    return "Underperforming backtest"


def write_markdown_report(
    total_recommendations: int,
    pending_df: pd.DataFrame,
    settled_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    clv_df: pd.DataFrame,
    event_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    status_warning = sample_size_warning(len(settled_df))
    backtest_roi_text = f"{BACKTEST_V2_CORE_ROI:.4f}"
    backtest_win_rate_text = f"{BACKTEST_V2_CORE_WIN_RATE:.4f}"

    lines = [
        "# Live Shadow Evidence Report",
        "",
        "## Strategy",
        "- V2_Core",
        "- edge_A > 0.04",
        "- p_model_A >= 0.65",
        "",
        "## Current Status",
        f"- total recommendations: {total_recommendations}",
        f"- pending bets: {len(pending_df)}",
        f"- settled bets: {len(settled_df)}",
        f"- current sample-size warning: {status_warning}",
        "",
        "## Performance vs Backtest",
        f"- backtest ROI benchmark: {backtest_roi_text}",
        f"- backtest win rate benchmark: {backtest_win_rate_text}",
    ]

    for row in performance_df.itertuples(index=False):
        lines.append(
            f"- {row.sizing_profile}: live ROI {row.roi:.4f}, live win rate {row.win_rate:.4f}, "
            f"actual minus expected profit {row.actual_minus_expected_profit:.2f}"
        )

    lines.extend(
        [
            "",
            "## CLV Check",
        ]
    )
    if not clv_df.empty:
        clv_row = clv_df.iloc[0]
        lines.append(f"- beat closing line rate: {float(clv_row['beat_closing_line_rate']):.4f}")
        lines.append(f"- average line movement: {float(clv_row['average_line_movement']):.2f}")
    else:
        lines.append("- beat closing line rate: n/a")
        lines.append("- average line movement: n/a")

    lines.extend(["", "## Event Summary"])
    if event_df.empty:
        lines.append("No event-level data yet.")
    else:
        lines.append("")
        lines.append(event_df.to_markdown(index=False))

    lines.extend(
        [
            "",
            "## Interpretation",
            interpretation_label(comparison_df),
            "",
            "Do not make automatic real-money recommendations.",
            "",
        ]
    )
    MARKDOWN_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def create_equity_curve_chart(settled_df: pd.DataFrame) -> None:
    plt.figure()
    if settled_df.empty:
        plt.text(0.5, 0.5, "No settled bets yet", ha="center", va="center")
        plt.axis("off")
    else:
        ordered = settled_df.sort_values(["run_timestamp", "event", "fight"]).reset_index(drop=True)
        x = np.arange(1, len(ordered) + 1)
        for profile in SIZING_PROFILES:
            cumulative = ordered[profile["profit_col"]].fillna(0.0).cumsum()
            plt.plot(x, cumulative, label=profile["profile"])
        plt.xlabel("Settled Bet Number")
        plt.ylabel("Cumulative Profit")
        plt.legend()
    plt.tight_layout()
    plt.savefig(EQUITY_CURVE_PATH)
    plt.close()


def create_event_profit_chart(event_df: pd.DataFrame) -> None:
    plt.figure()
    if event_df.empty:
        plt.text(0.5, 0.5, "No settled event results yet", ha="center", va="center")
        plt.axis("off")
    else:
        plot_df = event_df.copy()
        plot_df = plot_df.sort_values("event").reset_index(drop=True)
        x = np.arange(len(plot_df))
        width = 0.25
        plt.bar(x - width, plot_df["profit_flat_100"], width=width, label="flat_100")
        plt.bar(x, plot_df["profit_bankroll_1pct"], width=width, label="bankroll_1pct")
        plt.bar(x + width, plot_df["profit_kelly_capped_2pct"], width=width, label="kelly_capped_2pct")
        plt.xticks(x, plot_df["event"], rotation=45, ha="right")
        plt.ylabel("Event Profit")
        plt.legend()
    plt.tight_layout()
    plt.savefig(EVENT_PROFIT_CHART_PATH)
    plt.close()


def update_final_status_doc() -> None:
    section_lines = [
        "## Live Shadow Evidence Tracking",
        "- evidence report path: `outputs/strategy/live_tracking/LIVE_SHADOW_EVIDENCE.md`",
        "- pending bets path: `outputs/strategy/live_tracking/pending_live_bets.csv`",
        "- settled bets path: `outputs/strategy/live_tracking/settled_live_bets.csv`",
        "- performance summary path: `outputs/strategy/live_tracking/live_performance_summary.csv`",
        "- CLV summary path: `outputs/strategy/live_tracking/live_clv_summary.csv`",
        "- evidence report command: `vmathv\\Scripts\\python.exe src\\strategy\\live_shadow_evidence.py`",
        "",
    ]
    if FINAL_STATUS_PATH.exists():
        existing_text = FINAL_STATUS_PATH.read_text(encoding="utf-8")
    else:
        existing_text = "# Phase 3 Final Status\n\n"

    marker = "## Live Shadow Evidence Tracking"
    if marker in existing_text:
        prefix = existing_text.split(marker, maxsplit=1)[0].rstrip() + "\n\n"
        existing_text = prefix
    if not existing_text.endswith("\n"):
        existing_text += "\n"
    FINAL_STATUS_PATH.write_text(existing_text + "\n".join(section_lines), encoding="utf-8")


def run_evidence_report(live_tracking_path: Path = LIVE_TRACKING_PATH) -> dict[str, pd.DataFrame]:
    LIVE_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    tracking_df = load_live_tracking(live_tracking_path)
    pending_df, settled_df = split_pending_and_settled(tracking_df)

    pending_df.to_csv(PENDING_BETS_PATH, index=False)
    settled_df.to_csv(SETTLED_BETS_PATH, index=False)

    performance_df = summarize_performance(settled_df)
    performance_df.to_csv(PERFORMANCE_SUMMARY_PATH, index=False)

    clv_df = summarize_clv(settled_df)
    clv_df.to_csv(CLV_SUMMARY_PATH, index=False)

    event_df = summarize_events(tracking_df, settled_df)
    event_df.to_csv(EVENT_SUMMARY_PATH, index=False)

    comparison_df = compare_backtest(performance_df)
    comparison_df.to_csv(BACKTEST_COMPARISON_PATH, index=False)

    write_markdown_report(
        total_recommendations=len(tracking_df),
        pending_df=pending_df,
        settled_df=settled_df,
        performance_df=performance_df,
        clv_df=clv_df,
        event_df=event_df,
        comparison_df=comparison_df,
    )
    create_equity_curve_chart(settled_df)
    create_event_profit_chart(event_df)
    update_final_status_doc()

    return {
        "tracking": tracking_df,
        "pending": pending_df,
        "settled": settled_df,
        "performance": performance_df,
        "clv": clv_df,
        "events": event_df,
        "comparison": comparison_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize live shadow evidence from the append-only live tracking log.")
    parser.add_argument("--live-tracking", type=Path, default=LIVE_TRACKING_PATH)
    args = parser.parse_args()

    outputs = run_evidence_report(live_tracking_path=args.live_tracking)
    print(f"Total recommendations: {len(outputs['tracking'])}")
    print(f"Pending bets: {len(outputs['pending'])}")
    print(f"Settled bets: {len(outputs['settled'])}")
    print(f"Saved evidence report: {MARKDOWN_REPORT_PATH}")


if __name__ == "__main__":
    main()
