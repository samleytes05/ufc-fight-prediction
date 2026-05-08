from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling" / "prediction_behavior"
LOGIT_PATH = PROJECT_ROOT / "outputs" / "modeling" / "tuning" / "best_calibrated_logit_predictions.csv"
RF_PATH = PROJECT_ROOT / "outputs" / "modeling" / "tuning" / "baseline_predictions_rf.csv"
RESULTS_PATH = PROJECT_ROOT / "data" / "historical_backfill" / "ufc_fight_results.csv"
EVENT_CATALOG_PATH = PROJECT_ROOT / "data" / "historical_backfill" / "historical_event_catalog_scraped.csv"
FEATURE_DATASET_PATH = PROJECT_ROOT / "data" / "historical_backfill" / "ufc_rebuilt_features_scraped.csv"


CONF_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]
CONF_BUCKETS = [0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
CONF_LABELS = ["50-55%", "55-60%", "60-65%", "65-70%", "70%+"]


def safe_log_loss(y_true: pd.Series, probabilities: pd.Series) -> float:
    if len(y_true) == 0:
        return np.nan
    clipped = np.clip(probabilities.to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    return float(log_loss(y_true, clipped))


def safe_brier(y_true: pd.Series, probabilities: pd.Series) -> float:
    if len(y_true) == 0:
        return np.nan
    return float(brier_score_loss(y_true, probabilities))


def parse_fight_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df.copy()
    split_cols = parsed["fight_id"].str.split(" | ", n=1, expand=True, regex=False)
    parsed["EVENT"] = split_cols[0]
    parsed["BOUT"] = split_cols[1]
    return parsed


def load_prediction_behavior_table() -> pd.DataFrame:
    logit_df = pd.read_csv(LOGIT_PATH)
    rf_df = pd.read_csv(RF_PATH)

    merged = logit_df.merge(
        rf_df,
        on=["fold", "fight_id", "fight_order", "y_true"],
        suffixes=("_logit", "_rf"),
        how="inner",
    )
    merged = merged.rename(columns={"y_true": "target_A_win"})
    merged["logit_probability"] = merged["predicted_probability_logit"]
    merged["rf_probability"] = merged["predicted_probability_rf"]
    merged["logit_pick"] = (merged["logit_probability"] >= 0.5).astype(int)
    merged["rf_pick"] = (merged["rf_probability"] >= 0.5).astype(int)
    merged["logit_correct"] = (merged["logit_pick"] == merged["target_A_win"]).astype(int)
    merged["rf_correct"] = (merged["rf_pick"] == merged["target_A_win"]).astype(int)
    merged["agree"] = (merged["logit_pick"] == merged["rf_pick"]).astype(int)
    merged["agreement_label"] = np.where(merged["agree"] == 1, "agreement", "disagreement")
    merged["logit_confidence"] = (merged["logit_probability"] - 0.5).abs()
    merged["rf_confidence"] = (merged["rf_probability"] - 0.5).abs()
    merged["confidence_gap"] = merged["logit_confidence"] - merged["rf_confidence"]
    merged["abs_confidence_gap"] = merged["confidence_gap"].abs()
    merged["higher_confidence_model"] = np.select(
        [
            merged["logit_confidence"] > merged["rf_confidence"],
            merged["rf_confidence"] > merged["logit_confidence"],
        ],
        ["logistic_regression", "random_forest"],
        default="tie",
    )
    merged["max_confidence"] = merged[["logit_confidence", "rf_confidence"]].max(axis=1)
    merged["avg_confidence"] = merged[["logit_confidence", "rf_confidence"]].mean(axis=1)
    merged["logit_side_probability"] = np.where(
        merged["logit_pick"] == 1, merged["logit_probability"], 1 - merged["logit_probability"]
    )
    merged["rf_side_probability"] = np.where(
        merged["rf_pick"] == 1, merged["rf_probability"], 1 - merged["rf_probability"]
    )
    merged["confidence_gap_bucket"] = pd.cut(
        merged["abs_confidence_gap"],
        bins=[-0.001, 0.025, 0.075, 1.0],
        labels=["small", "medium", "large"],
    )
    merged = parse_fight_id_columns(merged)

    results_df = pd.read_csv(RESULTS_PATH)
    event_df = pd.read_csv(EVENT_CATALOG_PATH)
    feature_df = pd.read_csv(FEATURE_DATASET_PATH)[
        [
            "fight_id",
            "fight_order",
            "age_diff",
            "reach_diff",
            "experience_advantage_ratio_log",
            "elo_diff",
        ]
    ].copy()
    results_df["fight_id"] = results_df["EVENT"].astype(str) + " | " + results_df["BOUT"].astype(str)

    merged = merged.merge(
        results_df[["fight_id", "WEIGHTCLASS", "METHOD", "ROUND", "TIME", "URL"]],
        on="fight_id",
        how="left",
    )
    merged = merged.merge(
        event_df[["EVENT", "DATE", "LOCATION"]],
        on="EVENT",
        how="left",
    )
    merged = merged.merge(feature_df, on=["fight_id", "fight_order"], how="left")
    merged["DATE"] = pd.to_datetime(merged["DATE"], errors="coerce")
    merged["year"] = merged["DATE"].dt.year
    merged["era_bucket"] = pd.cut(
        merged["fight_order"],
        bins=[-1, 2000, 3500, 5000, 7000, np.inf],
        labels=["early", "growth", "modern_1", "modern_2", "current"],
    )
    merged["age_diff_bucket"] = pd.cut(
        merged["age_diff"],
        bins=[-np.inf, -4, -2, 0, 2, 4, np.inf],
        labels=["<=-4", "-4 to -2", "-2 to 0", "0 to 2", "2 to 4", "4+"],
    )
    merged["reach_diff_bucket"] = pd.cut(
        merged["reach_diff"],
        bins=[-np.inf, -5, -2, 0, 2, 5, np.inf],
        labels=["<=-5", "-5 to -2", "-2 to 0", "0 to 2", "2 to 5", "5+"],
    )
    merged["experience_gap_bucket"] = pd.cut(
        merged["experience_advantage_ratio_log"],
        bins=[-np.inf, -0.7, -0.3, 0, 0.3, 0.7, np.inf],
        labels=["<=-0.7", "-0.7 to -0.3", "-0.3 to 0", "0 to 0.3", "0.3 to 0.7", "0.7+"],
    )
    merged["elo_gap_bucket"] = pd.cut(
        merged["elo_diff"],
        bins=[-np.inf, -100, -40, 0, 40, 100, np.inf],
        labels=["<=-100", "-100 to -40", "-40 to 0", "0 to 40", "40 to 100", "100+"],
    )
    merged["logit_confidence_bucket"] = pd.cut(
        merged["logit_side_probability"],
        bins=CONF_BUCKETS,
        labels=CONF_LABELS,
        include_lowest=True,
        right=False,
    )
    merged["rf_confidence_bucket"] = pd.cut(
        merged["rf_side_probability"],
        bins=CONF_BUCKETS,
        labels=CONF_LABELS,
        include_lowest=True,
        right=False,
    )
    return merged.sort_values(["fight_order", "fight_id"]).reset_index(drop=True)


def accuracy_from_pick(target: pd.Series, pick: pd.Series) -> float:
    if len(target) == 0:
        return np.nan
    return float((target == pick).mean())


def summarize_group(frame: pd.DataFrame, label: str, total_rows: int) -> dict[str, object]:
    return {
        "group": label,
        "count": int(len(frame)),
        "share_of_total": float(len(frame) / total_rows) if total_rows else np.nan,
        "logit_accuracy": accuracy_from_pick(frame["target_A_win"], frame["logit_pick"]),
        "rf_accuracy": accuracy_from_pick(frame["target_A_win"], frame["rf_pick"]),
        "logit_log_loss": safe_log_loss(frame["target_A_win"], frame["logit_probability"]),
        "rf_log_loss": safe_log_loss(frame["target_A_win"], frame["rf_probability"]),
        "logit_brier": safe_brier(frame["target_A_win"], frame["logit_probability"]),
        "rf_brier": safe_brier(frame["target_A_win"], frame["rf_probability"]),
        "avg_logit_confidence": float(frame["logit_confidence"].mean()) if len(frame) else np.nan,
        "avg_rf_confidence": float(frame["rf_confidence"].mean()) if len(frame) else np.nan,
        "avg_confidence_gap_abs": float(frame["abs_confidence_gap"].mean()) if len(frame) else np.nan,
    }


def agreement_analysis(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rows = [
        summarize_group(df, "overall", total),
        summarize_group(df[df["agree"] == 1], "agreement", total),
        summarize_group(df[df["agree"] == 0], "disagreement", total),
    ]
    return pd.DataFrame(rows)


def disagreement_analysis(df: pd.DataFrame) -> pd.DataFrame:
    disagree = df[df["agree"] == 0].copy()
    rows = [
        summarize_group(disagree, "all_disagreement", len(df)),
        {
            "group": "all_disagreement",
            "count": int(len(disagree)),
            "share_of_total": float(len(disagree) / len(df)),
            "logit_accuracy": accuracy_from_pick(disagree["target_A_win"], disagree["logit_pick"]),
            "rf_accuracy": accuracy_from_pick(disagree["target_A_win"], disagree["rf_pick"]),
            "more_trustworthy_model": (
                "logistic_regression"
                if accuracy_from_pick(disagree["target_A_win"], disagree["logit_pick"])
                > accuracy_from_pick(disagree["target_A_win"], disagree["rf_pick"])
                else "random_forest"
            ),
            "avg_logit_confidence": float(disagree["logit_confidence"].mean()) if len(disagree) else np.nan,
            "avg_rf_confidence": float(disagree["rf_confidence"].mean()) if len(disagree) else np.nan,
            "accuracy_when_logit_more_confident": accuracy_from_pick(
                disagree[disagree["higher_confidence_model"] == "logistic_regression"]["target_A_win"],
                disagree[disagree["higher_confidence_model"] == "logistic_regression"]["logit_pick"],
            ),
            "accuracy_when_rf_more_confident": accuracy_from_pick(
                disagree[disagree["higher_confidence_model"] == "random_forest"]["target_A_win"],
                disagree[disagree["higher_confidence_model"] == "random_forest"]["rf_pick"],
            ),
            "accuracy_small_confidence_gap_more_confident_model": np.nan,
            "accuracy_large_confidence_gap_more_confident_model": np.nan,
        },
    ]
    small = disagree[disagree["confidence_gap_bucket"] == "small"].copy()
    large = disagree[disagree["confidence_gap_bucket"] == "large"].copy()

    def more_confident_accuracy(frame: pd.DataFrame) -> float:
        if len(frame) == 0:
            return np.nan
        chosen_pick = np.where(
            frame["higher_confidence_model"] == "logistic_regression",
            frame["logit_pick"],
            np.where(frame["higher_confidence_model"] == "random_forest", frame["rf_pick"], frame["logit_pick"]),
        )
        return float((frame["target_A_win"].to_numpy() == chosen_pick).mean())

    rows[1]["accuracy_small_confidence_gap_more_confident_model"] = more_confident_accuracy(small)
    rows[1]["accuracy_large_confidence_gap_more_confident_model"] = more_confident_accuracy(large)
    return pd.DataFrame(rows)


def confidence_bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes = {
        "overall": df,
        "agreement_only": df[df["agree"] == 1],
        "disagreement_only": df[df["agree"] == 0],
    }
    model_specs = {
        "logistic_regression": ("logit_side_probability", "logit_pick"),
        "random_forest": ("rf_side_probability", "rf_pick"),
    }
    for scope_name, scope_df in scopes.items():
        for model_name, (prob_col, pick_col) in model_specs.items():
            scope_df = scope_df.copy()
            scope_df["confidence_bucket"] = pd.cut(
                scope_df[prob_col],
                bins=CONF_BUCKETS,
                labels=CONF_LABELS,
                include_lowest=True,
                right=False,
            )
            grouped = scope_df.groupby("confidence_bucket", observed=False)
            for bucket, bucket_df in grouped:
                if len(bucket_df) == 0:
                    continue
                accuracy = accuracy_from_pick(bucket_df["target_A_win"], bucket_df[pick_col])
                avg_prob = float(bucket_df[prob_col].mean())
                rows.append(
                    {
                        "scope": scope_name,
                        "model": model_name,
                        "confidence_bucket": bucket,
                        "count": int(len(bucket_df)),
                        "accuracy": accuracy,
                        "avg_predicted_probability": avg_prob,
                        "calibration_gap": avg_prob - accuracy,
                    }
                )
    return pd.DataFrame(rows)


def apply_rule(df: pd.DataFrame, rule_name: str, threshold: float | None = None) -> pd.DataFrame:
    frame = df.copy()
    if rule_name == "always_logistic":
        frame["rule_pick"] = frame["logit_pick"]
        frame["rule_probability"] = frame["logit_probability"]
        return frame
    if rule_name == "always_random_forest":
        frame["rule_pick"] = frame["rf_pick"]
        frame["rule_probability"] = frame["rf_probability"]
        return frame
    if rule_name == "agree_then_logistic":
        frame["rule_pick"] = np.where(frame["agree"] == 1, frame["logit_pick"], frame["logit_pick"])
        frame["rule_probability"] = np.where(
            frame["agree"] == 1,
            (frame["logit_probability"] + frame["rf_probability"]) / 2.0,
            frame["logit_probability"],
        )
        return frame
    if rule_name == "agree_then_random_forest":
        frame["rule_pick"] = np.where(frame["agree"] == 1, frame["rf_pick"], frame["rf_pick"])
        frame["rule_probability"] = np.where(
            frame["agree"] == 1,
            (frame["logit_probability"] + frame["rf_probability"]) / 2.0,
            frame["rf_probability"],
        )
        return frame
    if rule_name == "agree_then_higher_confidence":
        frame["rule_pick"] = np.where(
            frame["agree"] == 1,
            frame["logit_pick"],
            np.where(frame["logit_confidence"] >= frame["rf_confidence"], frame["logit_pick"], frame["rf_pick"]),
        )
        frame["rule_probability"] = np.where(
            frame["agree"] == 1,
            (frame["logit_probability"] + frame["rf_probability"]) / 2.0,
            np.where(frame["logit_confidence"] >= frame["rf_confidence"], frame["logit_probability"], frame["rf_probability"]),
        )
        return frame
    if rule_name == "only_when_both_agree":
        frame = frame[frame["agree"] == 1].copy()
        frame["rule_pick"] = frame["logit_pick"]
        frame["rule_probability"] = (frame["logit_probability"] + frame["rf_probability"]) / 2.0
        return frame
    if rule_name == "logistic_confidence_threshold":
        assert threshold is not None
        frame = frame[frame["logit_confidence"] >= threshold].copy()
        frame["rule_pick"] = frame["logit_pick"]
        frame["rule_probability"] = frame["logit_probability"]
        return frame
    if rule_name == "agree_and_logistic_threshold":
        assert threshold is not None
        frame = frame[(frame["agree"] == 1) & (frame["logit_confidence"] >= threshold)].copy()
        frame["rule_pick"] = frame["logit_pick"]
        frame["rule_probability"] = (frame["logit_probability"] + frame["rf_probability"]) / 2.0
        return frame
    raise ValueError(f"Unsupported rule: {rule_name}")


def decision_rule_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rules: list[tuple[str, float | None]] = [
        ("always_logistic", None),
        ("always_random_forest", None),
        ("agree_then_logistic", None),
        ("agree_then_random_forest", None),
        ("agree_then_higher_confidence", None),
        ("only_when_both_agree", None),
    ]
    for threshold in CONF_THRESHOLDS:
        rules.append(("logistic_confidence_threshold", threshold))
        rules.append(("agree_and_logistic_threshold", threshold))

    for rule_name, threshold in rules:
        rule_df = apply_rule(df, rule_name, threshold)
        count = len(rule_df)
        accuracy = accuracy_from_pick(rule_df["target_A_win"], rule_df["rule_pick"])
        side_probability = np.where(rule_df["rule_pick"] == 1, rule_df["rule_probability"], 1 - rule_df["rule_probability"])
        avg_confidence = float(np.mean(np.abs(rule_df["rule_probability"] - 0.5))) if count else np.nan
        rows.append(
            {
                "rule_name": rule_name if threshold is None else f"{rule_name}_{int((0.5 + threshold) * 100)}",
                "threshold": threshold,
                "picks_made": int(count),
                "coverage_pct": float(count / len(df)) if len(df) else np.nan,
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
                "avg_side_probability": float(np.mean(side_probability)) if count else np.nan,
                "log_loss": safe_log_loss(rule_df["target_A_win"], rule_df["rule_probability"]),
                "brier_score": safe_brier(rule_df["target_A_win"], rule_df["rule_probability"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["accuracy", "coverage_pct"], ascending=[False, False]).reset_index(drop=True)


def error_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    segment_specs = {
        "year": "year",
        "era_bucket": "era_bucket",
        "weightclass": "WEIGHTCLASS",
        "age_diff_bucket": "age_diff_bucket",
        "reach_diff_bucket": "reach_diff_bucket",
        "experience_gap_bucket": "experience_gap_bucket",
        "elo_gap_bucket": "elo_gap_bucket",
        "logit_confidence_bucket": "logit_confidence_bucket",
        "rf_confidence_bucket": "rf_confidence_bucket",
    }
    rows: list[dict[str, object]] = []
    for segment_type, column in segment_specs.items():
        grouped = df.groupby(column, dropna=False)
        for segment_value, segment_df in grouped:
            if len(segment_df) == 0:
                continue
            rows.append(
                {
                    "segment_type": segment_type,
                    "segment_value": str(segment_value),
                    "count": int(len(segment_df)),
                    "logit_accuracy": accuracy_from_pick(segment_df["target_A_win"], segment_df["logit_pick"]),
                    "rf_accuracy": accuracy_from_pick(segment_df["target_A_win"], segment_df["rf_pick"]),
                    "disagreement_rate": float(segment_df["agree"].eq(0).mean()),
                    "avg_logit_confidence": float(segment_df["logit_confidence"].mean()),
                    "avg_rf_confidence": float(segment_df["rf_confidence"].mean()),
                    "avg_logit_log_loss": safe_log_loss(segment_df["target_A_win"], segment_df["logit_probability"]),
                    "avg_rf_log_loss": safe_log_loss(segment_df["target_A_win"], segment_df["rf_probability"]),
                    "weaker_model": (
                        "logistic_regression"
                        if accuracy_from_pick(segment_df["target_A_win"], segment_df["logit_pick"])
                        < accuracy_from_pick(segment_df["target_A_win"], segment_df["rf_pick"])
                        else "random_forest"
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["segment_type", "count"], ascending=[True, False]).reset_index(drop=True)


def render_report(
    behavior_df: pd.DataFrame,
    agreement_df: pd.DataFrame,
    disagreement_df: pd.DataFrame,
    confidence_df: pd.DataFrame,
    rule_df: pd.DataFrame,
    segment_df: pd.DataFrame,
) -> str:
    agreement_row = agreement_df[agreement_df["group"] == "agreement"].iloc[0]
    disagreement_row = agreement_df[agreement_df["group"] == "disagreement"].iloc[0]
    disagreement_detail = disagreement_df.iloc[-1]
    best_rule = rule_df.sort_values(["accuracy", "coverage_pct"], ascending=[False, False]).iloc[0]
    best_balanced_rule = rule_df[rule_df["coverage_pct"] >= 0.5].sort_values(
        ["accuracy", "log_loss"], ascending=[False, True]
    ).iloc[0]
    high_conf_buckets = confidence_df[
        (confidence_df["scope"] == "overall")
        & (confidence_df["confidence_bucket"].isin(["65-70%", "70%+"]))
    ].copy()
    worst_segments = segment_df.sort_values(["count", "logit_accuracy"], ascending=[False, True]).head(10)

    lines = [
        "# Phase 2.9 Prediction Behavior Report",
        "",
        "## Models Compared",
        "- Primary production candidate: tuned logistic regression + Platt calibration",
        "- Secondary benchmark: random forest + pruned_combined",
        "- Prediction sources: saved walk-forward outputs from `outputs/modeling/tuning/` plus metadata merged from existing historical tables",
        "",
        "## Agreement vs Disagreement",
        (
            f"- Models agree on {agreement_row['count']} of {len(behavior_df)} fights "
            f"({agreement_row['share_of_total']:.1%})."
        ),
        (
            f"- On agreement rows: logistic accuracy={agreement_row['logit_accuracy']:.3f}, "
            f"RF accuracy={agreement_row['rf_accuracy']:.3f}, "
            f"avg logistic confidence={agreement_row['avg_logit_confidence']:.3f}, "
            f"avg RF confidence={agreement_row['avg_rf_confidence']:.3f}."
        ),
        (
            f"- On disagreement rows: logistic accuracy={disagreement_row['logit_accuracy']:.3f}, "
            f"RF accuracy={disagreement_row['rf_accuracy']:.3f}."
        ),
        "",
        "## Disagreement Findings",
        (
            f"- When the models disagree, the more trustworthy model is `{disagreement_detail['more_trustworthy_model']}` "
            f"based on disagreement-only accuracy."
        ),
        (
            f"- Accuracy when logistic is more confident: {disagreement_detail['accuracy_when_logit_more_confident']:.3f}; "
            f"when RF is more confident: {disagreement_detail['accuracy_when_rf_more_confident']:.3f}."
        ),
        (
            f"- Small confidence-gap disagreement accuracy (more-confident model): "
            f"{disagreement_detail['accuracy_small_confidence_gap_more_confident_model']:.3f}; "
            f"large-gap: {disagreement_detail['accuracy_large_confidence_gap_more_confident_model']:.3f}."
        ),
        "",
        "## Confidence Analysis",
        (
            f"- Best high-confidence overall bucket snapshot:\n{high_conf_buckets.to_string(index=False)}"
            if not high_conf_buckets.empty
            else "- No high-confidence bucket summary available."
        ),
        "",
        "## Decision Rules",
        (
            f"- Best raw-accuracy rule: `{best_rule['rule_name']}` "
            f"(coverage={best_rule['coverage_pct']:.1%}, accuracy={best_rule['accuracy']:.3f})."
        ),
        (
            f"- Best balanced rule with at least 50% coverage: `{best_balanced_rule['rule_name']}` "
            f"(coverage={best_balanced_rule['coverage_pct']:.1%}, accuracy={best_balanced_rule['accuracy']:.3f}, "
            f"log_loss={best_balanced_rule['log_loss']:.3f})."
        ),
        "",
        "## Error Segments",
        worst_segments.to_string(index=False),
        "",
        "## Recommendation",
        (
            "- Logistic should remain the primary production model."
            if disagreement_row["logit_accuracy"] >= disagreement_row["rf_accuracy"]
            else "- RF should be reconsidered as the primary production model."
        ),
        (
            "- RF adds useful disagreement information and can be used as a confidence filter."
            if disagreement_row["logit_accuracy"] != disagreement_row["rf_accuracy"]
            else "- RF behaves mostly as a benchmark and adds limited disagreement edge."
        ),
        (
            "- Model agreement looks usable as a higher-confidence decision filter."
            if agreement_row["logit_accuracy"] > agreement_df[agreement_df["group"] == "overall"]["logit_accuracy"].iloc[0]
            else "- Model agreement does not materially raise reliability over overall predictions."
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    behavior_df = load_prediction_behavior_table()
    behavior_df.to_csv(OUTPUT_DIR / "prediction_behavior_table.csv", index=False)

    agreement_df = agreement_analysis(behavior_df)
    agreement_df.to_csv(OUTPUT_DIR / "agreement_summary.csv", index=False)

    disagreement_df = disagreement_analysis(behavior_df)
    disagreement_df.to_csv(OUTPUT_DIR / "disagreement_summary.csv", index=False)

    confidence_df = confidence_bucket_summary(behavior_df)
    confidence_df.to_csv(OUTPUT_DIR / "confidence_bucket_summary.csv", index=False)

    rule_df = decision_rule_summary(behavior_df)
    rule_df.to_csv(OUTPUT_DIR / "decision_rule_summary.csv", index=False)

    segment_df = error_segment_summary(behavior_df)
    segment_df.to_csv(OUTPUT_DIR / "error_segment_summary.csv", index=False)

    report = render_report(
        behavior_df=behavior_df,
        agreement_df=agreement_df,
        disagreement_df=disagreement_df,
        confidence_df=confidence_df,
        rule_df=rule_df,
        segment_df=segment_df,
    )
    (OUTPUT_DIR / "prediction_behavior_report.md").write_text(report, encoding="utf-8")

    print(report)
    print()
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
