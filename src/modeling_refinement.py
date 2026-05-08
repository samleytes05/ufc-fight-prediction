from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from model import build_model_factories
from modeling_baseline import (
    DATASET_PATH,
    TARGET_COLUMN,
    build_expanding_splits,
    describe_model_factory,
    evaluate_model_subset,
    extract_logistic_coefficients,
    extract_permutation_importance,
    extract_tree_importances,
    load_dataset,
)
from modeling_pruning import CONSERVATIVE_DROP, REDUNDANT_DROP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling" / "refinement"
PRIMARY_MODEL = "random_forest"
SECONDARY_MODEL = "logistic_regression"
MAX_FEATURES_TO_ADD = 5
SIGNIFICANT_LOGLOSS_DEGRADE = 0.0015
SIGNIFICANT_BRIER_DEGRADE = 0.00075


def build_pruned_combined_features(feature_columns: list[str]) -> list[str]:
    removed = set(CONSERVATIVE_DROP + REDUNDANT_DROP)
    return [feature for feature in feature_columns if feature not in removed]


def add_candidate_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    df = df.copy()
    df["recency_striking_trend_diff"] = df["l3_striking_efficiency_diff"] - df["career_striking_efficiency_diff"]
    df["opponent_adjusted_recent_striking"] = df["l3_net_striking_diff"] * (
        1.0 + (df["avg_elo_last3_opp_diff"] / 400.0)
    )
    df["striking_grappling_synergy"] = df["l3_net_striking_diff"] * df["career_td_landed_per_round_diff"]
    df["experience_age_interaction"] = df["experience_advantage_ratio_log"] * df["age_diff"]
    df["pace_normalized_striking_edge"] = df["l3_net_striking_diff"] / (1.0 + df["l3_pace_diff"].abs())
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    feature_descriptions = {
        "recency_striking_trend_diff": "Recency-weighted proxy: l3_striking_efficiency_diff - career_striking_efficiency_diff",
        "opponent_adjusted_recent_striking": "Opponent-strength adjustment: l3_net_striking_diff scaled by avg_elo_last3_opp_diff",
        "striking_grappling_synergy": "Interaction: l3_net_striking_diff * career_td_landed_per_round_diff",
        "experience_age_interaction": "Interaction: experience_advantage_ratio_log * age_diff",
        "pace_normalized_striking_edge": "Pace-normalized edge: l3_net_striking_diff / (1 + abs(l3_pace_diff))",
    }
    return df, feature_descriptions


def evaluate_experiment_pair(
    df: pd.DataFrame,
    feature_subset_name: str,
    feature_subset: list[str],
    model_factories: dict[str, object],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []

    for model_name in (PRIMARY_MODEL, SECONDARY_MODEL):
        summary, predictions, fold_metrics = evaluate_model_subset(
            df=df,
            feature_subset_name=feature_subset_name,
            feature_subset=feature_subset,
            model_name=model_name,
            model_factory=model_factories[model_name],
            splits=splits,
        )
        summary_rows.append(summary)
        prediction_frames.append(predictions)
        fold_frames.append(fold_metrics)

    return (
        pd.DataFrame(summary_rows),
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(fold_frames, ignore_index=True),
    )


def compare_models_with_predictions(
    predictions_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    experiment_name: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    rf_predictions = predictions_df[predictions_df["model"] == PRIMARY_MODEL].copy()
    logit_predictions = predictions_df[predictions_df["model"] == SECONDARY_MODEL].copy()
    merged = rf_predictions.merge(
        logit_predictions,
        on=["feature_subset", "fold", "fight_id", "fight_order", "y_true"],
        suffixes=("_rf", "_logit"),
        how="inner",
    )
    merged["rf_predicted_class"] = (merged["predicted_probability_rf"] >= 0.5).astype(int)
    merged["logit_predicted_class"] = (merged["predicted_probability_logit"] >= 0.5).astype(int)
    merged["probability_gap"] = merged["predicted_probability_rf"] - merged["predicted_probability_logit"]
    merged["abs_probability_gap"] = merged["probability_gap"].abs()
    merged["disagreement"] = (merged["rf_predicted_class"] != merged["logit_predicted_class"]).astype(int)
    merged["experiment_name"] = experiment_name

    rf_row = summary_df[summary_df["model"] == PRIMARY_MODEL].iloc[0]
    logit_row = summary_df[summary_df["model"] == SECONDARY_MODEL].iloc[0]
    comparison_row = {
        "experiment_name": experiment_name,
        "feature_subset": rf_row["feature_subset"],
        "feature_count": int(rf_row["feature_count"]),
        "rf_accuracy": float(rf_row["accuracy"]),
        "rf_roc_auc": float(rf_row["roc_auc"]),
        "rf_log_loss": float(rf_row["log_loss"]),
        "rf_brier_score": float(rf_row["brier_score"]),
        "logit_accuracy": float(logit_row["accuracy"]),
        "logit_roc_auc": float(logit_row["roc_auc"]),
        "logit_log_loss": float(logit_row["log_loss"]),
        "logit_brier_score": float(logit_row["brier_score"]),
        "prediction_correlation": float(merged["predicted_probability_rf"].corr(merged["predicted_probability_logit"])),
        "disagreement_rate": float(merged["disagreement"].mean()),
        "mean_abs_probability_gap": float(merged["abs_probability_gap"].mean()),
    }
    return comparison_row, merged


def add_delta_columns(
    summary_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    baseline_summary_df: pd.DataFrame,
    baseline_fold_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_lookup = baseline_summary_df.set_index("model")
    summary_df = summary_df.copy()
    for metric in ["accuracy", "roc_auc", "log_loss", "brier_score"]:
        summary_df[f"delta_{metric}_vs_baseline"] = summary_df.apply(
            lambda row: float(row[metric] - baseline_lookup.loc[row["model"], metric]),
            axis=1,
        )

    baseline_fold = baseline_fold_df[["model", "fold", "accuracy", "roc_auc", "log_loss", "brier_score"]].rename(
        columns={
            "accuracy": "baseline_accuracy",
            "roc_auc": "baseline_roc_auc",
            "log_loss": "baseline_log_loss",
            "brier_score": "baseline_brier_score",
        }
    )
    fold_df = fold_df.merge(baseline_fold, on=["model", "fold"], how="left")
    fold_df["delta_accuracy_vs_baseline"] = fold_df["accuracy"] - fold_df["baseline_accuracy"]
    fold_df["delta_roc_auc_vs_baseline"] = fold_df["roc_auc"] - fold_df["baseline_roc_auc"]
    fold_df["delta_log_loss_vs_baseline"] = fold_df["log_loss"] - fold_df["baseline_log_loss"]
    fold_df["delta_brier_score_vs_baseline"] = fold_df["brier_score"] - fold_df["baseline_brier_score"]
    return summary_df, fold_df


def feature_decision(
    candidate_name: str,
    summary_df: pd.DataFrame,
    fold_df: pd.DataFrame,
) -> dict[str, object]:
    rows = {row["model"]: row for _, row in summary_df.iterrows()}
    improved_models: list[str] = []
    rejected_reasons: list[str] = []
    fold_consistency: dict[str, object] = {}

    for model_name in (PRIMARY_MODEL, SECONDARY_MODEL):
        row = rows[model_name]
        helps_model = (row["delta_log_loss_vs_baseline"] < 0) or (row["delta_brier_score_vs_baseline"] < 0)
        significant_harm = (
            row["delta_log_loss_vs_baseline"] > SIGNIFICANT_LOGLOSS_DEGRADE
            or row["delta_brier_score_vs_baseline"] > SIGNIFICANT_BRIER_DEGRADE
        )
        if helps_model:
            improved_models.append(model_name)
        if significant_harm:
            rejected_reasons.append(
                f"{model_name} degraded materially (log_loss {row['delta_log_loss_vs_baseline']:+.4f}, "
                f"brier {row['delta_brier_score_vs_baseline']:+.4f})"
            )

        model_fold = fold_df[fold_df["model"] == model_name].copy()
        fold_consistency[model_name] = {
            "log_loss_improved_folds": int((model_fold["delta_log_loss_vs_baseline"] < 0).sum()),
            "brier_improved_folds": int((model_fold["delta_brier_score_vs_baseline"] < 0).sum()),
            "fold_count": int(len(model_fold)),
        }

    accepted = bool(improved_models) and not rejected_reasons
    return {
        "feature_name": candidate_name,
        "improved_models": improved_models,
        "accepted": accepted,
        "status": (
            "improved_both_models"
            if accepted and len(improved_models) == 2
            else "improved_one_model"
            if accepted and len(improved_models) == 1
            else "rejected"
        ),
        "rejected_reasons": rejected_reasons,
        "fold_consistency_json": json.dumps(fold_consistency, sort_keys=True),
    }


def candidate_priority(summary_df: pd.DataFrame) -> float:
    rf_row = summary_df[summary_df["model"] == PRIMARY_MODEL].iloc[0]
    logit_row = summary_df[summary_df["model"] == SECONDARY_MODEL].iloc[0]
    return float(
        rf_row["delta_log_loss_vs_baseline"]
        + logit_row["delta_log_loss_vs_baseline"]
        + 0.5 * (rf_row["delta_brier_score_vs_baseline"] + logit_row["delta_brier_score_vs_baseline"])
    )


def write_feature_list_md(
    output_path: Path,
    baseline_features: list[str],
    candidate_descriptions: dict[str, str],
    candidate_results_df: pd.DataFrame,
    final_features: list[str],
) -> None:
    lines = [
        "# Feature List Used",
        "",
        "## Locked Baseline",
        f"- Base subset: `pruned_combined` ({len(baseline_features)} features)",
        f"- Features: {', '.join(baseline_features)}",
        "",
        "## Candidate Additions",
    ]
    for feature_name, description in candidate_descriptions.items():
        row = candidate_results_df[candidate_results_df["feature_name"] == feature_name].iloc[0]
        lines.extend(
            [
                f"### {feature_name}",
                f"- Definition: {description}",
                f"- Status: {row['status']}",
                f"- Improved models: {row['improved_models_json']}",
                f"- Rejected reasons: {row['rejected_reasons_json']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Final Refined Feature Set",
            f"- Feature count: {len(final_features)}",
            f"- Features: {', '.join(final_features)}",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def render_report(
    baseline_summary_df: pd.DataFrame,
    final_summary_df: pd.DataFrame,
    candidate_results_df: pd.DataFrame,
    model_comparison_df: pd.DataFrame,
    final_features: list[str],
    baseline_features: list[str],
) -> str:
    baseline_rf = baseline_summary_df[baseline_summary_df["model"] == PRIMARY_MODEL].iloc[0]
    baseline_logit = baseline_summary_df[baseline_summary_df["model"] == SECONDARY_MODEL].iloc[0]
    final_rf = final_summary_df[final_summary_df["model"] == PRIMARY_MODEL].iloc[0]
    final_logit = final_summary_df[final_summary_df["model"] == SECONDARY_MODEL].iloc[0]

    accepted_df = candidate_results_df[candidate_results_df["accepted"]].copy()
    rejected_df = candidate_results_df[~candidate_results_df["accepted"]].copy()
    final_comparison = model_comparison_df[model_comparison_df["experiment_name"] == "final_refined"].iloc[0]
    baseline_comparison = model_comparison_df[model_comparison_df["experiment_name"] == "baseline_anchor"].iloc[0]

    rf_stability = model_comparison_df[["experiment_name", "rf_log_loss", "rf_brier_score"]]
    logit_stability = model_comparison_df[["experiment_name", "logit_log_loss", "logit_brier_score"]]

    lines = [
        "# Phase 2.5 Dual-Model Refinement Report",
        "",
        "## Scope",
        "- Locked baselines: random_forest + pruned_combined, logistic_regression + pruned_combined",
        "- Validation unchanged from prior walk-forward experiments",
        "- Feature refinement limited to five derived candidates built from the existing rebuilt dataset only",
        "",
        "## Baseline vs Refined",
        (
            f"- Random forest baseline -> refined: log_loss {baseline_rf['log_loss']:.4f} -> {final_rf['log_loss']:.4f}, "
            f"brier {baseline_rf['brier_score']:.4f} -> {final_rf['brier_score']:.4f}, "
            f"roc_auc {baseline_rf['roc_auc']:.4f} -> {final_rf['roc_auc']:.4f}, "
            f"accuracy {baseline_rf['accuracy']:.4f} -> {final_rf['accuracy']:.4f}"
        ),
        (
            f"- Logistic baseline -> refined: log_loss {baseline_logit['log_loss']:.4f} -> {final_logit['log_loss']:.4f}, "
            f"brier {baseline_logit['brier_score']:.4f} -> {final_logit['brier_score']:.4f}, "
            f"roc_auc {baseline_logit['roc_auc']:.4f} -> {final_logit['roc_auc']:.4f}, "
            f"accuracy {baseline_logit['accuracy']:.4f} -> {final_logit['accuracy']:.4f}"
        ),
        "",
        "## Candidate Feature Impact",
        candidate_results_df[
            [
                "feature_name",
                "status",
                "rf_delta_log_loss",
                "rf_delta_brier",
                "logit_delta_log_loss",
                "logit_delta_brier",
                "improved_models_json",
            ]
        ].to_string(index=False),
        "",
        "## RF vs Logistic Behavior",
        (
            f"- Baseline prediction correlation={baseline_comparison['prediction_correlation']:.4f}, "
            f"disagreement_rate={baseline_comparison['disagreement_rate']:.4f}"
        ),
        (
            f"- Final refined prediction correlation={final_comparison['prediction_correlation']:.4f}, "
            f"disagreement_rate={final_comparison['disagreement_rate']:.4f}"
        ),
        (
            "- RF still leads on log loss and Brier in the final refined setup."
            if (final_rf["log_loss"] <= final_logit["log_loss"] and final_rf["brier_score"] <= final_logit["brier_score"])
            else "- Logistic closes the probability-quality gap meaningfully in the final refined setup."
        ),
        (
            "- Logistic remains competitive on AUC/accuracy."
            if (final_logit["roc_auc"] >= final_rf["roc_auc"] or final_logit["accuracy"] >= final_rf["accuracy"])
            else "- Logistic loses ground on both AUC and accuracy in the refined setup."
        ),
        "",
        "## Accepted Features",
        (f"- {', '.join(accepted_df['feature_name'].tolist())}" if not accepted_df.empty else "- None"),
        "",
        "## Rejected Features",
        (f"- {', '.join(rejected_df['feature_name'].tolist())}" if not rejected_df.empty else "- None"),
        "",
        "## Final Feature Set",
        f"- Started from {len(baseline_features)} baseline features and finished with {len(final_features)} features.",
        f"- Final features: {', '.join(final_features)}",
        "",
        "## Recommendation",
        (
            f"- Final recommended production model: `{PRIMARY_MODEL}` with the refined feature set "
            f"(log_loss={final_rf['log_loss']:.4f}, brier={final_rf['brier_score']:.4f})."
        ),
        (
            "- Keeping both models is justified because logistic remains a useful linear reference and competitive ranking benchmark."
            if (final_logit["roc_auc"] >= baseline_logit["roc_auc"] - 0.001)
            else "- Keeping both models is optional; random forest remains the main production path."
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, original_feature_columns = load_dataset()
    df, candidate_descriptions = add_candidate_features(df)
    all_feature_columns = list(original_feature_columns) + list(candidate_descriptions.keys())
    baseline_features = build_pruned_combined_features(original_feature_columns)
    model_factories = build_model_factories()
    splits = build_expanding_splits(df)

    metrics_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    model_comparison_rows: list[dict[str, object]] = []
    side_by_side_frames: list[pd.DataFrame] = []

    baseline_summary_df, baseline_predictions_df, baseline_fold_df = evaluate_experiment_pair(
        df=df,
        feature_subset_name="pruned_combined",
        feature_subset=baseline_features,
        model_factories=model_factories,
        splits=splits,
    )
    baseline_summary_df["experiment_name"] = "baseline_anchor"
    baseline_summary_df["added_features_json"] = "[]"
    baseline_summary_df["feature_count"] = len(baseline_features)
    baseline_fold_df["experiment_name"] = "baseline_anchor"
    baseline_fold_df["added_features_json"] = "[]"
    baseline_predictions_df["experiment_name"] = "baseline_anchor"
    baseline_comparison_row, baseline_side_by_side = compare_models_with_predictions(
        baseline_predictions_df,
        baseline_summary_df,
        "baseline_anchor",
    )
    model_comparison_rows.append(baseline_comparison_row)
    side_by_side_frames.append(baseline_side_by_side)
    metrics_frames.append(baseline_summary_df)
    fold_frames.append(baseline_fold_df)
    prediction_frames.append(baseline_predictions_df)

    baseline_predictions_df[baseline_predictions_df["model"] == PRIMARY_MODEL].to_csv(
        OUTPUT_DIR / "baseline_anchor_predictions_rf.csv",
        index=False,
    )
    baseline_predictions_df[baseline_predictions_df["model"] == SECONDARY_MODEL].to_csv(
        OUTPUT_DIR / "baseline_anchor_predictions_logit.csv",
        index=False,
    )

    candidate_rows: list[dict[str, object]] = []
    candidate_summaries: dict[str, pd.DataFrame] = {}

    for feature_name in list(candidate_descriptions.keys())[:MAX_FEATURES_TO_ADD]:
        experiment_name = f"single_add__{feature_name}"
        feature_subset = baseline_features + [feature_name]
        summary_df, predictions_df, fold_df = evaluate_experiment_pair(
            df=df,
            feature_subset_name=experiment_name,
            feature_subset=feature_subset,
            model_factories=model_factories,
            splits=splits,
        )
        summary_df, fold_df = add_delta_columns(summary_df, fold_df, baseline_summary_df, baseline_fold_df)
        summary_df["experiment_name"] = experiment_name
        summary_df["added_features_json"] = json.dumps([feature_name])
        summary_df["feature_count"] = len(feature_subset)
        fold_df["experiment_name"] = experiment_name
        fold_df["added_features_json"] = json.dumps([feature_name])
        predictions_df["experiment_name"] = experiment_name

        comparison_row, side_by_side_df = compare_models_with_predictions(predictions_df, summary_df, experiment_name)
        for model_name in (PRIMARY_MODEL, SECONDARY_MODEL):
            model_row = summary_df[summary_df["model"] == model_name].iloc[0]
            prefix = "rf" if model_name == PRIMARY_MODEL else "logit"
            comparison_row[f"{prefix}_delta_log_loss_vs_baseline"] = float(model_row["delta_log_loss_vs_baseline"])
            comparison_row[f"{prefix}_delta_brier_vs_baseline"] = float(model_row["delta_brier_score_vs_baseline"])
            comparison_row[f"{prefix}_delta_roc_auc_vs_baseline"] = float(model_row["delta_roc_auc_vs_baseline"])
            comparison_row[f"{prefix}_delta_accuracy_vs_baseline"] = float(model_row["delta_accuracy_vs_baseline"])
        model_comparison_rows.append(comparison_row)
        side_by_side_frames.append(side_by_side_df)
        metrics_frames.append(summary_df)
        fold_frames.append(fold_df)
        prediction_frames.append(predictions_df)
        candidate_summaries[feature_name] = summary_df

        decision = feature_decision(feature_name, summary_df, fold_df)
        decision["rf_delta_log_loss"] = float(
            summary_df[summary_df["model"] == PRIMARY_MODEL]["delta_log_loss_vs_baseline"].iloc[0]
        )
        decision["rf_delta_brier"] = float(
            summary_df[summary_df["model"] == PRIMARY_MODEL]["delta_brier_score_vs_baseline"].iloc[0]
        )
        decision["logit_delta_log_loss"] = float(
            summary_df[summary_df["model"] == SECONDARY_MODEL]["delta_log_loss_vs_baseline"].iloc[0]
        )
        decision["logit_delta_brier"] = float(
            summary_df[summary_df["model"] == SECONDARY_MODEL]["delta_brier_score_vs_baseline"].iloc[0]
        )
        decision["priority_score"] = candidate_priority(summary_df)
        decision["improved_models_json"] = json.dumps(decision["improved_models"])
        decision["rejected_reasons_json"] = json.dumps(decision["rejected_reasons"])
        candidate_rows.append(decision)

    candidate_results_df = pd.DataFrame(candidate_rows).sort_values("priority_score").reset_index(drop=True)

    current_features = list(baseline_features)
    current_best_score = 0.0
    step_records: list[dict[str, object]] = []
    for feature_name in candidate_results_df[candidate_results_df["accepted"]]["feature_name"].tolist():
        trial_features = current_features + [feature_name]
        trial_name = f"cumulative__{feature_name}"
        summary_df, predictions_df, fold_df = evaluate_experiment_pair(
            df=df,
            feature_subset_name=trial_name,
            feature_subset=trial_features,
            model_factories=model_factories,
            splits=splits,
        )
        summary_df, fold_df = add_delta_columns(summary_df, fold_df, baseline_summary_df, baseline_fold_df)
        score = candidate_priority(summary_df)
        keep = score < current_best_score
        step_records.append(
            {
                "feature_name": feature_name,
                "trial_score": float(score),
                "kept_in_final": bool(keep),
            }
        )
        if keep:
            current_features = trial_features
            current_best_score = score

    final_features = current_features
    final_summary_df, final_predictions_df, final_fold_df = evaluate_experiment_pair(
        df=df,
        feature_subset_name="final_refined",
        feature_subset=final_features,
        model_factories=model_factories,
        splits=splits,
    )
    final_summary_df, final_fold_df = add_delta_columns(final_summary_df, final_fold_df, baseline_summary_df, baseline_fold_df)
    final_summary_df["experiment_name"] = "final_refined"
    final_summary_df["added_features_json"] = json.dumps([feature for feature in final_features if feature not in baseline_features])
    final_summary_df["feature_count"] = len(final_features)
    final_fold_df["experiment_name"] = "final_refined"
    final_fold_df["added_features_json"] = final_summary_df["added_features_json"].iloc[0]
    final_predictions_df["experiment_name"] = "final_refined"
    final_comparison_row, final_side_by_side = compare_models_with_predictions(
        final_predictions_df,
        final_summary_df,
        "final_refined",
    )
    for model_name in (PRIMARY_MODEL, SECONDARY_MODEL):
        model_row = final_summary_df[final_summary_df["model"] == model_name].iloc[0]
        prefix = "rf" if model_name == PRIMARY_MODEL else "logit"
        final_comparison_row[f"{prefix}_delta_log_loss_vs_baseline"] = float(model_row["delta_log_loss_vs_baseline"])
        final_comparison_row[f"{prefix}_delta_brier_vs_baseline"] = float(model_row["delta_brier_score_vs_baseline"])
        final_comparison_row[f"{prefix}_delta_roc_auc_vs_baseline"] = float(model_row["delta_roc_auc_vs_baseline"])
        final_comparison_row[f"{prefix}_delta_accuracy_vs_baseline"] = float(model_row["delta_accuracy_vs_baseline"])

    metrics_frames.append(final_summary_df)
    fold_frames.append(final_fold_df)
    prediction_frames.append(final_predictions_df)
    model_comparison_rows.append(final_comparison_row)
    side_by_side_frames.append(final_side_by_side)

    metrics_summary_df = pd.concat(metrics_frames, ignore_index=True)
    fold_level_metrics_df = pd.concat(fold_frames, ignore_index=True)
    all_predictions_df = pd.concat(prediction_frames, ignore_index=True)
    model_comparison_df = pd.DataFrame(model_comparison_rows)
    side_by_side_predictions_df = pd.concat(side_by_side_frames, ignore_index=True)

    final_predictions_df[final_predictions_df["model"] == PRIMARY_MODEL].to_csv(OUTPUT_DIR / "predictions_rf.csv", index=False)
    final_predictions_df[final_predictions_df["model"] == SECONDARY_MODEL].to_csv(
        OUTPUT_DIR / "predictions_logit.csv",
        index=False,
    )
    metrics_summary_df.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    fold_level_metrics_df.to_csv(OUTPUT_DIR / "fold_level_metrics.csv", index=False)
    model_comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    side_by_side_predictions_df.to_csv(OUTPUT_DIR / "model_prediction_side_by_side.csv", index=False)
    all_predictions_df.to_csv(OUTPUT_DIR / "all_experiment_predictions.csv", index=False)
    candidate_results_df.to_csv(OUTPUT_DIR / "candidate_feature_results.csv", index=False)
    pd.DataFrame(step_records).to_csv(OUTPUT_DIR / "final_feature_build_steps.csv", index=False)

    extract_logistic_coefficients(df, final_features, splits).to_csv(
        OUTPUT_DIR / "final_logistic_coefficients.csv",
        index=False,
    )
    for model_name in (PRIMARY_MODEL,):
        extract_tree_importances(df, final_features, splits, model_name).to_csv(
            OUTPUT_DIR / f"final_{model_name}_feature_importance.csv",
            index=False,
        )
    extract_tree_importances(df, baseline_features, splits, PRIMARY_MODEL).to_csv(
        OUTPUT_DIR / "baseline_random_forest_feature_importance.csv",
        index=False,
    )
    extract_logistic_coefficients(df, baseline_features, splits).to_csv(
        OUTPUT_DIR / "baseline_logistic_coefficients.csv",
        index=False,
    )

    top_refined_experiments = model_comparison_df[
        model_comparison_df["experiment_name"].isin(["final_refined", "baseline_anchor"])
    ]
    permutation_frames: list[pd.DataFrame] = []
    for experiment_name in ["baseline_anchor", "final_refined"]:
        experiment_row = model_comparison_df[model_comparison_df["experiment_name"] == experiment_name].iloc[0]
        feature_subset = baseline_features if experiment_name == "baseline_anchor" else final_features
        permutation_df = extract_permutation_importance(
            df=df,
            features=feature_subset,
            splits=splits,
            model_name=PRIMARY_MODEL,
        ).rename(columns={f"{PRIMARY_MODEL}_permutation": "permutation_importance"})
        permutation_df.insert(0, "feature_subset", experiment_name)
        permutation_df.insert(0, "model", PRIMARY_MODEL)
        permutation_frames.append(permutation_df)
    pd.concat(permutation_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "random_forest_permutation_importance_comparison.csv",
        index=False,
    )

    write_feature_list_md(
        OUTPUT_DIR / "feature_list_used.md",
        baseline_features=baseline_features,
        candidate_descriptions=candidate_descriptions,
        candidate_results_df=candidate_results_df,
        final_features=final_features,
    )

    model_descriptions = {
        model_name: describe_model_factory(model_factory)
        for model_name, model_factory in model_factories.items()
        if model_name in {PRIMARY_MODEL, SECONDARY_MODEL}
    }
    ledger_rows: list[dict[str, object]] = []
    for row in metrics_summary_df.itertuples():
        ledger_rows.append(
            {
                "experiment_name": row.experiment_name,
                "model": row.model,
                "feature_subset": row.feature_subset,
                "feature_count": int(row.feature_count),
                "added_features_json": row.added_features_json,
                "parameters_json": json.dumps(model_descriptions[row.model]["parameters"], sort_keys=True),
                "accuracy": float(row.accuracy),
                "roc_auc": float(row.roc_auc),
                "log_loss": float(row.log_loss),
                "brier_score": float(row.brier_score),
                "delta_log_loss_vs_baseline": float(getattr(row, "delta_log_loss_vs_baseline", 0.0)),
                "delta_brier_vs_baseline": float(getattr(row, "delta_brier_score_vs_baseline", 0.0)),
                "delta_roc_auc_vs_baseline": float(getattr(row, "delta_roc_auc_vs_baseline", 0.0)),
                "delta_accuracy_vs_baseline": float(getattr(row, "delta_accuracy_vs_baseline", 0.0)),
                "metrics_summary_path": str(OUTPUT_DIR / "metrics_summary.csv"),
                "fold_level_metrics_path": str(OUTPUT_DIR / "fold_level_metrics.csv"),
                "model_comparison_path": str(OUTPUT_DIR / "model_comparison.csv"),
                "feature_list_path": str(OUTPUT_DIR / "feature_list_used.md"),
                "report_path": str(OUTPUT_DIR / "model_refinement_report_2.md"),
            }
        )
    pd.DataFrame(ledger_rows).to_csv(OUTPUT_DIR / "experiment_ledger.csv", index=False)

    report_text = render_report(
        baseline_summary_df=baseline_summary_df,
        final_summary_df=final_summary_df,
        candidate_results_df=candidate_results_df,
        model_comparison_df=model_comparison_df,
        final_features=final_features,
        baseline_features=baseline_features,
    )
    (OUTPUT_DIR / "model_refinement_report_2.md").write_text(report_text, encoding="utf-8")

    with (OUTPUT_DIR / "experiment_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_path": str(DATASET_PATH),
                "baseline_features": baseline_features,
                "candidate_descriptions": candidate_descriptions,
                "final_features": final_features,
                "model_descriptions": model_descriptions,
            },
            handle,
            indent=2,
        )

    print(report_text)
    print()
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
