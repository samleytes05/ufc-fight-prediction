from __future__ import annotations

import json
from pathlib import Path

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling" / "pruning"
BASELINE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling" / "baseline_experiments"
BASELINE_METRICS_PATH = BASELINE_OUTPUT_DIR / "metrics_summary.csv"
TOP_FEATURE_COUNT = 12

CONSERVATIVE_DROP = [
    "height_diff",
    "career_sub_rate_diff",
    "career_ko_rate_diff",
    "l3_td_def_diff",
    "B_elo",
    "career_control_share_diff",
]

REDUNDANT_DROP = [
    "l3_offense_defense_ratio_diff",
    "l3_adjusted_sig_strike_diff",
    "l3_sig_str_absorbed_per_round_diff",
]


def load_prior_importance() -> pd.DataFrame:
    return pd.read_csv(BASELINE_OUTPUT_DIR / "feature_importance_summary.csv")


def build_pruning_feature_sets(feature_columns: list[str], importance_df: pd.DataFrame) -> dict[str, list[str]]:
    all_features = list(feature_columns)
    conservative = [feature for feature in all_features if feature not in CONSERVATIVE_DROP]
    redundant = [feature for feature in all_features if feature not in REDUNDANT_DROP]
    combined = [feature for feature in all_features if feature not in set(CONSERVATIVE_DROP + REDUNDANT_DROP)]
    top_features = [
        feature
        for feature in importance_df["feature"].tolist()
        if feature in all_features
    ][:TOP_FEATURE_COUNT]
    return {
        "all_features": all_features,
        "pruned_conservative": conservative,
        "pruned_redundant": redundant,
        "pruned_combined": combined,
        "top_features_only": top_features,
    }


def subset_definition_rows(feature_sets: dict[str, list[str]], all_features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    full_set = set(all_features)
    for subset_name, features in feature_sets.items():
        included = list(features)
        removed = [feature for feature in all_features if feature not in set(features)]
        rows.append(
            {
                "feature_subset": subset_name,
                "feature_count": len(included),
                "included_features_json": json.dumps(included),
                "removed_features_json": json.dumps(removed),
            }
        )
    return pd.DataFrame(rows)


def write_feature_sets_markdown(feature_sets: dict[str, list[str]], all_features: list[str], output_path: Path) -> None:
    lines = [
        "# Feature Sets Used",
        "",
        f"- Dataset: `{DATASET_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Total available features: {len(all_features)}",
        "",
    ]
    for subset_name, features in feature_sets.items():
        removed = [feature for feature in all_features if feature not in set(features)]
        lines.extend(
            [
                f"## {subset_name}",
                f"- Feature count: {len(features)}",
                f"- Included: {', '.join(features)}",
                f"- Removed: {', '.join(removed) if removed else 'None'}",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def rank_results(results_df: pd.DataFrame) -> pd.DataFrame:
    return results_df.sort_values(
        ["log_loss", "brier_score", "roc_auc", "accuracy"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def compare_to_baseline(
    pruning_results: pd.DataFrame,
    baseline_results: pd.DataFrame,
) -> pd.DataFrame:
    baseline_reference = baseline_results[baseline_results["feature_subset"] == "all_features"][
        ["model", "accuracy", "roc_auc", "log_loss", "brier_score"]
    ].rename(
        columns={
            "accuracy": "baseline_accuracy",
            "roc_auc": "baseline_roc_auc",
            "log_loss": "baseline_log_loss",
            "brier_score": "baseline_brier_score",
        }
    )
    merged = pruning_results.merge(baseline_reference, on="model", how="left")
    merged["delta_accuracy"] = merged["accuracy"] - merged["baseline_accuracy"]
    merged["delta_roc_auc"] = merged["roc_auc"] - merged["baseline_roc_auc"]
    merged["delta_log_loss"] = merged["log_loss"] - merged["baseline_log_loss"]
    merged["delta_brier_score"] = merged["brier_score"] - merged["baseline_brier_score"]
    return merged


def performance_label(delta_log_loss: float, tolerance: float = 0.0015) -> str:
    if delta_log_loss <= -tolerance:
        return "improved"
    if delta_log_loss >= tolerance:
        return "worsened"
    return "neutral"


def render_report(
    df: pd.DataFrame,
    ranked_results: pd.DataFrame,
    comparison_df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    baseline_results: pd.DataFrame,
    top_pruned_permutation_df: pd.DataFrame,
) -> str:
    target_rate = df[TARGET_COLUMN].mean()
    best_overall = ranked_results.iloc[0]
    pruned_only = ranked_results[ranked_results["feature_subset"] != "all_features"].reset_index(drop=True)
    best_pruned = pruned_only.iloc[0]
    baseline_all = baseline_results[baseline_results["feature_subset"] == "all_features"].copy()
    best_baseline_full = baseline_all.sort_values(
        ["log_loss", "brier_score", "roc_auc", "accuracy"],
        ascending=[True, True, False, False],
    ).iloc[0]

    model_comparison_rows: list[str] = []
    for model_name in ranked_results["model"].drop_duplicates():
        model_rows = comparison_df[comparison_df["model"] == model_name].copy()
        best_model_row = model_rows.sort_values(
            ["log_loss", "brier_score", "roc_auc", "accuracy"],
            ascending=[True, True, False, False],
        ).iloc[0]
        model_comparison_rows.append(
            (
                f"- {model_name}: best pruning set was `{best_model_row['feature_subset']}` "
                f"with log loss change {best_model_row['delta_log_loss']:+.4f} vs the original full-feature baseline, "
                f"so pruning {performance_label(float(best_model_row['delta_log_loss']))} this model."
            )
        )

    safe_to_drop = sorted(set(CONSERVATIVE_DROP + REDUNDANT_DROP) - set(best_pruned["removed_features"]))
    subset_table = ranked_results[
        ["model", "feature_subset", "feature_count", "accuracy", "roc_auc", "log_loss", "brier_score"]
    ].to_string(index=False)
    comparison_table = comparison_df[
        [
            "model",
            "feature_subset",
            "delta_log_loss",
            "delta_brier_score",
            "delta_roc_auc",
            "delta_accuracy",
        ]
    ].sort_values(
        ["delta_log_loss", "delta_brier_score", "delta_roc_auc", "delta_accuracy"],
        ascending=[True, True, False, False],
    ).to_string(index=False)
    permutation_table = (
        top_pruned_permutation_df.head(16).to_string(index=False)
        if not top_pruned_permutation_df.empty
        else "No pruned permutation importance results saved."
    )

    lines = [
        "# Phase 2 Pruning Report",
        "",
        "## Purpose",
        "- Measure whether conservative and aggressive feature pruning improves or hurts baseline models under the same time-safe validation framework as the first Phase 2 suite.",
        "",
        "## Dataset",
        f"- Source: `{DATASET_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Shape used: {df.shape[0]} fights x {len(feature_sets['all_features'])} predictive features",
        f"- Target column: `{TARGET_COLUMN}`",
        f"- Target distribution: A-side win rate = {target_rate:.3f}",
        "",
        "## Validation Setup",
        "- Same expanding-window walk-forward setup as the prior experiment suite.",
        "- Same chronological folds for every pruning experiment.",
        "- No random shuffling, no dataset rebuild, and no heavy tuning.",
        "",
        "## Feature Subsets Tested",
        *[f"- `{name}`: {len(features)} features" for name, features in feature_sets.items()],
        "",
        "## Models Tested",
        "- logistic_regression",
        "- random_forest",
        "- xgboost",
        "",
        "## Results",
        subset_table,
        "",
        "## Comparison Versus Original Full-Feature Baseline",
        (
            f"- Best original full-feature baseline: `{best_baseline_full['model']}` on `all_features` "
            f"(log_loss={best_baseline_full['log_loss']:.4f}, brier={best_baseline_full['brier_score']:.4f}, "
            f"roc_auc={best_baseline_full['roc_auc']:.4f}, accuracy={best_baseline_full['accuracy']:.4f})"
        ),
        (
            f"- Best pruning result overall: `{best_pruned['model']}` on `{best_pruned['feature_subset']}` "
            f"(log_loss={best_pruned['log_loss']:.4f}, delta vs original full-feature baseline={best_pruned['delta_log_loss']:+.4f})"
        ),
        (
            f"- Overall best result in this pruning suite: `{best_overall['model']}` on `{best_overall['feature_subset']}` "
            f"(log_loss={best_overall['log_loss']:.4f}, brier={best_overall['brier_score']:.4f})"
        ),
        comparison_table,
        "",
        "## Model-Level Pruning Effects",
        *model_comparison_rows,
        "",
        "## Top Pruned-Experiment Permutation Importance",
        permutation_table,
        "",
        "## Recommendation",
        (
            f"- Pruning strategy that worked best: `{best_pruned['feature_subset']}` for `{best_pruned['model']}`."
        ),
        (
            "- Simpler models benefited more from pruning than tree models."
            if comparison_df.groupby("model")["delta_log_loss"].min().get("logistic_regression", 0.0)
            < comparison_df.groupby("model")["delta_log_loss"].min().get("random_forest", 0.0)
            else "- Tree models benefited at least as much as simpler linear models from pruning."
        ),
        (
            "- Adopt the pruned baseline going forward."
            if float(best_pruned["delta_log_loss"]) < 0
            else "- Keep the full set as the default baseline; use pruning as a secondary simplification path only."
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, feature_columns = load_dataset()
    splits = build_expanding_splits(df)
    model_factories = build_model_factories()
    baseline_results = pd.read_csv(BASELINE_METRICS_PATH)
    importance_df = load_prior_importance()

    feature_sets = build_pruning_feature_sets(feature_columns, importance_df)
    feature_set_defs_df = subset_definition_rows(feature_sets, feature_columns)
    feature_set_defs_df.to_csv(OUTPUT_DIR / "feature_sets_used.csv", index=False)
    write_feature_sets_markdown(feature_sets, feature_columns, OUTPUT_DIR / "feature_sets_used.md")

    results_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    fold_metric_frames: list[pd.DataFrame] = []
    for subset_name, subset_features in feature_sets.items():
        for model_name, model_factory in model_factories.items():
            summary, predictions, fold_metrics = evaluate_model_subset(
                df=df,
                feature_subset_name=subset_name,
                feature_subset=subset_features,
                model_name=model_name,
                model_factory=model_factory,
                splits=splits,
            )
            results_rows.append(summary)
            prediction_frames.append(predictions)
            fold_metric_frames.append(fold_metrics)

    results_df = pd.DataFrame(results_rows)
    all_predictions_df = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics_df = pd.concat(fold_metric_frames, ignore_index=True)
    comparison_df = compare_to_baseline(results_df, baseline_results)
    feature_removed_lookup = {
        subset_name: [feature for feature in feature_columns if feature not in set(features)]
        for subset_name, features in feature_sets.items()
    }
    comparison_df["removed_features"] = comparison_df["feature_subset"].map(feature_removed_lookup)
    ranked_results = rank_results(comparison_df)

    pruned_ranked = ranked_results[ranked_results["feature_subset"] != "all_features"].reset_index(drop=True)
    top_pruned_rows = pruned_ranked.head(2)

    top_pruned_permutation_frames: list[pd.DataFrame] = []
    for row in top_pruned_rows.itertuples():
        permutation_df = extract_permutation_importance(
            df=df,
            features=feature_sets[row.feature_subset],
            splits=splits,
            model_name=row.model,
        ).rename(columns={f"{row.model}_permutation": "permutation_importance"})
        permutation_df.insert(0, "feature_subset", row.feature_subset)
        permutation_df.insert(0, "model", row.model)
        top_pruned_permutation_frames.append(permutation_df)
    top_pruned_permutation_df = pd.concat(top_pruned_permutation_frames, ignore_index=True)

    best_pruned_models = top_pruned_rows[["model", "feature_subset"]].drop_duplicates()
    for row in best_pruned_models.itertuples():
        subset_features = feature_sets[row.feature_subset]
        if row.model == "logistic_regression":
            extract_logistic_coefficients(df, subset_features, splits).to_csv(
                OUTPUT_DIR / f"{row.model}_{row.feature_subset}_coefficients.csv",
                index=False,
            )
        elif row.model in {"random_forest", "xgboost"}:
            extract_tree_importances(df, subset_features, splits, row.model).to_csv(
                OUTPUT_DIR / f"{row.model}_{row.feature_subset}_feature_importance.csv",
                index=False,
            )

    report_text = render_report(
        df=df,
        ranked_results=ranked_results,
        comparison_df=comparison_df,
        feature_sets=feature_sets,
        baseline_results=baseline_results,
        top_pruned_permutation_df=top_pruned_permutation_df,
    )

    ranked_results.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    fold_metrics_df.to_csv(OUTPUT_DIR / "fold_level_metrics.csv", index=False)
    all_predictions_df.to_csv(OUTPUT_DIR / "all_model_predictions.csv", index=False)
    top_pruned_permutation_df.to_csv(OUTPUT_DIR / "top_pruned_permutation_importance.csv", index=False)

    best_pruned_prediction_df = all_predictions_df.merge(
        top_pruned_rows[["model", "feature_subset"]],
        on=["model", "feature_subset"],
        how="inner",
    )
    best_pruned_prediction_df.to_csv(OUTPUT_DIR / "top_pruned_predictions.csv", index=False)

    model_descriptions = {
        model_name: describe_model_factory(model_factory)
        for model_name, model_factory in model_factories.items()
    }
    ledger_rows: list[dict[str, object]] = []
    for row in ranked_results.itertuples():
        ledger_rows.append(
            {
                "model": row.model,
                "feature_subset": row.feature_subset,
                "feature_count": int(row.feature_count),
                "parameters_json": json.dumps(model_descriptions[row.model]["parameters"], sort_keys=True),
                "accuracy": float(row.accuracy),
                "roc_auc": float(row.roc_auc),
                "log_loss": float(row.log_loss),
                "brier_score": float(row.brier_score),
                "delta_log_loss_vs_prior_all_features": float(row.delta_log_loss),
                "delta_brier_vs_prior_all_features": float(row.delta_brier_score),
                "delta_roc_auc_vs_prior_all_features": float(row.delta_roc_auc),
                "delta_accuracy_vs_prior_all_features": float(row.delta_accuracy),
                "pruning_effect": performance_label(float(row.delta_log_loss)),
                "removed_features_json": json.dumps(feature_removed_lookup[row.feature_subset]),
                "fold_metrics_path": str(OUTPUT_DIR / "fold_level_metrics.csv"),
                "metrics_summary_path": str(OUTPUT_DIR / "metrics_summary.csv"),
                "feature_sets_path": str(OUTPUT_DIR / "feature_sets_used.md"),
                "predictions_path": str(OUTPUT_DIR / "top_pruned_predictions.csv"),
                "report_path": str(OUTPUT_DIR / "model_comparison_report_2.md"),
            }
        )
    pd.DataFrame(ledger_rows).to_csv(OUTPUT_DIR / "experiment_ledger.csv", index=False)

    with (OUTPUT_DIR / "experiment_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_path": str(DATASET_PATH),
                "baseline_metrics_path": str(BASELINE_METRICS_PATH),
                "output_dir": str(OUTPUT_DIR),
                "feature_sets": feature_sets,
                "conservative_drop": CONSERVATIVE_DROP,
                "redundant_drop": REDUNDANT_DROP,
                "model_descriptions": model_descriptions,
            },
            handle,
            indent=2,
        )

    (OUTPUT_DIR / "model_comparison_report_2.md").write_text(report_text, encoding="utf-8")

    print(report_text)
    print()
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
