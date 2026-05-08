from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from model import build_model_factories


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "historical_backfill" / "ufc_rebuilt_features_scraped.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling" / "baseline_experiments"
RANDOM_STATE = 42
N_SPLITS = 5
MIN_TRAIN_FRACTION = 0.50
TOP_N_FEATURES = 12
PERMUTATION_REPEATS = 5
PERMUTATION_SAMPLE_LIMIT = 1200

IDENTIFIER_COLUMNS = ["fight_id", "fight_order"]
TARGET_COLUMN = "target_A_win"
PHYSICAL_FEATURES = ["age_diff", "height_diff", "reach_diff"]
ELO_FEATURES = [
    "A_elo",
    "B_elo",
    "elo_diff",
    "A_avg_elo_last3_opp",
    "B_avg_elo_last3_opp",
    "avg_elo_last3_opp_diff",
]


def load_dataset(path: Path = DATASET_PATH) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    df = df.sort_values("fight_order").reset_index(drop=True)
    feature_columns = [
        col
        for col in df.columns
        if col not in IDENTIFIER_COLUMNS + [TARGET_COLUMN]
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    return df, feature_columns


def build_expanding_splits(
    df: pd.DataFrame,
    n_splits: int = N_SPLITS,
    min_train_fraction: float = MIN_TRAIN_FRACTION,
) -> list[tuple[np.ndarray, np.ndarray]]:
    total_rows = len(df)
    min_train_rows = max(1, int(total_rows * min_train_fraction))
    remaining_rows = total_rows - min_train_rows
    if remaining_rows < n_splits:
        raise ValueError("Not enough rows for the requested time-based folds.")

    base_fold_size = remaining_rows // n_splits
    remainder = remaining_rows % n_splits
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    train_end = min_train_rows

    for fold_idx in range(n_splits):
        fold_size = base_fold_size + (1 if fold_idx < remainder else 0)
        test_end = train_end + fold_size
        train_index = np.arange(0, train_end)
        test_index = np.arange(train_end, test_end)
        if len(test_index) == 0:
            continue
        splits.append((train_index, test_index))
        train_end = test_end

    return splits


def split_train_calibration(train_index: np.ndarray, calibration_fraction: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    calibration_size = max(200, int(len(train_index) * calibration_fraction))
    calibration_size = min(calibration_size, len(train_index) - 1)
    fit_index = train_index[:-calibration_size]
    calibration_index = train_index[-calibration_size:]
    if len(fit_index) < 1 or len(calibration_index) < 1:
        raise ValueError("Unable to create a time-safe calibration split.")
    return fit_index, calibration_index

def metric_dict(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def feature_group_map(feature_columns: list[str]) -> dict[str, list[str]]:
    recent = [col for col in feature_columns if col.startswith("l3_")]
    career = [col for col in feature_columns if col.startswith("career_")]
    matchup_context = [col for col in feature_columns if col in {"win_streak_diff", "experience_advantage_ratio_log"}]
    differential_only = [col for col in feature_columns if col.endswith("_diff") or col.endswith("_log")]
    core_statistical = [
        col
        for col in feature_columns
        if col not in PHYSICAL_FEATURES + ELO_FEATURES
        and col not in matchup_context
    ]
    recent_plus_elo = recent + ELO_FEATURES
    career_recent_statistical = career + recent + matchup_context
    career_recent_physical = career + recent + PHYSICAL_FEATURES + matchup_context
    return {
        "all_features": feature_columns,
        "differential_only": [col for col in differential_only if col in feature_columns],
        "non_physical": [col for col in feature_columns if col not in PHYSICAL_FEATURES],
        "no_elo": [col for col in feature_columns if col not in ELO_FEATURES],
        "core_statistical_differentials": core_statistical,
        "recent_form_plus_elo": [col for col in recent_plus_elo if col in feature_columns],
        "career_recent_statistical": [col for col in career_recent_statistical if col in feature_columns],
        "career_recent_physicals": [col for col in career_recent_physical if col in feature_columns],
        "physical_only": [col for col in PHYSICAL_FEATURES if col in feature_columns],
        "elo_only": [col for col in ELO_FEATURES if col in feature_columns],
    }


def evaluate_model_subset(
    df: pd.DataFrame,
    feature_subset_name: str,
    feature_subset: list[str],
    model_name: str,
    model_factory,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    fold_metrics: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold_number, (train_index, test_index) in enumerate(splits, start=1):
        x_train = df.iloc[train_index][feature_subset]
        y_train = df.iloc[train_index][TARGET_COLUMN]
        x_test = df.iloc[test_index][feature_subset]
        y_test = df.iloc[test_index][TARGET_COLUMN]

        model = model_factory()
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        metrics = metric_dict(y_test, probabilities)
        fold_metrics.append(
            {
                "model": model_name,
                "feature_subset": feature_subset_name,
                "fold": fold_number,
                "train_size": int(len(train_index)),
                "test_size": int(len(test_index)),
                **metrics,
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "feature_subset": feature_subset_name,
                    "fold": fold_number,
                    "fight_id": df.iloc[test_index]["fight_id"].values,
                    "fight_order": df.iloc[test_index]["fight_order"].values,
                    "y_true": y_test.to_numpy(),
                    "predicted_probability": probabilities,
                }
            )
        )

    fold_metrics_df = pd.DataFrame(fold_metrics)
    avg_metrics = fold_metrics_df[["accuracy", "roc_auc", "log_loss", "brier_score"]].mean().to_dict()
    summary = {
        "model": model_name,
        "feature_subset": feature_subset_name,
        "feature_count": len(feature_subset),
        **{metric: float(value) for metric, value in avg_metrics.items()},
    }
    return summary, pd.concat(prediction_frames, ignore_index=True), fold_metrics_df


def extract_logistic_coefficients(
    df: pd.DataFrame,
    features: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    model_factory = build_model_factories()["logistic_regression"]

    for fold_number, (train_index, _) in enumerate(splits, start=1):
        model = model_factory()
        x_train = df.iloc[train_index][features]
        y_train = df.iloc[train_index][TARGET_COLUMN]
        model.fit(x_train, y_train)
        coefficients = model.named_steps["classifier"].coef_[0]
        for feature, coefficient in zip(features, coefficients, strict=False):
            rows.append(
                {
                    "fold": fold_number,
                    "feature": feature,
                    "logistic_coefficient": float(coefficient),
                    "logistic_abs_coefficient": float(abs(coefficient)),
                }
            )

    coef_df = pd.DataFrame(rows)
    return (
        coef_df.groupby("feature", as_index=False)
        .agg(
            logistic_coefficient=("logistic_coefficient", "mean"),
            logistic_abs_coefficient=("logistic_abs_coefficient", "mean"),
        )
        .sort_values("logistic_abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def extract_tree_importances(
    df: pd.DataFrame,
    features: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    model_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    model_factory = build_model_factories()[model_name]

    for fold_number, (train_index, _) in enumerate(splits, start=1):
        model = model_factory()
        x_train = df.iloc[train_index][features]
        y_train = df.iloc[train_index][TARGET_COLUMN]
        model.fit(x_train, y_train)
        importances = model.named_steps["classifier"].feature_importances_
        for feature, importance in zip(features, importances, strict=False):
            rows.append({"fold": fold_number, "feature": feature, f"{model_name}_importance": float(importance)})

    importance_column = f"{model_name}_importance"
    return (
        pd.DataFrame(rows)
        .groupby("feature", as_index=False)
        .agg(**{importance_column: (importance_column, "mean")})
        .sort_values(importance_column, ascending=False)
        .reset_index(drop=True)
    )


def extract_permutation_importance(
    df: pd.DataFrame,
    features: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    model_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    model_factory = build_model_factories()[model_name]

    for fold_number, (train_index, test_index) in enumerate(splits, start=1):
        model = model_factory()
        x_train = df.iloc[train_index][features]
        y_train = df.iloc[train_index][TARGET_COLUMN]
        x_test = df.iloc[test_index][features]
        y_test = df.iloc[test_index][TARGET_COLUMN]
        if len(x_test) > PERMUTATION_SAMPLE_LIMIT:
            x_test = x_test.iloc[:PERMUTATION_SAMPLE_LIMIT]
            y_test = y_test.iloc[:PERMUTATION_SAMPLE_LIMIT]
        model.fit(x_train, y_train)
        result = permutation_importance(
            model,
            x_test,
            y_test,
            scoring="neg_log_loss",
            n_repeats=PERMUTATION_REPEATS,
            random_state=RANDOM_STATE + fold_number,
            n_jobs=1,
        )
        for feature, importance in zip(features, result.importances_mean, strict=False):
            rows.append({"fold": fold_number, "feature": feature, f"{model_name}_permutation": float(importance)})

    importance_column = f"{model_name}_permutation"
    return (
        pd.DataFrame(rows)
        .groupby("feature", as_index=False)
        .agg(**{importance_column: (importance_column, "mean")})
        .sort_values(importance_column, ascending=False)
        .reset_index(drop=True)
    )


def describe_model_factory(model_factory) -> dict[str, object]:
    model = model_factory()
    classifier = model.named_steps["classifier"]
    keep_keys = [
        "max_iter",
        "penalty",
        "solver",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
    ]
    params = classifier.get_params()
    compact_params = {key: params[key] for key in keep_keys if key in params}
    return {
        "pipeline_type": type(model).__name__,
        "classifier_type": type(classifier).__name__,
        "parameters": compact_params,
    }


def combine_feature_importance_views(
    logistic_df: pd.DataFrame,
    rf_df: pd.DataFrame,
    xgb_df: pd.DataFrame | None,
    logreg_perm_df: pd.DataFrame,
    xgb_perm_df: pd.DataFrame | None,
) -> pd.DataFrame:
    merged = logistic_df.merge(rf_df, on="feature", how="outer")
    if xgb_df is not None:
        merged = merged.merge(xgb_df, on="feature", how="outer")
    merged = merged.merge(logreg_perm_df, on="feature", how="outer")
    if xgb_perm_df is not None:
        merged = merged.merge(xgb_perm_df, on="feature", how="outer")

    rank_columns = [
        "logistic_abs_coefficient",
        "random_forest_importance",
        "logistic_regression_permutation",
    ]
    if "xgboost_importance" in merged.columns:
        rank_columns.append("xgboost_importance")
    if "xgboost_permutation" in merged.columns:
        rank_columns.append("xgboost_permutation")

    for column in rank_columns:
        merged[f"{column}_rank"] = merged[column].rank(ascending=False, method="average")
    merged["average_rank"] = merged[[f"{column}_rank" for column in rank_columns]].mean(axis=1)
    return merged.sort_values("average_rank").reset_index(drop=True)


def top_feature_subset(importance_df: pd.DataFrame, feature_columns: list[str], top_n: int = TOP_N_FEATURES) -> list[str]:
    candidates = [feature for feature in importance_df["feature"].tolist() if feature in feature_columns]
    return candidates[:top_n]


def summarize_redundancy(df: pd.DataFrame, feature_columns: list[str], threshold: float = 0.85) -> pd.DataFrame:
    corr_matrix = df[feature_columns].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    rows: list[dict[str, float | str]] = []
    for left in upper.columns:
        for right, value in upper[left].dropna().items():
            if value >= threshold:
                rows.append({"feature_left": left, "feature_right": right, "abs_correlation": float(value)})
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def calibration_trial(
    df: pd.DataFrame,
    features: list[str],
    model_name: str,
    model_factory,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold_number, (train_index, test_index) in enumerate(splits, start=1):
        fit_index, calibration_index = split_train_calibration(train_index)
        x_fit = df.iloc[fit_index][features]
        y_fit = df.iloc[fit_index][TARGET_COLUMN]
        x_cal = df.iloc[calibration_index][features]
        y_cal = df.iloc[calibration_index][TARGET_COLUMN]
        x_test = df.iloc[test_index][features]
        y_test = df.iloc[test_index][TARGET_COLUMN]

        base_model = model_factory()
        base_model.fit(x_fit, y_fit)
        calibration_probabilities = base_model.predict_proba(x_cal)[:, 1]
        base_probabilities = base_model.predict_proba(x_test)[:, 1]

        sigmoid_model = LogisticRegression(random_state=RANDOM_STATE)
        sigmoid_model.fit(calibration_probabilities.reshape(-1, 1), y_cal)
        sigmoid_probabilities = sigmoid_model.predict_proba(base_probabilities.reshape(-1, 1))[:, 1]

        isotonic_model = IsotonicRegression(out_of_bounds="clip")
        isotonic_model.fit(calibration_probabilities, y_cal)
        isotonic_probabilities = isotonic_model.predict(base_probabilities)

        for calibration_name, probabilities in {
            "uncalibrated": base_probabilities,
            "platt_sigmoid": sigmoid_probabilities,
            "isotonic": isotonic_probabilities,
        }.items():
            rows.append(
                {
                    "model": model_name,
                    "fold": fold_number,
                    "calibration_method": calibration_name,
                    **metric_dict(y_test, probabilities),
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "fold": fold_number,
                        "calibration_method": calibration_name,
                        "fight_id": df.iloc[test_index]["fight_id"].values,
                        "fight_order": df.iloc[test_index]["fight_order"].values,
                        "y_true": y_test.to_numpy(),
                        "predicted_probability": probabilities,
                    }
                )
            )

    return pd.DataFrame(rows), pd.concat(prediction_frames, ignore_index=True)


def reliability_table(prediction_df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    frame = prediction_df.copy()
    frame["probability_bin"] = pd.cut(frame["predicted_probability"], bins=n_bins, include_lowest=True, duplicates="drop")
    return (
        frame.groupby(["model", "calibration_method", "probability_bin"], observed=False, as_index=False)
        .agg(
            predicted_probability=("predicted_probability", "mean"),
            actual_win_rate=("y_true", "mean"),
            count=("y_true", "size"),
        )
        .query("count > 0")
    )


def best_configuration(results_df: pd.DataFrame) -> pd.Series:
    ranked = results_df.sort_values(
        ["log_loss", "roc_auc", "brier_score", "accuracy"],
        ascending=[True, False, True, False],
    ).reset_index(drop=True)
    return ranked.iloc[0]


def top_permutation_experiments(results_df: pd.DataFrame, top_n: int = 2) -> pd.DataFrame:
    return results_df.sort_values(
        ["log_loss", "roc_auc", "brier_score", "accuracy"],
        ascending=[True, False, True, False],
    ).head(top_n)


def json_ready_dict(data: dict[str, object]) -> dict[str, object]:
    ready: dict[str, object] = {}
    for key, value in data.items():
        if isinstance(value, (np.floating, np.integer)):
            ready[key] = value.item()
        else:
            ready[key] = value
    return ready


def render_report(
    df: pd.DataFrame,
    feature_columns: list[str],
    results_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    redundancy_df: pd.DataFrame,
    calibration_summary_df: pd.DataFrame,
    permutation_top_df: pd.DataFrame,
    best_row: pd.Series,
    top_features: list[str],
) -> str:
    target_rate = df[TARGET_COLUMN].mean()
    best_model_results = results_df[results_df["model"] == best_row["model"]].sort_values(
        ["log_loss", "roc_auc"], ascending=[True, False]
    )
    elo_rows = results_df[results_df["feature_subset"].isin(["all_features", "no_elo"])]
    physical_rows = results_df[results_df["feature_subset"].isin(["all_features", "non_physical"])]
    elo_effect = ""
    physical_effect = ""
    if len(elo_rows["feature_subset"].unique()) == 2:
        all_elo = elo_rows[elo_rows["feature_subset"] == "all_features"]["log_loss"].mean()
        no_elo = elo_rows[elo_rows["feature_subset"] == "no_elo"]["log_loss"].mean()
        elo_effect = f"ELO helps when included: average log loss improves by {no_elo - all_elo:+.4f} versus dropping ELO."
    if len(physical_rows["feature_subset"].unique()) == 2:
        all_physical = physical_rows[physical_rows["feature_subset"] == "all_features"]["log_loss"].mean()
        non_physical = physical_rows[physical_rows["feature_subset"] == "non_physical"]["log_loss"].mean()
        physical_effect = f"Physical features change average log loss by {non_physical - all_physical:+.4f} versus removing them."

    weakest_features = importance_df.sort_values("average_rank", ascending=False).head(6)["feature"].tolist()
    redundant_preview = redundancy_df.head(8)
    calibration_best = calibration_summary_df.sort_values(
        ["model", "log_loss", "brier_score"], ascending=[True, True, True]
    )
    model_table = results_df.sort_values(["log_loss", "roc_auc"], ascending=[True, False]).to_string(index=False)
    calibration_table_md = calibration_best.to_string(index=False)
    permutation_preview = permutation_top_df.head(12).to_string(index=False) if not permutation_top_df.empty else "No top-experiment permutation results saved."
    best_subset_overall = (
        results_df.groupby("feature_subset", as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            roc_auc=("roc_auc", "mean"),
            log_loss=("log_loss", "mean"),
            brier_score=("brier_score", "mean"),
        )
        .sort_values(["log_loss", "roc_auc"], ascending=[True, False])
        .iloc[0]
    )
    simpler_close = results_df[
        (results_df["feature_count"] < len(feature_columns))
        & (results_df["log_loss"] <= float(best_row["log_loss"]) + 0.005)
    ].sort_values(["log_loss", "roc_auc"], ascending=[True, False])

    lines = [
        "# Phase 2 Baseline Report",
        "",
        "## Dataset",
        f"- Source: `{DATASET_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Shape used: {df.shape[0]} fights x {len(feature_columns)} predictive features",
        f"- Target column: `{TARGET_COLUMN}`",
        f"- Chronology column: `fight_order` (the dataset does not include a fight date column)",
        f"- Target distribution: A-side win rate = {target_rate:.3f}",
        "",
        "## Validation",
        f"- Method: expanding-window walk-forward validation with {N_SPLITS} folds",
        f"- Initial training window: first {MIN_TRAIN_FRACTION:.0%} of fights",
        "- No random shuffling and no feature regeneration",
        "",
        "## Model Results",
        model_table,
        "",
        "## Baseline Recommendation",
        (
            f"- Best baseline model: `{best_row['model']}` on `{best_row['feature_subset']}` "
            f"(accuracy={best_row['accuracy']:.4f}, roc_auc={best_row['roc_auc']:.4f}, "
            f"log_loss={best_row['log_loss']:.4f}, brier={best_row['brier_score']:.4f})"
        ),
        (
            f"- Best feature subset overall across models: `{best_subset_overall['feature_subset']}` "
            f"(mean log_loss={best_subset_overall['log_loss']:.4f}, mean roc_auc={best_subset_overall['roc_auc']:.4f})"
        ),
        (
            f"- Best feature subset for the winning model: `{best_row['feature_subset']}` "
            f"with {int(best_row['feature_count'])} features"
        ),
        (
            f"- Best alternate setup inside the winning model family: "
            f"`{best_model_results.iloc[1]['feature_subset']}` "
            f"(log_loss={best_model_results.iloc[1]['log_loss']:.4f})"
            if len(best_model_results) > 1
            else "- No alternate setup available for the winning model family."
        ),
        f"- {elo_effect}" if elo_effect else "",
        f"- {physical_effect}" if physical_effect else "",
        (
            f"- Simpler subsets within 0.005 log loss of the best result: "
            f"{', '.join((simpler_close['model'] + ' / ' + simpler_close['feature_subset']).tolist())}"
            if not simpler_close.empty
            else "- No simpler subset finished within 0.005 log loss of the best result."
        ),
        "",
        "## Feature Signals",
        f"- Top individual features across coefficients, tree importance, and permutation importance: {', '.join(top_features)}",
        f"- Weakest features across the same combined ranking: {', '.join(weakest_features)}",
        "",
        "## Top-Experiment Permutation Importance",
        permutation_preview,
        "",
        "## Calibration",
        calibration_table_md,
        "",
        "## Redundancy Notes",
        (
            "- Strongest redundancy pairs: "
            + "; ".join(
                [
                    f"{row.feature_left} ~ {row.feature_right} ({row.abs_correlation:.3f})"
                    for row in redundant_preview.itertuples()
                ]
            )
            if not redundant_preview.empty
            else "- No feature pairs exceeded the redundancy threshold."
        ),
        "",
        "## Next Step",
        "- Carry forward the best baseline configuration above, then use the weakest and redundant features as the first candidates for ablation or cleanup before inventing new features.",
    ]
    return "\n".join(line for line in lines if line != "")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, feature_columns = load_dataset()
    splits = build_expanding_splits(df)
    group_map = feature_group_map(feature_columns)
    model_factories = build_model_factories()

    logistic_importance = extract_logistic_coefficients(df, feature_columns, splits)
    random_forest_importance = extract_tree_importances(df, feature_columns, splits, "random_forest")
    xgboost_importance = None
    if "xgboost" in model_factories:
        xgboost_importance = extract_tree_importances(df, feature_columns, splits, "xgboost")

    logreg_permutation = extract_permutation_importance(df, feature_columns, splits, "logistic_regression")
    xgboost_permutation = None
    if "xgboost" in model_factories:
        xgboost_permutation = extract_permutation_importance(df, feature_columns, splits, "xgboost")

    importance_df = combine_feature_importance_views(
        logistic_df=logistic_importance,
        rf_df=random_forest_importance,
        xgb_df=xgboost_importance,
        logreg_perm_df=logreg_permutation,
        xgb_perm_df=xgboost_permutation,
    )

    group_map["top_n_features"] = top_feature_subset(importance_df, feature_columns)

    results_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    fold_metric_frames: list[pd.DataFrame] = []
    for subset_name in [
        "all_features",
        "top_n_features",
        "differential_only",
        "non_physical",
        "no_elo",
        "core_statistical_differentials",
        "recent_form_plus_elo",
        "career_recent_statistical",
        "career_recent_physicals",
    ]:
        subset_features = group_map[subset_name]
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

    results_df = pd.DataFrame(results_rows).sort_values(["log_loss", "roc_auc"], ascending=[True, False]).reset_index(drop=True)
    all_predictions_df = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics_df = pd.concat(fold_metric_frames, ignore_index=True)
    best_row = best_configuration(results_df)
    top_experiment_rows = top_permutation_experiments(results_df, top_n=2)

    top_two_configs = results_df.head(2)
    calibration_rows: list[pd.DataFrame] = []
    calibration_predictions: list[pd.DataFrame] = []
    for row in top_two_configs.itertuples():
        model_factory = model_factories[row.model]
        calibration_metrics_df, calibration_prediction_df = calibration_trial(
            df=df,
            features=group_map[row.feature_subset],
            model_name=f"{row.model}__{row.feature_subset}",
            model_factory=model_factory,
            splits=splits,
        )
        calibration_rows.append(calibration_metrics_df)
        calibration_predictions.append(calibration_prediction_df)

    calibration_metrics_df = pd.concat(calibration_rows, ignore_index=True)
    calibration_summary_df = (
        calibration_metrics_df.groupby(["model", "calibration_method"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            roc_auc=("roc_auc", "mean"),
            log_loss=("log_loss", "mean"),
            brier_score=("brier_score", "mean"),
        )
        .sort_values(["model", "log_loss", "roc_auc"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    calibration_predictions_df = pd.concat(calibration_predictions, ignore_index=True)
    reliability_df = reliability_table(calibration_predictions_df)

    top_permutation_rows: list[pd.DataFrame] = []
    for row in top_experiment_rows.itertuples():
        permutation_df = extract_permutation_importance(
            df=df,
            features=group_map[row.feature_subset],
            splits=splits,
            model_name=row.model,
        ).rename(columns={f"{row.model}_permutation": "permutation_importance"})
        permutation_df.insert(0, "feature_subset", row.feature_subset)
        permutation_df.insert(0, "model", row.model)
        top_permutation_rows.append(permutation_df)
    top_experiment_permutation_df = pd.concat(top_permutation_rows, ignore_index=True)

    redundancy_df = summarize_redundancy(df, feature_columns)
    top_features = importance_df.head(12)["feature"].tolist()
    report_text = render_report(
        df=df,
        feature_columns=feature_columns,
        results_df=results_df,
        importance_df=importance_df,
        redundancy_df=redundancy_df,
        calibration_summary_df=calibration_summary_df,
        permutation_top_df=top_experiment_permutation_df,
        best_row=best_row,
        top_features=top_features,
    )

    results_df.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    fold_metrics_df.to_csv(OUTPUT_DIR / "fold_level_metrics.csv", index=False)
    all_predictions_df.to_csv(OUTPUT_DIR / "all_model_predictions.csv", index=False)
    importance_df.to_csv(OUTPUT_DIR / "feature_importance_summary.csv", index=False)
    redundancy_df.to_csv(OUTPUT_DIR / "redundant_feature_pairs.csv", index=False)
    calibration_summary_df.to_csv(OUTPUT_DIR / "calibration_summary.csv", index=False)
    reliability_df.to_csv(OUTPUT_DIR / "calibration_reliability_table.csv", index=False)
    top_experiment_permutation_df.to_csv(OUTPUT_DIR / "top_experiment_permutation_importance.csv", index=False)

    best_predictions_df = all_predictions_df[
        (all_predictions_df["model"] == best_row["model"])
        & (all_predictions_df["feature_subset"] == best_row["feature_subset"])
    ].copy()
    best_predictions_df.to_csv(OUTPUT_DIR / "best_model_predictions.csv", index=False)
    top_experiment_predictions_df = all_predictions_df.merge(
        top_experiment_rows[["model", "feature_subset"]],
        on=["model", "feature_subset"],
        how="inner",
    )
    top_experiment_predictions_df.to_csv(OUTPUT_DIR / "top_experiment_predictions.csv", index=False)

    logistic_importance.to_csv(OUTPUT_DIR / "logistic_coefficient_summary.csv", index=False)
    random_forest_importance.to_csv(OUTPUT_DIR / "random_forest_feature_importance.csv", index=False)
    if xgboost_importance is not None:
        xgboost_importance.to_csv(OUTPUT_DIR / "xgboost_feature_importance.csv", index=False)
    logreg_permutation.to_csv(OUTPUT_DIR / "logistic_permutation_importance_all_features.csv", index=False)
    if xgboost_permutation is not None:
        xgboost_permutation.to_csv(OUTPUT_DIR / "xgboost_permutation_importance_all_features.csv", index=False)

    model_descriptions = {
        model_name: describe_model_factory(model_factory)
        for model_name, model_factory in model_factories.items()
    }
    ledger_rows: list[dict[str, object]] = []
    for result_row in results_df.itertuples():
        ledger_rows.append(
            {
                "model": result_row.model,
                "feature_subset": result_row.feature_subset,
                "feature_count": int(result_row.feature_count),
                "parameters_json": json.dumps(model_descriptions[result_row.model]["parameters"], sort_keys=True),
                "accuracy": float(result_row.accuracy),
                "roc_auc": float(result_row.roc_auc),
                "log_loss": float(result_row.log_loss),
                "brier_score": float(result_row.brier_score),
                "fold_metrics_path": str(OUTPUT_DIR / "fold_level_metrics.csv"),
                "metrics_summary_path": str(OUTPUT_DIR / "metrics_summary.csv"),
                "all_predictions_path": str(OUTPUT_DIR / "all_model_predictions.csv"),
                "best_predictions_path": str(OUTPUT_DIR / "best_model_predictions.csv"),
                "feature_importance_path": str(OUTPUT_DIR / "feature_importance_summary.csv"),
                "report_path": str(OUTPUT_DIR / "model_comparison_report.md"),
            }
        )
    experiment_ledger_df = pd.DataFrame(ledger_rows)
    experiment_ledger_df.to_csv(OUTPUT_DIR / "experiment_ledger.csv", index=False)

    with (OUTPUT_DIR / "experiment_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_path": str(DATASET_PATH),
                "output_dir": str(OUTPUT_DIR),
                "target_column": TARGET_COLUMN,
                "chronology_column": "fight_order",
                "feature_count": len(feature_columns),
                "feature_subsets": group_map,
                "top_n_features": group_map["top_n_features"],
                "model_descriptions": model_descriptions,
                "best_configuration": json_ready_dict(best_row.to_dict()),
            },
            handle,
            indent=2,
        )

    (OUTPUT_DIR / "model_comparison_report.md").write_text(report_text, encoding="utf-8")

    print(report_text)
    print()
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
