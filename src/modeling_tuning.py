from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modeling_baseline import (
    DATASET_PATH,
    RANDOM_STATE,
    build_expanding_splits,
    describe_model_factory,
    evaluate_model_subset,
    load_dataset,
    metric_dict,
    split_train_calibration,
)
from modeling_pruning import CONSERVATIVE_DROP, REDUNDANT_DROP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling" / "tuning"
PRIMARY_MODEL = "random_forest"
SECONDARY_MODEL = "logistic_regression"


def build_pruned_combined_features(feature_columns: list[str]) -> list[str]:
    removed = set(CONSERVATIVE_DROP + REDUNDANT_DROP)
    return [feature for feature in feature_columns if feature not in removed]


def build_logistic_factory(
    *,
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "lbfgs",
    class_weight: str | None = None,
) -> Any:
    preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    classifier_kwargs: dict[str, Any] = {
        "C": C,
        "penalty": penalty,
        "solver": solver,
        "max_iter": 4000,
        "random_state": RANDOM_STATE,
        "class_weight": class_weight,
    }
    if solver == "liblinear":
        classifier_kwargs["n_jobs"] = 1
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(**classifier_kwargs)),
        ]
    )


def build_random_forest_factory(
    *,
    n_estimators: int = 500,
    max_depth: int | None = None,
    min_samples_leaf: int = 3,
    min_samples_split: int = 2,
    max_features: str | float = "sqrt",
) -> Any:
    preprocess = ColumnTransformer(
        transformers=[("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), slice(0, None))]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    min_samples_split=min_samples_split,
                    max_features=max_features,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )


def rf_search_space() -> list[dict[str, Any]]:
    return [
        {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 3, "min_samples_split": 2, "max_features": "sqrt"},
        {"n_estimators": 500, "max_depth": 16, "min_samples_leaf": 3, "min_samples_split": 2, "max_features": "sqrt"},
        {"n_estimators": 700, "max_depth": None, "min_samples_leaf": 3, "min_samples_split": 2, "max_features": "sqrt"},
        {"n_estimators": 500, "max_depth": 12, "min_samples_leaf": 5, "min_samples_split": 4, "max_features": "sqrt"},
        {"n_estimators": 700, "max_depth": 16, "min_samples_leaf": 5, "min_samples_split": 4, "max_features": 0.5},
        {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 8, "min_samples_split": 4, "max_features": "sqrt"},
    ]


def logit_search_space() -> list[dict[str, Any]]:
    return [
        {"C": 0.25, "penalty": "l2", "solver": "lbfgs", "class_weight": None},
        {"C": 0.5, "penalty": "l2", "solver": "lbfgs", "class_weight": None},
        {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "class_weight": None},
        {"C": 2.0, "penalty": "l2", "solver": "lbfgs", "class_weight": None},
        {"C": 0.5, "penalty": "l1", "solver": "liblinear", "class_weight": None},
        {"C": 1.0, "penalty": "l1", "solver": "liblinear", "class_weight": None},
    ]


def rank_results(results_df: pd.DataFrame) -> pd.DataFrame:
    return results_df.sort_values(
        ["log_loss", "brier_score", "roc_auc", "accuracy"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def calibration_experiments(
    df: pd.DataFrame,
    features: list[str],
    model_name: str,
    model_factory,
    experiment_name: str,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold_number, (train_index, test_index) in enumerate(splits, start=1):
        fit_index, calibration_index = split_train_calibration(train_index)
        x_fit = df.iloc[fit_index][features]
        y_fit = df.iloc[fit_index]["target_A_win"]
        x_cal = df.iloc[calibration_index][features]
        y_cal = df.iloc[calibration_index]["target_A_win"]
        x_test = df.iloc[test_index][features]
        y_test = df.iloc[test_index]["target_A_win"]

        base_model = model_factory()
        base_model.fit(x_fit, y_fit)
        cal_probabilities = base_model.predict_proba(x_cal)[:, 1]
        test_probabilities = base_model.predict_proba(x_test)[:, 1]

        sigmoid_calibrator = LogisticRegression(random_state=RANDOM_STATE)
        sigmoid_calibrator.fit(cal_probabilities.reshape(-1, 1), y_cal)
        sigmoid_probabilities = sigmoid_calibrator.predict_proba(test_probabilities.reshape(-1, 1))[:, 1]

        isotonic_calibrator = IsotonicRegression(out_of_bounds="clip")
        isotonic_calibrator.fit(cal_probabilities, y_cal)
        isotonic_probabilities = isotonic_calibrator.predict(test_probabilities)

        probability_sets = {
            "none": test_probabilities,
            "platt_sigmoid": sigmoid_probabilities,
            "isotonic": isotonic_probabilities,
        }
        for calibration_name, probabilities in probability_sets.items():
            metrics = metric_dict(y_test, probabilities)
            metric_rows.append(
                {
                    "experiment_name": experiment_name,
                    "model": model_name,
                    "calibration_method": calibration_name,
                    "fold": fold_number,
                    "train_size": int(len(train_index)),
                    "fit_size": int(len(fit_index)),
                    "calibration_size": int(len(calibration_index)),
                    "test_size": int(len(test_index)),
                    **metrics,
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "experiment_name": experiment_name,
                        "model": model_name,
                        "calibration_method": calibration_name,
                        "fold": fold_number,
                        "fight_id": df.iloc[test_index]["fight_id"].values,
                        "fight_order": df.iloc[test_index]["fight_order"].values,
                        "y_true": y_test.to_numpy(),
                        "predicted_probability": probabilities,
                    }
                )
            )

    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def summarize_calibration(calibration_fold_df: pd.DataFrame) -> pd.DataFrame:
    return (
        calibration_fold_df.groupby(["experiment_name", "model", "calibration_method"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            roc_auc=("roc_auc", "mean"),
            log_loss=("log_loss", "mean"),
            brier_score=("brier_score", "mean"),
        )
        .sort_values(["model", "log_loss", "brier_score", "roc_auc", "accuracy"], ascending=[True, True, True, False, False])
        .reset_index(drop=True)
    )


def report_text(
    baseline_summary: pd.DataFrame,
    tuning_summary: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    best_rf_tuned: pd.Series,
    best_logit_tuned: pd.Series,
    best_rf_calibrated: pd.Series,
    best_logit_calibrated: pd.Series,
    rf_grid: list[dict[str, Any]],
    logit_grid: list[dict[str, Any]],
) -> str:
    baseline_ranked = rank_results(baseline_summary[["model", "feature_subset", "accuracy", "roc_auc", "log_loss", "brier_score"]])
    tuning_ranked = rank_results(
        tuning_summary[["model", "feature_subset", "accuracy", "roc_auc", "log_loss", "brier_score", "experiment_name"]]
    )
    rf_baseline = baseline_summary[baseline_summary["model"] == PRIMARY_MODEL].iloc[0]
    logit_baseline = baseline_summary[baseline_summary["model"] == SECONDARY_MODEL].iloc[0]
    calibration_ranked = calibration_summary.sort_values(
        ["log_loss", "brier_score", "roc_auc", "accuracy"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)

    rf_tuning_gain = float(best_rf_tuned["log_loss"] - rf_baseline["log_loss"])
    rf_calibration_gain = float(best_rf_calibrated["log_loss"] - rf_baseline["log_loss"])
    logit_tuning_gain = float(best_logit_tuned["log_loss"] - logit_baseline["log_loss"])
    logit_calibration_gain = float(best_logit_calibrated["log_loss"] - logit_baseline["log_loss"])

    lines = [
        "# Phase 2.75 Model Tuning Report",
        "",
        "## Baseline Models",
        "- Primary: `random_forest + pruned_combined`",
        "- Secondary: `logistic_regression + pruned_combined`",
        f"- Dataset: `{DATASET_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "## Validation",
        "- Same expanding-window walk-forward validation as prior Phase 2 runs.",
        "- Same `pruned_combined` feature subset across all tuning and calibration experiments.",
        "",
        "## Tuning Search Space",
        f"- Random forest grid: {json.dumps(rf_grid)}",
        f"- Logistic regression grid: {json.dumps(logit_grid)}",
        "",
        "## Baseline Anchors",
        baseline_ranked.to_string(index=False),
        "",
        "## Tuning Results",
        tuning_ranked.to_string(index=False),
        "",
        "## Calibration Results",
        calibration_ranked.to_string(index=False),
        "",
        "## Best RF",
        (
            f"- Best tuned RF: `{best_rf_tuned['experiment_name']}` "
            f"(log_loss={best_rf_tuned['log_loss']:.4f}, brier={best_rf_tuned['brier_score']:.4f}, "
            f"roc_auc={best_rf_tuned['roc_auc']:.4f}, accuracy={best_rf_tuned['accuracy']:.4f})"
        ),
        (
            f"- Best calibrated RF: `{best_rf_calibrated['calibration_method']}` on `{best_rf_calibrated['experiment_name']}` "
            f"(log_loss={best_rf_calibrated['log_loss']:.4f}, brier={best_rf_calibrated['brier_score']:.4f})"
        ),
        f"- RF tuning delta vs untuned baseline log_loss: {rf_tuning_gain:+.4f}",
        f"- RF calibration delta vs untuned baseline log_loss: {rf_calibration_gain:+.4f}",
        "",
        "## Best Logistic",
        (
            f"- Best tuned logistic: `{best_logit_tuned['experiment_name']}` "
            f"(log_loss={best_logit_tuned['log_loss']:.4f}, brier={best_logit_tuned['brier_score']:.4f}, "
            f"roc_auc={best_logit_tuned['roc_auc']:.4f}, accuracy={best_logit_tuned['accuracy']:.4f})"
        ),
        (
            f"- Best calibrated logistic: `{best_logit_calibrated['calibration_method']}` on `{best_logit_calibrated['experiment_name']}` "
            f"(log_loss={best_logit_calibrated['log_loss']:.4f}, brier={best_logit_calibrated['brier_score']:.4f})"
        ),
        f"- Logistic tuning delta vs untuned baseline log_loss: {logit_tuning_gain:+.4f}",
        f"- Logistic calibration delta vs untuned baseline log_loss: {logit_calibration_gain:+.4f}",
        "",
        "## Interpretation",
        (
            "- Calibration helped more than parameter tuning for random forest."
            if rf_calibration_gain < rf_tuning_gain
            else "- Parameter tuning helped at least as much as calibration for random forest."
        ),
        (
            "- Calibration helped more than parameter tuning for logistic regression."
            if logit_calibration_gain < logit_tuning_gain
            else "- Parameter tuning helped at least as much as calibration for logistic regression."
        ),
        (
            f"- Final recommended production model: `{PRIMARY_MODEL}` with "
            f"`{best_rf_calibrated['calibration_method']}` calibration."
            if best_rf_calibrated["log_loss"] <= best_rf_tuned["log_loss"]
            else f"- Final recommended production model: `{PRIMARY_MODEL}` tuned but uncalibrated."
        ),
        (
            f"- Final recommended secondary benchmark model: `{SECONDARY_MODEL}` with "
            f"`{best_logit_calibrated['calibration_method']}` calibration."
            if best_logit_calibrated["log_loss"] <= best_logit_tuned["log_loss"]
            else f"- Final recommended secondary benchmark model: `{SECONDARY_MODEL}` tuned but uncalibrated."
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, feature_columns = load_dataset()
    active_features = build_pruned_combined_features(feature_columns)
    splits = build_expanding_splits(df)

    baseline_factories = {
        PRIMARY_MODEL: lambda: build_random_forest_factory(),
        SECONDARY_MODEL: lambda: build_logistic_factory(),
    }
    rf_grid = rf_search_space()
    logit_grid = logit_search_space()

    summary_rows: list[dict[str, Any]] = []
    fold_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    ledger_rows: list[dict[str, Any]] = []

    baseline_summaries: list[dict[str, Any]] = []
    baseline_folds: list[pd.DataFrame] = []
    baseline_predictions: list[pd.DataFrame] = []
    for model_name, model_factory in baseline_factories.items():
        summary, predictions, fold_metrics = evaluate_model_subset(
            df=df,
            feature_subset_name="pruned_combined",
            feature_subset=active_features,
            model_name=model_name,
            model_factory=model_factory,
            splits=splits,
        )
        summary["experiment_name"] = f"{model_name}__untuned_baseline"
        summary["parameter_json"] = json.dumps(describe_model_factory(model_factory)["parameters"], sort_keys=True)
        fold_metrics["experiment_name"] = f"{model_name}__untuned_baseline"
        fold_metrics["parameter_json"] = summary["parameter_json"]
        predictions["experiment_name"] = f"{model_name}__untuned_baseline"
        predictions["calibration_method"] = "none"
        baseline_summaries.append(summary)
        baseline_folds.append(fold_metrics)
        baseline_predictions.append(predictions)

    baseline_summary_df = pd.DataFrame(baseline_summaries)
    baseline_fold_df = pd.concat(baseline_folds, ignore_index=True)
    baseline_prediction_df = pd.concat(baseline_predictions, ignore_index=True)

    summary_rows.extend(baseline_summary_df.to_dict("records"))
    fold_frames.append(baseline_fold_df)
    prediction_frames.append(baseline_prediction_df)

    rf_results: list[dict[str, Any]] = []
    for idx, params in enumerate(rf_grid, start=1):
        experiment_name = f"random_forest__tuned_{idx}"
        model_factory = lambda params=params: build_random_forest_factory(**params)
        summary, predictions, fold_metrics = evaluate_model_subset(
            df=df,
            feature_subset_name="pruned_combined",
            feature_subset=active_features,
            model_name=PRIMARY_MODEL,
            model_factory=model_factory,
            splits=splits,
        )
        summary["experiment_name"] = experiment_name
        summary["parameter_json"] = json.dumps(params, sort_keys=True)
        fold_metrics["experiment_name"] = experiment_name
        fold_metrics["parameter_json"] = json.dumps(params, sort_keys=True)
        predictions["experiment_name"] = experiment_name
        predictions["calibration_method"] = "none"
        rf_results.append(summary)
        fold_frames.append(fold_metrics)
        prediction_frames.append(predictions)

    logit_results: list[dict[str, Any]] = []
    for idx, params in enumerate(logit_grid, start=1):
        experiment_name = f"logistic_regression__tuned_{idx}"
        model_factory = lambda params=params: build_logistic_factory(**params)
        summary, predictions, fold_metrics = evaluate_model_subset(
            df=df,
            feature_subset_name="pruned_combined",
            feature_subset=active_features,
            model_name=SECONDARY_MODEL,
            model_factory=model_factory,
            splits=splits,
        )
        summary["experiment_name"] = experiment_name
        summary["parameter_json"] = json.dumps(params, sort_keys=True)
        fold_metrics["experiment_name"] = experiment_name
        fold_metrics["parameter_json"] = json.dumps(params, sort_keys=True)
        predictions["experiment_name"] = experiment_name
        predictions["calibration_method"] = "none"
        logit_results.append(summary)
        fold_frames.append(fold_metrics)
        prediction_frames.append(predictions)

    tuning_summary_df = pd.DataFrame(rf_results + logit_results)
    ranked_tuning_df = rank_results(tuning_summary_df)
    best_rf_tuned = rank_results(tuning_summary_df[tuning_summary_df["model"] == PRIMARY_MODEL]).iloc[0]
    best_logit_tuned = rank_results(tuning_summary_df[tuning_summary_df["model"] == SECONDARY_MODEL]).iloc[0]

    calibration_fold_frames: list[pd.DataFrame] = []
    calibration_prediction_frames: list[pd.DataFrame] = []

    best_rf_params = json.loads(best_rf_tuned["parameter_json"])
    best_rf_factory = lambda: build_random_forest_factory(**best_rf_params)
    rf_cal_fold_df, rf_cal_predictions = calibration_experiments(
        df=df,
        features=active_features,
        model_name=PRIMARY_MODEL,
        model_factory=best_rf_factory,
        experiment_name=str(best_rf_tuned["experiment_name"]),
        splits=splits,
    )
    calibration_fold_frames.append(rf_cal_fold_df)
    calibration_prediction_frames.append(rf_cal_predictions)

    best_logit_params = json.loads(best_logit_tuned["parameter_json"])
    best_logit_factory = lambda: build_logistic_factory(**best_logit_params)
    logit_cal_fold_df, logit_cal_predictions = calibration_experiments(
        df=df,
        features=active_features,
        model_name=SECONDARY_MODEL,
        model_factory=best_logit_factory,
        experiment_name=str(best_logit_tuned["experiment_name"]),
        splits=splits,
    )
    calibration_fold_frames.append(logit_cal_fold_df)
    calibration_prediction_frames.append(logit_cal_predictions)

    calibration_fold_df = pd.concat(calibration_fold_frames, ignore_index=True)
    calibration_prediction_df = pd.concat(calibration_prediction_frames, ignore_index=True)
    calibration_summary_df = summarize_calibration(calibration_fold_df)

    best_rf_calibrated = calibration_summary_df[calibration_summary_df["model"] == PRIMARY_MODEL].iloc[0]
    best_logit_calibrated = calibration_summary_df[calibration_summary_df["model"] == SECONDARY_MODEL].iloc[0]

    metrics_summary_df = pd.concat(
        [
            baseline_summary_df,
            tuning_summary_df,
            calibration_summary_df.assign(feature_subset="pruned_combined", feature_count=len(active_features), parameter_json=""),
        ],
        ignore_index=True,
        sort=False,
    )
    fold_level_metrics_df = pd.concat([baseline_fold_df, *fold_frames[1:], calibration_fold_df], ignore_index=True, sort=False)
    all_predictions_df = pd.concat(prediction_frames + [calibration_prediction_df], ignore_index=True, sort=False)

    baseline_prediction_df[baseline_prediction_df["model"] == PRIMARY_MODEL].to_csv(
        OUTPUT_DIR / "baseline_predictions_rf.csv",
        index=False,
    )
    baseline_prediction_df[baseline_prediction_df["model"] == SECONDARY_MODEL].to_csv(
        OUTPUT_DIR / "baseline_predictions_logit.csv",
        index=False,
    )

    best_tuned_rf_predictions = all_predictions_df[
        (all_predictions_df["experiment_name"] == best_rf_tuned["experiment_name"])
        & (all_predictions_df["model"] == PRIMARY_MODEL)
        & (all_predictions_df["calibration_method"] == "none")
    ].copy()
    best_tuned_logit_predictions = all_predictions_df[
        (all_predictions_df["experiment_name"] == best_logit_tuned["experiment_name"])
        & (all_predictions_df["model"] == SECONDARY_MODEL)
        & (all_predictions_df["calibration_method"] == "none")
    ].copy()
    best_tuned_rf_predictions.to_csv(OUTPUT_DIR / "best_tuned_rf_predictions.csv", index=False)
    best_tuned_logit_predictions.to_csv(OUTPUT_DIR / "best_tuned_logit_predictions.csv", index=False)

    best_calibrated_rf_predictions = calibration_prediction_df[
        (calibration_prediction_df["experiment_name"] == best_rf_calibrated["experiment_name"])
        & (calibration_prediction_df["model"] == PRIMARY_MODEL)
        & (calibration_prediction_df["calibration_method"] == best_rf_calibrated["calibration_method"])
    ].copy()
    best_calibrated_logit_predictions = calibration_prediction_df[
        (calibration_prediction_df["experiment_name"] == best_logit_calibrated["experiment_name"])
        & (calibration_prediction_df["model"] == SECONDARY_MODEL)
        & (calibration_prediction_df["calibration_method"] == best_logit_calibrated["calibration_method"])
    ].copy()
    best_calibrated_rf_predictions.to_csv(OUTPUT_DIR / "best_calibrated_rf_predictions.csv", index=False)
    best_calibrated_logit_predictions.to_csv(OUTPUT_DIR / "best_calibrated_logit_predictions.csv", index=False)

    metrics_summary_df.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    fold_level_metrics_df.to_csv(OUTPUT_DIR / "fold_level_metrics.csv", index=False)

    best_params = {
        "random_forest": {
            "best_tuned_experiment": str(best_rf_tuned["experiment_name"]),
            "params": best_rf_params,
            "best_calibration_method": str(best_rf_calibrated["calibration_method"]),
        },
        "logistic_regression": {
            "best_tuned_experiment": str(best_logit_tuned["experiment_name"]),
            "params": best_logit_params,
            "best_calibration_method": str(best_logit_calibrated["calibration_method"]),
        },
    }
    with (OUTPUT_DIR / "best_params.json").open("w", encoding="utf-8") as handle:
        json.dump(best_params, handle, indent=2)

    for row in metrics_summary_df.itertuples():
        ledger_rows.append(
            {
                "experiment_name": getattr(row, "experiment_name", ""),
                "model": row.model,
                "feature_subset": getattr(row, "feature_subset", "pruned_combined"),
                "feature_count": int(getattr(row, "feature_count", len(active_features))),
                "parameter_json": getattr(row, "parameter_json", ""),
                "calibration_method": getattr(row, "calibration_method", "none"),
                "accuracy": float(row.accuracy),
                "roc_auc": float(row.roc_auc),
                "log_loss": float(row.log_loss),
                "brier_score": float(row.brier_score),
                "metrics_summary_path": str(OUTPUT_DIR / "metrics_summary.csv"),
                "fold_level_metrics_path": str(OUTPUT_DIR / "fold_level_metrics.csv"),
                "best_params_path": str(OUTPUT_DIR / "best_params.json"),
                "report_path": str(OUTPUT_DIR / "model_tuning_report.md"),
            }
        )
    pd.DataFrame(ledger_rows).to_csv(OUTPUT_DIR / "experiment_ledger.csv", index=False)

    report = report_text(
        baseline_summary=baseline_summary_df,
        tuning_summary=tuning_summary_df,
        calibration_summary=calibration_summary_df,
        best_rf_tuned=best_rf_tuned,
        best_logit_tuned=best_logit_tuned,
        best_rf_calibrated=best_rf_calibrated,
        best_logit_calibrated=best_logit_calibrated,
        rf_grid=rf_grid,
        logit_grid=logit_grid,
    )
    (OUTPUT_DIR / "model_tuning_report.md").write_text(report, encoding="utf-8")

    print(report)
    print()
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
