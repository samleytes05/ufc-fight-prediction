from __future__ import annotations

"""Shared model builders for the Phase 2 baseline pipeline."""

from collections.abc import Callable

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


RANDOM_STATE = 42


def build_model_factories() -> dict[str, Callable[[], Pipeline]]:
    """Return baseline model factories for the Phase 2 benchmark suite."""
    numeric_preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    passthrough_preprocess = ColumnTransformer(
        transformers=[("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), slice(0, None))]
    )

    factories: dict[str, Callable[[], Pipeline]] = {
        "logistic_regression": lambda: Pipeline(
            steps=[
                ("preprocess", numeric_preprocess),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "random_forest": lambda: Pipeline(
            steps=[
                ("preprocess", passthrough_preprocess),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=3,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }
    if XGBClassifier is not None:
        factories["xgboost"] = lambda: Pipeline(
            steps=[
                ("preprocess", passthrough_preprocess),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    return factories
