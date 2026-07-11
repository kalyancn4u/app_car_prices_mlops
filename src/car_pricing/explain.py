"""Explainability — *why* does the model predict what it predicts?

A model that says "₹6.5 Lakhs" is more trustworthy if it can also say *which
inputs mattered*. Two simple, dependency-free views:

  * **Built-in importances** — how often the model used each feature to split.
  * **Permutation importance** — shuffle one feature's values and see how much
    worse the predictions get; the bigger the drop, the more that feature matters.
    (This one is model-agnostic and more honest.)

**SHAP** is the richer upgrade — it explains *individual* predictions, not just
the model overall. See docs/MLOPS_GUIDE.md.
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from . import config, data, features


def _load_pipeline(pipeline=None):
    return pipeline if pipeline is not None else joblib.load(config.PIPELINE_PATH)


def builtin_importances(pipeline=None) -> pd.Series:
    """The fitted model's own feature_importances_ (aligned to config.FEATURES)."""
    model = _load_pipeline(pipeline).named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(f"{type(model).__name__} has no feature_importances_")
    return (pd.Series(model.feature_importances_, index=config.FEATURES)
            .sort_values(ascending=False))


def permutation_importances(pipeline=None, n_repeats: int = 5) -> pd.Series:
    """How much test MAE grows when each raw input column is shuffled."""
    df = data.clean(data.load_raw())
    X, y = features.split_xy(df)
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    result = permutation_importance(
        _load_pipeline(pipeline), X_te, y_te, n_repeats=n_repeats,
        random_state=config.RANDOM_STATE, scoring="neg_mean_absolute_error", n_jobs=1)
    return (pd.Series(result.importances_mean, index=config.FEATURES)
            .sort_values(ascending=False))


if __name__ == "__main__":  # `python -m car_pricing.explain`
    print("Top features (built-in importance):")
    print(builtin_importances().head(8).round(4))
