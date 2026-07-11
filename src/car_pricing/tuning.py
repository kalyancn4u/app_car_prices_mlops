"""Hyperparameter tuning — let the computer search for better model settings.

Plain idea: a model has "knobs" (how many trees, how deep, learning rate…).
Instead of guessing, we try many random combinations, score each with
cross-validation, and keep the best. That's what `RandomizedSearchCV` does.

Dependency-free (scikit-learn + scipy). **Optuna** is the smarter, popular
upgrade (it learns which knob-settings to try next) — see docs/MLOPS_GUIDE.md.
Every search is recorded via the experiment tracker (tracking.py).
"""

from __future__ import annotations

from typing import Dict

from scipy.stats import randint, uniform
from sklearn.model_selection import KFold, RandomizedSearchCV

from . import config, data, features, models, tracking
from .pipeline import make_pipeline

# A small search space per model. Keys are `model__<param>` because the estimator
# is the "model" step inside the Pipeline.
SEARCH_SPACES: Dict[str, dict] = {
    "Random Forest": {
        "model__n_estimators": randint(150, 400),
        "model__max_depth": randint(8, 24),
        "model__min_samples_leaf": randint(2, 12),
    },
    "HistGradientBoosting": {
        "model__max_iter": randint(200, 600),
        "model__learning_rate": uniform(0.02, 0.15),
        "model__max_depth": randint(4, 12),
    },
    "LightGBM": {
        "model__n_estimators": randint(300, 900),
        "model__num_leaves": randint(15, 127),
        "model__learning_rate": uniform(0.01, 0.12),
        "model__subsample": uniform(0.7, 0.3),
    },
    "XGBoost": {
        "model__n_estimators": randint(300, 900),
        "model__max_depth": randint(4, 10),
        "model__learning_rate": uniform(0.01, 0.12),
        "model__subsample": uniform(0.7, 0.3),
    },
}


def tune(model_name: str = "LightGBM", n_iter: int = 15) -> Dict:
    """Randomly search `n_iter` settings for `model_name`; return the best.

    Returns a dict with `best_params`, `best_cv_mae` (Lakhs) and the baseline MAE
    for comparison. The run is logged via tracking.log_run.
    """
    if model_name not in SEARCH_SPACES:
        raise ValueError(f"No search space for {model_name!r}. "
                         f"Options: {list(SEARCH_SPACES)}")

    df = data.clean(data.load_raw())
    X, y = features.split_xy(df)

    pipe = make_pipeline(models.model_zoo()[model_name])
    search = RandomizedSearchCV(
        pipe, SEARCH_SPACES[model_name], n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE),
        random_state=config.RANDOM_STATE, n_jobs=1, refit=True,
    )
    search.fit(X, y)

    best_mae = float(-search.best_score_)
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}

    result = {
        "model": model_name,
        "best_cv_mae_lakhs": round(best_mae, 3),
        "best_params": best_params,
        "n_iter": n_iter,
    }
    tracking.log_run(
        run_name=f"tune-{model_name}",
        params={"model": model_name, "n_iter": n_iter, **best_params},
        metrics={"cv_mae_lakhs": best_mae},
        tags={"stage": "tuning"},
    )
    return result


if __name__ == "__main__":  # `python -m car_pricing.tuning`
    res = tune("LightGBM", n_iter=10)
    print(f"Best CV MAE for {res['model']}: Rs {res['best_cv_mae_lakhs']} Lakhs")
    print("Best params:", res["best_params"])
