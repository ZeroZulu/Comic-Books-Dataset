"""
models.py — ML Training, Evaluation & Comparison
==================================================
Mirrors notebook Cell 28 exactly.

Key design decisions (matching the notebook):
  • OrdinalEncoder fitted ONLY on training data — no data leakage
  • 5-fold cross-validation scores reported alongside test metrics
  • 4 models: Linear, Ridge, Random Forest, Gradient Boosting
  • Low R² (~0.03–0.05) is expected and explained in output

Usage
-----
    from src.models import train_and_evaluate, best_model
    results = train_and_evaluate(df_clean)
    name, info = best_model(results)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ── Feature column definitions ────────────────────────────────────────────────
NUMERIC_FEATURES     = ["pages", "year", "volume_count"]
CATEGORICAL_FEATURES = [
    "publisher", "genre", "format", "country",
    "language", "age_rating", "status", "theme",
]

# ── Default model zoo ─────────────────────────────────────────────────────────
DEFAULT_MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=1.0),
    "Random Forest":     RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=5, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
    ),
}


def prepare_features(
    df: pd.DataFrame,
    target: str = "rating",
    numeric_features: list[str] | None = None,
    cat_features: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Split data and encode categoricals with OrdinalEncoder.
    Encoder is fit ONLY on the training split to prevent leakage.

    Returns
    -------
    X_train, X_test, y_train, y_test, all_feature_names
    """
    if numeric_features is None:
        numeric_features = [c for c in NUMERIC_FEATURES if c in df.columns]
    if cat_features is None:
        cat_features = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    ml_df = df[numeric_features + cat_features + [target]].dropna().copy()

    X_raw = ml_df[numeric_features + cat_features]
    y     = ml_df[target].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state
    )

    X_train = X_train_raw.copy()
    X_test  = X_test_raw.copy()

    for col in cat_features:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[col] = enc.fit_transform(X_train_raw[[col]])
        X_test[col]  = enc.transform(X_test_raw[[col]])

    all_features = numeric_features + cat_features
    return X_train, X_test, y_train, y_test, all_features


def train_and_evaluate(
    df: pd.DataFrame,
    target: str = "rating",
    models: dict | None = None,
    cv_folds: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Train multiple models and return a results dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (output of clean_pipeline).
    target : str
        Target column name. Default 'rating'.
    models : dict, optional
        Dict of {name: sklearn_model}. Defaults to DEFAULT_MODELS.
    cv_folds : int
        Number of cross-validation folds. Default 5.
    verbose : bool
        Print progress table. Default True.

    Returns
    -------
    dict
        {model_name: {RMSE, MAE, R2, CV_R2, model, preds, y_test}}

    Notes
    -----
    R² ≈ 0.03–0.05 is expected for this dataset — ratings are not
    predictable from tabular metadata alone (likely synthetic scores).
    Feature importance is still valid for ranking predictor relevance.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    if models is None:
        models = DEFAULT_MODELS

    X_train, X_test, y_train, y_test, all_features = prepare_features(df, target)

    if verbose:
        print(f"Training on {len(X_train) + len(X_test):,} rows "
              f"({len(X_train):,} train / {len(X_test):,} test)")
        print(f"Features: {all_features}")
        print(f"\n{'Model':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'CV-R² (5-fold)':>16}")
        print("─" * 70)

    results: dict = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        cv_r2 = cross_val_score(model, X_train, y_train,
                                cv=cv_folds, scoring="r2").mean()
        results[name] = {
            "RMSE":    np.sqrt(mean_squared_error(y_test, preds)),
            "MAE":     mean_absolute_error(y_test, preds),
            "R2":      r2_score(y_test, preds),
            "CV_R2":   cv_r2,
            "model":   model,
            "preds":   preds,
            "y_test":  y_test,
            "features": all_features,
        }
        if verbose:
            r = results[name]
            print(f"{name:<25} {r['RMSE']:>8.4f} {r['MAE']:>8.4f} "
                  f"{r['R2']:>8.4f} {cv_r2:>16.4f}")

    if verbose and max(r["R2"] for r in results.values()) < 0.15:
        print("\n⚠️  Note: R² < 0.15 across all models. This typically means:")
        print("   • Ratings are independently assigned — not correlated with metadata")
        print("   • Quality depends on story/art which tabular features cannot capture")
        print("   • Possible synthetic/random ratings in this dataset")
        print("   → Feature importance is still valid for ranking predictor relevance.")

    return results


def best_model(results: dict) -> tuple[str, dict]:
    """
    Return the name and result-dict of the best model by test R².

    Returns
    -------
    (model_name, result_dict)
    """
    name = max(results, key=lambda k: results[k]["R2"])
    return name, results[name]


def feature_importance_df(model_name: str, result: dict) -> pd.Series:
    """
    Return a sorted Series of feature importances or |coefficients|.
    Works for tree-based models and linear models.
    """
    model    = result["model"]
    features = result["features"]

    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=features).sort_values()
    elif hasattr(model, "coef_"):
        return pd.Series(np.abs(model.coef_), index=features).sort_values()
    else:
        raise ValueError(f"Model '{model_name}' has no feature_importances_ or coef_.")
